#!/usr/bin/env python
"""Measure this GPU's throughput on the real operations, then price the full run.

Run this FIRST on a rented box, before launching anything long. It costs a few
minutes and it turns "how long will this take and what will it cost" from a
guess into a measurement -- which matters when the meter is running.

    python scripts/gpu_benchmark.py --rate 0.40

Reports per-operation timings, peak VRAM, and an end-to-end estimate for the
full experimental programme at the configured grid size.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from medlift3d.diffusion import DiffusionConfig, GaussianDiffusion
from medlift3d.geometry import Grid
from medlift3d.prior2d import UNet2D, UNetConfig
from medlift3d.projector import ParallelGeometry, Projector
from medlift3d.utils import pick_device


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=1, repeats=3):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / repeats


def peak_mb(device):
    if device.type != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256])
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--chunk-rays", type=int, default=65536)
    ap.add_argument("--batch-size", type=int, default=8, help="prior training batch")
    ap.add_argument("--slice-res", type=int, default=256)
    ap.add_argument("--base-dim", type=int, default=64)
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="disable mixed precision in the training-step timing")
    ap.add_argument("--rate", type=float, default=None, help="instance cost, $/hr")
    # Programme size, for the end-to-end estimate.
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--slices-per-case", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--dc-steps", type=int, default=2)
    ap.add_argument("--n-posterior", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"gpu   : {p.name}  {p.total_memory/2**30:.1f} GiB  "
              f"sm_{p.major}{p.minor}  {torch.cuda.device_count()} visible")
    print(f"torch : {torch.__version__}\n")

    grid = Grid.centred(tuple(args.shape), tuple(args.spacing))
    geom = ParallelGeometry.covering(grid, args.views, 180.0)
    proj = Projector(grid, geom, chunk_rays=args.chunk_rays, device=device)
    print(f"grid  : {grid.shape} @ {grid.spacing} mm")
    print(f"geom  : {args.views} views, detector {geom.det_shape}, "
          f"{proj.n_rays:,} rays x {proj.n_samples} samples "
          f"= {proj.n_rays*proj.n_samples/1e9:.2f}G sample-evals per fp\n")

    mu = torch.rand(grid.shape, device=device)
    projs = proj.fp(mu)

    print("=== projector ===")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_fp = timeit(lambda: proj.fp(mu), device)
    fp_mb = peak_mb(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_bp = timeit(lambda: proj.bp(projs), device)
    bp_mb = peak_mb(device)
    print(f"  fp : {t_fp*1000:8.1f} ms   peak {fp_mb:7.0f} MB")
    print(f"  bp : {t_bp*1000:8.1f} ms   peak {bp_mb:7.0f} MB")

    # Which adjoint is faster is hardware-dependent -- measure, do not assume.
    bm = proj.benchmark_bp()
    print(f"  adjoint impl: vjp {bm['vjp']*1000:.0f} ms vs scatter "
          f"{bm['scatter']*1000:.0f} ms -> use '{bm['faster']}'")
    if bm["faster"] != proj.bp_impl:
        slow, fast = max(bm["vjp"], bm["scatter"]), min(bm["vjp"], bm["scatter"])
        print(f"  NOTE: default is '{proj.bp_impl}'. Pass bp_impl='{bm['faster']}' "
              f"to Projector for ~{slow/max(fast, 1e-9):.1f}x.")

    print("\n=== prior (2D UNet) ===")
    net = UNet2D(UNetConfig(base_dim=args.base_dim)).to(device)
    diff = GaussianDiffusion(net, DiffusionConfig()).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=(args.amp and device.type == "cuda"))
    x = torch.randn(args.batch_size, 1, args.slice_res, args.slice_res, device=device)
    print(f"  params: {net.n_params/1e6:.1f}M   batch {args.batch_size} "
          f"@ {args.slice_res}^2   amp={args.amp and device.type=='cuda'}")

    def train_step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=(args.amp and device.type == "cuda")):
            loss = diff.loss(x, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    diff.train()
    t_step = timeit(train_step, device, warmup=3, repeats=10)
    train_mb = peak_mb(device)
    print(f"  train step: {t_step*1000:7.1f} ms   {args.batch_size/t_step:6.1f} slices/s"
          f"   peak {train_mb:7.0f} MB")

    diff.eval()
    with torch.no_grad():
        t_inf = timeit(lambda: diff.predict_eps(
            x, torch.full((x.shape[0],), 500, device=device, dtype=torch.long), x),
            device, repeats=5)
    print(f"  infer step: {t_inf*1000:7.1f} ms   {args.batch_size/t_inf:6.1f} slices/s")

    # ---------------- end-to-end estimate ----------------
    print("\n=== full programme estimate ===")
    nz = grid.shape[0]

    total_slices = args.n_cases * args.slices_per_case
    steps_per_epoch = total_slices / args.batch_size
    h_train = steps_per_epoch * args.epochs * t_step / 3600

    # Simulation: one fp per (case, track). Assume 3 tracks.
    h_sim = args.n_cases * 3 * t_fp / 3600

    # Solver per case: K samples x steps x [prior over all slices + DC (fp+bp)].
    prior_per_step = (nz / args.batch_size) * t_inf
    dc_per_step = args.dc_steps * (t_fp + t_bp)
    per_case = args.n_posterior * args.n_steps * (prior_per_step + dc_per_step)
    h_recon = args.n_test * per_case / 3600

    # Baselines: ~60 iterations x (fp+bp), x4 methods.
    h_base = args.n_test * 4 * 60 * (t_fp + t_bp) / 3600
    # Experiments re-reconstruct many variants; scale off the solver cost.
    h_exp = 3 * h_recon

    rows = [("phantom generation", h_sim),
            ("prior training", h_train),
            ("baselines (4 methods)", h_base),
            ("diffusion reconstruction", h_recon),
            ("experiments (mdvc/halluc/ablation)", h_exp)]
    total = sum(h for _, h in rows)
    for name, h in rows:
        line = f"  {name:36s} {h:7.2f} h"
        if args.rate:
            line += f"   ${h*args.rate:6.2f}"
        print(line)
    print(f"  {'-'*36} {'-'*7}")
    line = f"  {'TOTAL':36s} {total:7.2f} h"
    if args.rate:
        line += f"   ${total*args.rate:6.2f}"
    print(line)

    peaks = [v for v in (fp_mb, bp_mb, train_mb) if v == v]   # drop NaN (CPU)
    if peaks:
        print(f"\n  peak VRAM observed: {max(peaks):.0f} MB")
    else:
        print("\n  peak VRAM: not measurable on CPU -- rerun on the GPU box")
    print(f"  solver cost per case: {per_case/60:.1f} min "
          f"(K={args.n_posterior} x {args.n_steps} steps)")
    if not args.rate:
        print("\n  pass --rate <$/hr> for a cost column.")
    print("\n  Levers if this is too slow or too expensive:")
    print("    --shape 192 192 192      ~2.4x less projector work")
    print("    --n-posterior 4          halves reconstruction and experiment time")
    print("    --n-steps 30             ~1.7x faster solving")
    print("    larger --batch-size      better GPU utilisation if VRAM allows")


if __name__ == "__main__":
    main()
