#!/usr/bin/env python
"""Minimum detectable volume change, as a function of view count and arc.

    python scripts/mdvc.py --tracks A32 A16 B15 --gains 0.05 0.1 0.25 0.5

Volume Doubling Time needs a *change* in volume to be measurable and LIDC has no
follow-up scans, so growth is synthesised on the ground truth, the projections
are re-simulated, and the same measurement is applied to both reconstructions.
The output curve says where absolute volumetry stops being trustworthy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from medlift3d.baselines import cgls, sirt_tv
from medlift3d.datasets import (default_dirs, geometry_from_dict, load_case,
                                load_split, simulate, verify_split)
from medlift3d.diffusion import GaussianDiffusion
from medlift3d.metrics import measure_nodule_volume, relative_volume_change
from medlift3d.phantom import grow_nodule, lung_mask
from medlift3d.projector import Projector
from medlift3d.solver import DiffusionSolver, SolverConfig
from medlift3d.utils import load_checkpoint, pick_device, seed_everything


def reconstruct(method, projs, proj, diffusion, args):
    if method == "diffusion":
        s = DiffusionSolver(diffusion, proj, SolverConfig(
            n_steps=args.n_steps, n_posterior=args.n_posterior,
            slice_batch=args.slice_batch, progress=False))
        return s.reconstruct(projs, seed=args.seed)["mean"]
    if method == "sirt_tv":
        return sirt_tv(projs, proj, args.n_iter)
    if method == "cgls":
        return cgls(projs, proj, min(args.n_iter, 30))
    raise ValueError(method)


def main():
    d = default_dirs()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=d["data"] / "phantom")
    ap.add_argument("--out", type=Path, default=d["out"] / "mdvc")
    ap.add_argument("--tracks", nargs="*", default=["A32", "A16", "B15"])
    ap.add_argument("--methods", nargs="*", default=["sirt_tv"])
    ap.add_argument("--gains", type=float, nargs="*",
                    default=[0.05, 0.10, 0.20, 0.30, 0.50])
    ap.add_argument("--prior", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=5, help="cases to use")
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-posterior", type=int, default=4)
    ap.add_argument("--slice-batch", type=int, default=16)
    ap.add_argument("--i0", type=float, default=1e5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    diffusion = None
    if "diffusion" in args.methods:
        if args.prior is None:
            ap.error("--methods diffusion requires --prior")
        diffusion = GaussianDiffusion.from_checkpoint(
            load_checkpoint(args.prior, map_location=device)).to(device).eval()

    split = load_split(args.data / "splits.csv")
    verify_split(split)
    case_ids = split["test"][:args.limit]
    print(f"device={device}  cases={len(case_ids)}  tracks={args.tracks}  "
          f"methods={args.methods}  gains={args.gains}")

    rows = []
    for cid in tqdm(case_ids, desc="cases"):
        case = load_case(args.data / f"{cid}.npz")
        grid, mu0 = case["grid"], case["mu"]
        region = lung_mask(mu0, grid)
        nodules = case["meta"].get("nodules", [])
        if not nodules:
            continue
        nod = max(nodules, key=lambda n: n["radius_mm"])

        for track in args.tracks:
            if track not in case["projections"]:
                continue
            geom = geometry_from_dict(case["projections"][track]["geometry"])
            proj = Projector(grid, geom, device=device)

            # Baseline and follow-ups are simulated and reconstructed
            # identically, so the measured change reflects growth, not protocol.
            for method in args.methods:
                vols = {}
                for gain in [0.0] + list(args.gains):
                    grown, _, rec = grow_nodule(grid, mu0, nod, gain)
                    p = torch.from_numpy(
                        simulate(grown, grid, geom, i0=(args.i0 or None),
                                 seed=args.seed, device=device)).to(device)
                    r = reconstruct(method, p, proj, diffusion, args)
                    v = measure_nodule_volume(r.cpu().numpy(), nod["centre_xyz"],
                                              grid, region=region)
                    vols[gain] = (v, rec["true_volume_mm3"])

                v0, t0 = vols[0.0]
                for gain in args.gains:
                    v1, t1 = vols[gain]
                    rows.append({
                        "case_id": cid, "track": track, "method": method,
                        "gain_pct": gain * 100,
                        "true_change_pct": relative_volume_change(t0, t1),
                        "measured_change_pct": (relative_volume_change(v0, v1)
                                                if v0 > 0 else np.nan),
                        "v0_mm3": v0, "v1_mm3": v1,
                        "nodule_diameter_mm": 2 * nod["radius_mm"],
                        "detected": bool(v0 > 0 and v1 > 0),
                    })

    if not rows:
        raise SystemExit("no measurements produced; check --tracks against the dataset")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "mdvc_raw.csv", index=False)

    df["error_pp"] = (df["measured_change_pct"] - df["true_change_pct"]).abs()
    summary = (df.groupby(["method", "track", "gain_pct"])
                 .agg(measured_mean=("measured_change_pct", "mean"),
                      measured_std=("measured_change_pct", "std"),
                      abs_error_pp=("error_pp", "mean"),
                      detected=("detected", "mean"))
                 .round(2))
    summary.to_csv(args.out / "mdvc_summary.csv")
    print("\n=== measured vs true volume change ===")
    print(summary.to_string())

    # MDVC: the smallest true change whose mean measurement is separated from
    # the method's own noise floor (mean error > 1 sigma is not resolvable).
    print("\n=== minimum detectable volume change ===")
    for (m, tr), sub in df.groupby(["method", "track"]):
        mdvc = None
        for gain, s in sub.groupby("gain_pct"):
            sd = s["measured_change_pct"].std(ddof=0)
            if np.isfinite(sd) and abs(s["measured_change_pct"].mean()) > 2 * max(sd, 1e-6):
                mdvc = gain
                break
        print(f"  {m:10s} {tr:5s}  MDVC = "
              + (f"{mdvc:.0f}% volume change" if mdvc else "not resolved in tested range"))

    _plot(df, args.out / "mdvc.png")
    print(f"\nwrote {args.out}")


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (m, tr), sub in df.groupby(["method", "track"]):
        g = sub.groupby("gain_pct")["measured_change_pct"]
        ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().values,
                    marker="o", capsize=3, lw=1.3, label=f"{m} · {tr}")
    lim = df["true_change_pct"].max() * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=0.9, label="ideal")
    ax.set_xlabel("true volume change (%)")
    ax.set_ylabel("measured volume change (%)")
    ax.set_title("Minimum detectable volume change")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
