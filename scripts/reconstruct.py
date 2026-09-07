#!/usr/bin/env python
"""Reconstruct test cases with any method, on one canonical grid.

    # classical baselines
    python scripts/reconstruct.py --track A16 --method sirt_tv
    # prior + data consistency, 8 posterior samples
    python scripts/reconstruct.py --track A16 --method diffusion \
        --prior runs/prior/best.pt --n-posterior 8

Every method writes to the same layout so `evaluate.py` can compare them without
special cases. Reconstructions live on the case's own grid, always -- comparing
volumes on different grids is how the FYDP-1 numbers became meaningless.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from tqdm.auto import tqdm

from medlift3d.baselines import cgls, fbp, sirt, sirt_tv
from medlift3d.datasets import (default_dirs, geometry_from_dict, load_case,
                                load_split, verify_split)
from medlift3d.diffusion import GaussianDiffusion
from medlift3d.export import save_nifti, save_triplanar_png, save_uncertainty
from medlift3d.projector import Projector
from medlift3d.solver import DiffusionSolver, SolverConfig
from medlift3d.utils import load_checkpoint, pick_device, seed_everything, write_json

METHODS = ("fbp", "sirt", "sirt_tv", "cgls", "diffusion")


def main():
    d = default_dirs()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=d["data"] / "phantom")
    ap.add_argument("--out", type=Path, default=d["out"] / "recon")
    ap.add_argument("--track", default="A16", help="projection track key, e.g. A16 or B15")
    ap.add_argument("--method", default="sirt_tv", choices=METHODS)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--cases", nargs="*", default=None, help="explicit case ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prior", type=Path, default=None, help="prior checkpoint (diffusion)")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--dc-steps", type=int, default=2)
    ap.add_argument("--dc-step", type=float, default=0.9)
    ap.add_argument("--lam-z", type=float, default=0.03)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--n-posterior", type=int, default=8)
    ap.add_argument("--slice-batch", type=int, default=16)
    ap.add_argument("--warm-start-t", type=float, default=0.7)
    ap.add_argument("--n-iter", type=int, default=60, help="iterations for classical methods")
    ap.add_argument("--chunk-rays", type=int, default=8192)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    tag = f"{args.method}_{args.track}"
    out_dir = args.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        case_ids = list(args.cases)
    else:
        split = load_split(args.data / "splits.csv")
        verify_split(split)
        case_ids = split[args.split]
    if args.limit:
        case_ids = case_ids[:args.limit]
    print(f"device={device}  method={args.method}  track={args.track}  "
          f"cases={len(case_ids)} ({args.split})")

    diffusion = None
    if args.method == "diffusion":
        if args.prior is None:
            ap.error("--method diffusion requires --prior")
        ck = load_checkpoint(args.prior, map_location=device)
        # Rebuilt from the checkpoint's own configs: no chance of passing
        # mismatched arguments by hand.
        diffusion = GaussianDiffusion.from_checkpoint(ck).to(device).eval()
        print(f"prior: {args.prior}  objective={diffusion.cfg.objective}  "
              f"timesteps={diffusion.cfg.timesteps}  "
              f"cond_ch={diffusion.model.cfg.cond_ch}")

    records = []
    for cid in tqdm(case_ids, desc="cases"):
        case = load_case(args.data / f"{cid}.npz")
        grid = case["grid"]
        if args.track not in case["projections"]:
            raise KeyError(f"{cid} has no track {args.track!r}; "
                           f"available: {list(case['projections'])}")
        rec = case["projections"][args.track]
        geom = geometry_from_dict(rec["geometry"])
        projs = torch.from_numpy(rec["projections"]).to(device)
        proj = Projector(grid, geom, chunk_rays=args.chunk_rays, device=device)

        std = None
        if args.method == "diffusion":
            solver = DiffusionSolver(diffusion, proj, SolverConfig(
                n_steps=args.n_steps, dc_steps=args.dc_steps, dc_step=args.dc_step,
                lam_z=args.lam_z, guidance=args.guidance, eta=args.eta,
                n_posterior=args.n_posterior, slice_batch=args.slice_batch,
                warm_start_t=args.warm_start_t, progress=False))
            res = solver.reconstruct(projs, seed=args.seed)
            mu, std, extra = res["mean"], res["std"], {
                "proj_rmse": res["proj_rmse"], "lipschitz": res["lipschitz"],
                "n_posterior": res["n_posterior"]}
        else:
            fn = {"fbp": lambda: fbp(projs, proj),
                  "sirt": lambda: sirt(projs, proj, args.n_iter),
                  "sirt_tv": lambda: sirt_tv(projs, proj, args.n_iter),
                  "cgls": lambda: cgls(projs, proj, min(args.n_iter, 30))}[args.method]
            mu = fn()
            with torch.no_grad():
                extra = {"proj_rmse": float((proj.fp(mu) - projs).pow(2).mean().sqrt())}

        mu_np = mu.detach().cpu().numpy().astype(np.float32)
        payload = {"mu": mu_np, "shape": np.asarray(grid.shape),
                   "spacing": np.asarray(grid.spacing), "origin": np.asarray(grid.origin)}
        if std is not None:
            payload["std"] = std.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(out_dir / f"{cid}.npz", **payload)
        save_nifti(out_dir / f"{cid}.nii.gz", mu_np, grid)
        if std is not None:
            save_uncertainty(out_dir / f"{cid}_uncertainty.nii.gz",
                             payload["std"], grid)
        if not args.no_png:
            save_triplanar_png(out_dir / f"{cid}.png", mu_np,
                               case.get("nodule_mask"), f"{cid} · {tag}")
        records.append({"case_id": cid, **extra})

    write_json(out_dir / "settings.json",
               {"args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()},
                "records": records})
    print(f"\nwrote {len(records)} reconstructions to {out_dir}")
    if records:
        print(f"mean projection RMSE: "
              f"{np.mean([r['proj_rmse'] for r in records]):.5g}")


if __name__ == "__main__":
    main()
