#!/usr/bin/env python
"""Objective O5: does the Gaussian parameterisation earn its place?

    python scripts/ablate_roi.py --track A16 --iters 600

Runs the identical two-scale ROI refinement twice -- once with an adaptive
Gaussian field, once on a plain voxel grid -- at matched iteration count, and
reports nodule Dice, volume error and wall-clock for each.

This ablation is deliberately falsifiable. The FYDP-1 report justified Gaussian
splatting on efficiency grounds that its own implementation did not deliver (it
materialised a dense volume every iteration). The defensible claim is narrower:
Gaussians are a compact *adaptive* basis. Whether that helps is an empirical
question, and a negative answer here is a legitimate result to publish -- it
just means the ROI stage should be plain voxels and 3DGS belongs in the viewer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from medlift3d.baselines import sirt_tv
from medlift3d.datasets import (default_dirs, geometry_from_dict, load_case,
                                load_split, verify_split)
from medlift3d.gaussians import GaussianConfig
from medlift3d.metrics import measure_nodule_volume
from medlift3d.phantom import lung_mask
from medlift3d.projector import Projector
from medlift3d.roi import refine_roi, resample
from medlift3d.utils import pick_device, seed_everything


def main():
    d = default_dirs()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=d["data"] / "phantom")
    ap.add_argument("--out", type=Path, default=d["out"] / "ablate_roi")
    ap.add_argument("--track", default="A16")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--roi-shape", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--roi-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--n-gaussians", type=int, default=30000)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--coarse-iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    split = load_split(args.data / "splits.csv")
    verify_split(split)
    case_ids = split["test"][:args.limit]
    print(f"device={device}  track={args.track}  cases={len(case_ids)}  "
          f"iters={args.iters}  roi={tuple(args.roi_shape)}@{tuple(args.roi_spacing)}mm")

    rows = []
    for cid in tqdm(case_ids, desc="cases"):
        case = load_case(args.data / f"{cid}.npz")
        grid, mu_gt = case["grid"], case["mu"]
        nodules = case["meta"].get("nodules", [])
        if not nodules or args.track not in case["projections"]:
            continue
        nod = max(nodules, key=lambda n: n["radius_mm"])

        geom = geometry_from_dict(case["projections"][args.track]["geometry"])
        projs = torch.from_numpy(case["projections"][args.track]["projections"]).to(device)
        proj = Projector(grid, geom, device=device)

        # Shared coarse reconstruction: both arms start from the same place.
        coarse = sirt_tv(projs, proj, args.coarse_iters)

        for use_g in (True, False):
            t0 = time.perf_counter()
            res = refine_roi(
                projs, coarse, proj, nod["centre_xyz"],
                roi_shape=tuple(args.roi_shape), roi_spacing=tuple(args.roi_spacing),
                n_iter=args.iters, use_gaussians=use_g, progress=False,
                gcfg=GaussianConfig(n_init=args.n_gaussians, window=args.window),
                seed=args.seed)
            dt = time.perf_counter() - t0

            roi_grid = res["roi_grid"]
            gt_roi = resample(torch.from_numpy(mu_gt).to(device), grid,
                              roi_grid).cpu().numpy()
            region = lung_mask(gt_roi, roi_grid)
            mu_roi = res["mu_roi"].cpu().numpy()

            v_pred = measure_nodule_volume(mu_roi, nod["centre_xyz"], roi_grid,
                                           region=region)
            v_gt = measure_nodule_volume(gt_roi, nod["centre_xyz"], roi_grid,
                                         region=region)
            rows.append({
                "case_id": cid, "parameterisation": res["parameterisation"],
                "seconds": round(dt, 2), "n_params": res["n_primitives"],
                "residual_rmse": res["residual_rmse"],
                "pred_vol_mm3": v_pred, "gt_vol_mm3": v_gt,
                "volume_ape_pct": (abs(v_pred - v_gt) / v_gt * 100
                                   if v_gt > 0 else np.nan),
                "rmse_vs_gt": float(np.sqrt(((mu_roi - gt_roi) ** 2).mean())),
            })

    if not rows:
        raise SystemExit("no results; check --track against the dataset")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "ablate_roi_raw.csv", index=False)
    summary = df.groupby("parameterisation")[
        ["residual_rmse", "rmse_vs_gt", "volume_ape_pct", "seconds", "n_params"]
    ].mean().round(4)
    summary.to_csv(args.out / "ablate_roi_summary.csv")

    print("\n=== O5: Gaussian vs voxel ROI refinement (matched iterations) ===")
    print(summary.to_string())

    if {"gaussian", "voxel"} <= set(summary.index):
        g, v = summary.loc["gaussian"], summary.loc["voxel"]
        better = g["rmse_vs_gt"] < v["rmse_vs_gt"]
        print(f"\nVerdict: Gaussian parameterisation is "
              f"{'BETTER' if better else 'NOT better'} than voxels on RMSE vs GT "
              f"({g['rmse_vs_gt']:.5g} vs {v['rmse_vs_gt']:.5g}), at "
              f"{g['seconds']:.1f}s vs {v['seconds']:.1f}s.")
        if not better:
            print("A negative result here is a legitimate finding: report it, drop the\n"
                  "Gaussian stage from the reconstruction path, and keep 3DGS (if at\n"
                  "all) for interactive visualisation only.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
