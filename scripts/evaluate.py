#!/usr/bin/env python
"""Score reconstructions against ground truth and write a comparison table.

    python scripts/evaluate.py --recon runs/recon --data data/phantom

Reports the metric set from `docs/PLAN.md` section 7.1: global fidelity inside
the lung mask, and per-nodule Dice / absolute percent volume error. Not FID or
MMD -- those measure distributional similarity for unconditional generation,
whereas this is a paired per-patient reconstruction task with ground truth for
every case.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from medlift3d.datasets import default_dirs, load_case
from medlift3d.geometry import Grid
from medlift3d.metrics import global_metrics, nodule_metrics
from medlift3d.phantom import lung_mask


def load_recon(path):
    with np.load(path, allow_pickle=False) as z:
        grid = Grid(tuple(int(v) for v in z["shape"]),
                    tuple(float(v) for v in z["spacing"]),
                    tuple(float(v) for v in z["origin"]))
        return z["mu"].astype(np.float32), (z["std"].astype(np.float32)
                                            if "std" in z else None), grid


def main():
    d = default_dirs()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", type=Path, default=d["out"] / "recon",
                    help="directory of <method>_<track>/ subdirectories")
    ap.add_argument("--data", type=Path, default=d["data"] / "phantom")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()

    out = args.out or args.recon / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p for p in args.recon.iterdir()
                  if p.is_dir() and p.name != "metrics"
                  and (args.methods is None or p.name in args.methods))
    if not dirs:
        raise SystemExit(f"no method directories in {args.recon}")

    rows, nod_rows = [], []
    for mdir in dirs:
        for f in tqdm(sorted(mdir.glob("*.npz")), desc=mdir.name):
            cid = f.stem
            case_path = args.data / f"{cid}.npz"
            if not case_path.exists():
                continue
            case = load_case(case_path, with_projections=False)
            gt, grid = case["mu"], case["grid"]
            mu, std, rgrid = load_recon(f)
            # Same grid, same frame -- otherwise the comparison is meaningless.
            rgrid.assert_matches(grid, what=f"{mdir.name}/{cid}")

            lung = lung_mask(gt, grid)
            row = {"method": mdir.name, "case_id": cid,
                   **global_metrics(mu, gt, lung)}
            if std is not None:
                row["mean_uncertainty"] = float(std[lung].mean()) if lung.any() else float(std.mean())
            rows.append(row)

            mask = case.get("nodule_mask")
            if mask is not None:
                for lab in np.unique(mask):
                    if lab == 0:
                        continue
                    # `lung` is the hole-filled lung region from GT anatomy: an
                    # ROI, not the answer. It stops a pleural nodule's
                    # segmentation escaping into the same-density chest wall.
                    nod_rows.append({"method": mdir.name, "case_id": cid,
                                     **nodule_metrics(mu, gt, mask, grid, int(lab),
                                                      region=lung)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "global.csv", index=False)
    print("\n=== global fidelity (mean over cases) ===")
    gcols = [c for c in ("psnr", "psnr_lung", "ssim", "ssim_lung", "mae_hu_lung",
                         "mean_uncertainty") if c in df.columns]
    print(df.groupby("method")[gcols].mean().round(4).to_string())

    if nod_rows:
        nd = pd.DataFrame(nod_rows)
        nd.to_csv(out / "nodules.csv", index=False)
        # Stratify by size: small nodules are the hard, clinically decisive case
        # and averaging over all sizes hides exactly where a method fails.
        nd["size_band"] = pd.cut(nd["gt_diameter_mm"], [0, 6, 10, 20, 1e9],
                                 labels=["<6mm", "6-10mm", "10-20mm", ">20mm"])
        print("\n=== nodule metrics (mean) ===")
        print(nd.groupby("method")[["nodule_dice", "volume_ape_pct",
                                    "dice_vs_consensus", "ape_vs_consensus_pct",
                                    "detected"]].mean().round(4).to_string())
        print("\n=== by nodule size ===")
        print(nd.groupby(["method", "size_band"], observed=True)[
            ["nodule_dice", "volume_ape_pct", "detected"]].mean().round(4).to_string())

    print(f"\nwrote {out/'global.csv'}" + (f" and {out/'nodules.csv'}" if nod_rows else ""))


if __name__ == "__main__":
    main()
