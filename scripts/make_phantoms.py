#!/usr/bin/env python
"""Generate a synthetic phantom dataset with simulated projections.

This is the route that makes the whole pipeline runnable with no dataset
download, which matters on an offline Kaggle image. Real data goes through
`prepare_lidc.py`; everything downstream is identical.

    python scripts/make_phantoms.py --n-cases 40 --out data/phantom
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm

from medlift3d.datasets import (default_dirs, geometry_to_dict, make_split,
                                save_case, save_split, simulate, verify_split)
from medlift3d.geometry import Grid
from medlift3d.phantom import make_chest_phantom
from medlift3d.projector import DTSGeometry, ParallelGeometry
from medlift3d.utils import write_json


def build_geometries(grid, views_a, arc_a, views_b, arc_b, det_spacing):
    """Track A: sparse-view CT over a wide arc. Track B: limited-angle DTS.

    Track A first, always: if the pipeline cannot do 32 views over 180 degrees
    it certainly cannot do 15 over 30, and diagnosing that on the easy case is
    far cheaper.
    """
    g = {}
    for n in views_a:
        g[f"A{n}"] = ParallelGeometry.covering(grid, n, arc_a, det_spacing)
    for n in views_b:
        g[f"B{n}"] = DTSGeometry.covering(grid, n, arc_b, det_spacing)
    return g


def main():
    d = default_dirs()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--out", type=Path, default=d["data"] / "phantom")
    ap.add_argument("--shape", type=int, nargs=3, default=[128, 128, 128],
                    help="(nz ny nx). 256 256 256 is the paper target; 128 fits smaller GPUs.")
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    ap.add_argument("--views-a", type=int, nargs="*", default=[16, 32],
                    help="Track A view counts (sparse-view CT, 180 deg)")
    ap.add_argument("--views-b", type=int, nargs="*", default=[15],
                    help="Track B view counts (limited-angle DTS)")
    ap.add_argument("--arc-a", type=float, default=180.0)
    ap.add_argument("--arc-b", type=float, default=30.0)
    ap.add_argument("--det-spacing", type=float, default=None)
    ap.add_argument("--nodules", type=int, default=3)
    ap.add_argument("--i0", type=float, default=1e5, help="photon count; 0 disables noise")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    grid = Grid.centred(tuple(args.shape), tuple(args.spacing))
    geoms = build_geometries(grid, args.views_a, args.arc_a, args.views_b,
                             args.arc_b, args.det_spacing)
    print(f"grid {grid.shape} @ {grid.spacing} mm  extent {grid.extent_mm} mm")
    for k, g in geoms.items():
        extra = "" if g.kind == "parallel" else f"  missing wedge {g.missing_wedge_deg:.0f} deg"
        print(f"  {k:5s} {g.kind:8s} views={g.n_views:3d} arc={g.arc_deg:5.1f} "
              f"det={g.det_shape}{extra}")

    args.out.mkdir(parents=True, exist_ok=True)
    ids = []
    for i in tqdm(range(args.n_cases), desc="cases"):
        cid = f"phantom{i:04d}"
        mu, mask, meta = make_chest_phantom(grid, seed=args.seed + i,
                                           n_nodules=args.nodules)
        projections = {}
        for name, geom in geoms.items():
            p = simulate(mu, grid, geom, i0=(args.i0 or None),
                         seed=args.seed + i, device=args.device)
            projections[name] = {"projections": p, "geometry": geometry_to_dict(geom)}
        meta["tracks"] = list(geoms)
        save_case(args.out / f"{cid}.npz", mu, grid, mask, projections, meta)
        ids.append(cid)

    split = make_split(ids, seed=args.seed)
    verify_split(split)
    save_split(args.out / "splits.csv", split)
    write_json(args.out / "dataset.json", {
        "grid": grid.as_dict(),
        "geometries": {k: geometry_to_dict(v) for k, v in geoms.items()},
        "n_cases": len(ids), "i0": args.i0, "seed": args.seed,
        "nodules_per_case": args.nodules,
    })
    print(f"\nwrote {len(ids)} cases to {args.out}")
    print(f"split: train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}  ->  {args.out / 'splits.csv'}")


if __name__ == "__main__":
    main()
