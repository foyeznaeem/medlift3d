#!/usr/bin/env python
"""Hallucination audit: does the prior invent or erase nodules?

    python scripts/hallucination.py --tracks A16 B15 --methods sirt_tv diffusion --prior runs/prior/best.pt

Under sparse or limited-angle acquisition the reconstruction is
under-determined, so a learned prior *must* invent structure -- and nodules are
exactly the small high-frequency features a prior is most likely to invent or
smooth away. This is the safety experiment.

Three controlled conditions:
  present   nodule in the ground truth  -> should be detected (sensitivity)
  erased    nodule removed from the GT   -> must NOT appear (false positives)
  inserted  synthetic nodule added       -> should survive the pipeline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from tqdm.auto import tqdm

from medlift3d.baselines import cgls, sirt_tv
from medlift3d.datasets import (default_dirs, geometry_from_dict, load_case,
                                load_split, simulate, verify_split)
from medlift3d.diffusion import GaussianDiffusion
from medlift3d.metrics import hallucinated_at
from medlift3d.phantom import erase_nodule, grow_nodule, lung_mask
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
    ap.add_argument("--out", type=Path, default=d["out"] / "hallucination")
    ap.add_argument("--tracks", nargs="*", default=["A16", "B15"])
    ap.add_argument("--methods", nargs="*", default=["sirt_tv"])
    ap.add_argument("--prior", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=5)
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
          f"methods={args.methods}")

    rows = []
    for cid in tqdm(case_ids, desc="cases"):
        case = load_case(args.data / f"{cid}.npz")
        grid, mu0 = case["grid"], case["mu"]
        region = lung_mask(mu0, grid)
        nodules = case["meta"].get("nodules", [])
        if not nodules:
            continue

        for nod in nodules:
            # "inserted" enlarges a real nodule substantially, which tests
            # whether a *new* structure of that size survives the pipeline.
            variants = {
                "present": mu0,
                "erased": erase_nodule(grid, mu0, nod),
                "inserted": grow_nodule(grid, mu0, nod, 1.0)[0],
            }
            for track in args.tracks:
                if track not in case["projections"]:
                    continue
                geom = geometry_from_dict(case["projections"][track]["geometry"])
                proj = Projector(grid, geom, device=device)
                for method in args.methods:
                    for cond, vol in variants.items():
                        p = torch.from_numpy(
                            simulate(vol, grid, geom, i0=(args.i0 or None),
                                     seed=args.seed, device=device)).to(device)
                        r = reconstruct(method, p, proj, diffusion, args).cpu().numpy()
                        res = hallucinated_at(r, nod["centre_xyz"], grid, region=region)
                        rows.append({
                            "case_id": cid, "label": nod["label"], "track": track,
                            "method": method, "condition": cond,
                            "nodule_diameter_mm": 2 * nod["radius_mm"],
                            "found": res["hallucinated"],
                            "blob_volume_mm3": res["blob_volume_mm3"],
                        })

    if not rows:
        raise SystemExit("no results; check --tracks against the dataset")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "hallucination_raw.csv", index=False)

    piv = (df.pivot_table(index=["method", "track"], columns="condition",
                          values="found", aggfunc="mean") * 100).round(1)
    piv = piv.rename(columns={"present": "sensitivity_%",
                              "erased": "false_positive_%",
                              "inserted": "insertion_recall_%"})
    piv.to_csv(args.out / "hallucination_summary.csv")
    print("\n=== hallucination audit (%) ===")
    print(piv.to_string())
    print("\n  sensitivity_%      : real nodules recovered            (higher better)")
    print("  false_positive_%   : nodules INVENTED where none exist  (lower better)")
    print("  insertion_recall_% : new structure that survived        (higher better)")

    band = df[df["condition"] == "present"].copy()
    band["size_band"] = pd.cut(band["nodule_diameter_mm"], [0, 6, 10, 20, 1e9],
                               labels=["<6mm", "6-10mm", "10-20mm", ">20mm"])
    print("\n=== detection sensitivity by nodule size ===")
    print((band.groupby(["method", "track", "size_band"], observed=True)["found"]
           .mean() * 100).round(1).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
