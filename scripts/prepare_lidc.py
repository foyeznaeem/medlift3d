#!/usr/bin/env python
"""Ingest real chest CT onto the canonical grid.

Two input routes, because Kaggle notebooks are offline and the full LIDC-IDRI
download (~124 GB) is not feasible there:

    # A) LIDC-IDRI via pylidc (needs a local DICOM store + ~/.pylidcrc)
    python scripts/prepare_lidc.py --source pylidc --limit 400 --out data/lidc

    # B) a directory of volumes already on disk (Kaggle-hosted LUNA/LIDC subsets)
    python scripts/prepare_lidc.py --source dir --in /kaggle/input/luna16 --out data/lidc

LIDC-IDRI is the only public source with **voxel-level nodule contours** (four
independent readers per nodule). Challenge sets in the LUNA family ship
centroids and malignancy labels, from which no segmentation ground truth can be
built, so Dice and volume error are not computable. Route B therefore imports
volumes without masks: usable for training the prior, not for clinical metrics.

Ground-truth masks use >= 50% reader consensus.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from tqdm.auto import tqdm

from medlift3d.datasets import (default_dirs, geometry_to_dict, make_split,
                                save_case, save_split, simulate, verify_split)
from medlift3d.geometry import CHEST, Grid
from medlift3d.projector import DTSGeometry, ParallelGeometry
from medlift3d.units import HU_CLIP, hu_to_mu
from medlift3d.utils import write_json


def resample_to_canonical(hu: np.ndarray, src: Grid, dst: Grid) -> np.ndarray:
    """Trilinear resample HU onto the canonical grid, preserving physical size."""
    import torch.nn.functional as F
    pts = dst.voxel_centres_xyz(dtype=torch.float32)
    g = src.world_to_norm(pts.reshape(-1, 3)).view(1, *dst.shape, 3)
    t = torch.from_numpy(np.ascontiguousarray(hu, dtype=np.float32))[None, None]
    out = F.grid_sample(t, g, mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out.view(dst.shape).numpy()


def grid_from_sitk(image) -> tuple[Grid, np.ndarray]:
    """SimpleITK image -> (Grid, HU array in (z, y, x)), converted LPS -> RAS.

    Going from LPS to RAS negates the x and y world axes. Negating the origin
    alone is not the conversion: with spacing kept positive the array itself has
    to be reversed along those two axes, and the origin becomes the *far* corner
    negated. Doing only half of it leaves a volume that is mirrored left-right
    and anterior-posterior with respect to the frame it claims, which is
    self-consistent inside this pipeline and wrong the moment laterality
    matters -- an exported NIfTI, or a nodule reported in the wrong lung.
    """
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(image).astype(np.float32)      # (z, y, x) LPS
    sp = image.GetSpacing()                                     # (x, y, z)
    org = image.GetOrigin()                                     # (x, y, z) LPS
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    if not np.allclose(np.abs(direction), np.eye(3), atol=1e-3):
        raise ValueError(
            f"oblique acquisition (direction={direction.tolist()}); this "
            f"importer assumes axis-aligned volumes")

    nz, ny, nx = arr.shape
    arr = np.ascontiguousarray(arr[:, ::-1, ::-1])              # -> RAS
    origin = (float(org[2]),
              -(float(org[1]) + (ny - 1) * float(sp[1])),
              -(float(org[0]) + (nx - 1) * float(sp[0])))
    grid = Grid((nz, ny, nx),
                (float(sp[2]), float(sp[1]), float(sp[0])), origin)
    return grid, arr


def _patch_pylidc_deps():
    """Restore the APIs pylidc needs but Python and NumPy have since removed.

    pylidc was last released in 2020 and calls several names that have since
    gone away. Found by grepping the installed package, so this is the complete
    set rather than whichever one happened to raise first:

      configparser.SafeConfigParser   Scan.py            (removed in Python 3.12)
      np.int                          Contour.to_matrix, Annotation.bbox
      np.float                        utils, Annotation diameter/volume
      np.bool                         Annotation.boolean_mask   (all NumPy 1.24)

    All of them were deprecated aliases of builtins, so aliasing them back is
    exactly what the code expects and changes no behaviour.
    """
    import configparser
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

    # Exactly the three the grep found. Checking for np.object or np.str would
    # itself raise a FutureWarning, and pylidc does not use them.
    import numpy as _np
    for name, builtin in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(_np, name):
            setattr(_np, name, builtin)


def iter_pylidc(limit, max_slice_thickness):
    _patch_pylidc_deps()

    try:
        import pylidc as pl
    except ModuleNotFoundError as e:
        raise SystemExit(
            "pylidc is not installed, and --source pylidc needs it: it ships "
            "the nodule annotations (1,018 scans, 6,859 readings, 41,406 "
            "contours) that become the ground-truth masks.\n"
            "    pip install pylidc"
        ) from e
    scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= max_slice_thickness)
    for scan in scans.limit(limit) if limit else scans:
        try:
            vol = scan.to_volume().astype(np.float32)           # (y, x, z) HU
        except Exception as e:                                  # noqa: BLE001
            print(f"  skip {scan.patient_id}: {e}")
            continue
        hu = np.transpose(vol, (2, 0, 1))                       # -> (z, y, x)
        grid = Grid(hu.shape,
                    (float(scan.slice_thickness), float(scan.pixel_spacing),
                     float(scan.pixel_spacing)))
        # >= 50% reader consensus over clustered annotations.
        mask = np.zeros(hu.shape, dtype=np.uint8)
        nodules = []
        for i, anns in enumerate(scan.cluster_annotations(), start=1):
            if not anns:
                continue
            acc = np.zeros(hu.shape, dtype=np.uint8)
            for a in anns:
                m, bbox = a.boolean_mask(), a.bbox()
                sub = (slice(bbox[2].start, bbox[2].stop),
                       slice(bbox[0].start, bbox[0].stop),
                       slice(bbox[1].start, bbox[1].stop))
                acc[sub] += np.transpose(m, (2, 0, 1)).astype(np.uint8)
            consensus = acc >= max(1, int(np.ceil(len(anns) / 2)))
            if not consensus.any():
                continue
            mask[consensus] = i
            idx = np.argwhere(consensus).mean(0)
            nodules.append({
                "label": i, "n_readers": len(anns),
                "centre_xyz": [float(idx[2] * grid.spacing[2] + grid.origin[2]),
                               float(idx[1] * grid.spacing[1] + grid.origin[1]),
                               float(idx[0] * grid.spacing[0] + grid.origin[0])],
                "mask_volume_mm3": float(consensus.sum() * grid.voxel_volume_mm3),
            })
        yield scan.patient_id, hu, grid, mask, {"nodules": nodules,
                                                "source": "LIDC-IDRI"}


def iter_dir(in_dir, limit, max_slice_thickness):
    import SimpleITK as sitk
    pats = ("*.mhd", "*.mha", "*.nii", "*.nii.gz")
    files = sorted(f for p in pats for f in Path(in_dir).rglob(p))
    if not files:
        raise SystemExit(f"no volumes ({', '.join(pats)}) under {in_dir}")
    print(f"found {len(files)} volumes under {in_dir}")
    for f in files[:limit] if limit else files:
        try:
            grid, hu = grid_from_sitk(sitk.ReadImage(str(f)))
        except Exception as e:                                  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        if grid.spacing[0] > max_slice_thickness:
            continue
        # No contours available on this route -> no nodule mask.
        yield f.name.split(".")[0], hu, grid, None, {"source": str(f.parent)}


def main():
    d = default_dirs()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["pylidc", "dir"], default="pylidc")
    ap.add_argument("--in", dest="in_dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=d["data"] / "lidc")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-slice-thickness", type=float, default=1.5,
                    help="mm; LIDC ranges 0.6-3.0 and thick slices make "
                         "volumetry ground truth unreliable")
    ap.add_argument("--shape", type=int, nargs=3, default=list(CHEST.shape))
    ap.add_argument("--spacing", type=float, nargs=3, default=list(CHEST.spacing))
    ap.add_argument("--views-a", type=int, nargs="*", default=[16, 32])
    ap.add_argument("--views-b", type=int, nargs="*", default=[15])
    ap.add_argument("--arc-a", type=float, default=180.0)
    ap.add_argument("--arc-b", type=float, default=30.0)
    ap.add_argument("--det-spacing", type=float, default=None)
    ap.add_argument("--i0", type=float, default=1e5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.source == "dir" and args.in_dir is None:
        ap.error("--source dir requires --in")

    dst = Grid.centred(tuple(args.shape), tuple(args.spacing))
    geoms = {}
    for n in args.views_a:
        geoms[f"A{n}"] = ParallelGeometry.covering(dst, n, args.arc_a, args.det_spacing)
    for n in args.views_b:
        geoms[f"B{n}"] = DTSGeometry.covering(dst, n, args.arc_b, args.det_spacing)

    print(f"canonical grid {dst.shape} @ {dst.spacing} mm")
    for k, g in geoms.items():
        print(f"  {k:5s} {g.kind:8s} views={g.n_views:3d} arc={g.arc_deg:5.1f} "
              f"det={g.det_shape}")

    args.out.mkdir(parents=True, exist_ok=True)
    it = (iter_pylidc(args.limit, args.max_slice_thickness) if args.source == "pylidc"
          else iter_dir(args.in_dir, args.limit, args.max_slice_thickness))

    ids, n_with_masks = [], 0
    for cid, hu, src_grid, mask, meta in tqdm(it, desc="cases"):
        hu = np.clip(hu, *HU_CLIP)
        hu_r = resample_to_canonical(hu, src_grid, dst)
        mu = hu_to_mu(hu_r).astype(np.float32)

        mask_r = None
        if mask is not None:
            # Nearest-neighbour equivalent: resample each label and threshold,
            # so labels are never blended into each other.
            mask_r = np.zeros(dst.shape, dtype=np.uint8)
            for lab in np.unique(mask):
                if lab == 0:
                    continue
                m = resample_to_canonical((mask == lab).astype(np.float32),
                                          src_grid, dst)
                mask_r[m >= 0.5] = lab
            n_with_masks += 1

        projections = {}
        for name, geom in geoms.items():
            p = simulate(mu, dst, geom, i0=(args.i0 or None), seed=args.seed,
                         device=args.device)
            projections[name] = {"projections": p, "geometry": geometry_to_dict(geom)}

        meta.update(tracks=list(geoms), original_grid=src_grid.as_dict())
        save_case(args.out / f"{cid}.npz", mu, dst, mask_r, projections, meta)
        ids.append(cid)

    if not ids:
        raise SystemExit("no cases ingested")
    split = make_split(ids, seed=args.seed)
    verify_split(split)
    save_split(args.out / "splits.csv", split)
    write_json(args.out / "dataset.json", {
        "grid": dst.as_dict(), "source": args.source, "n_cases": len(ids),
        "n_with_nodule_masks": n_with_masks,
        "geometries": {k: geometry_to_dict(v) for k, v in geoms.items()},
        "max_slice_thickness_mm": args.max_slice_thickness, "i0": args.i0,
    })
    print(f"\ningested {len(ids)} cases ({n_with_masks} with nodule masks) "
          f"-> {args.out}")
    print(f"split: train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])}")
    if n_with_masks == 0:
        print("\nNo nodule masks: this data can train the prior but cannot support\n"
              "nodule Dice or volume error. Use --source pylidc on LIDC-IDRI for those.")


if __name__ == "__main__":
    main()
