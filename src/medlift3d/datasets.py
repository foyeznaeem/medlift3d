"""Data layer: cases on disk, patient-level splits, and simulation.

Every case is one `.npz` carrying its own physical frame. A loader that returns
a bare array has already lost the information needed to evaluate it, which is
how the FYDP-1 pipeline ended up comparing a 324x65x94 reconstruction against a
512x512x184 ground truth.

Splits are **patient-level and committed to disk**. If the prior has seen a test
patient then hallucination is indistinguishable from reconstruction and the
entire safety argument collapses, so `verify_split` is a hard check, not advice.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import Grid
from .projector import DTSGeometry, ParallelGeometry, Projector
from .units import apply_poisson

REQUIRED_KEYS = ("mu", "shape", "spacing", "origin")


# ----------------------------------------------------------------------------
# environment
# ----------------------------------------------------------------------------

def default_dirs() -> dict:
    """Auto-detect Kaggle vs local paths.

    Kaggle mounts datasets read-only at /kaggle/input and gives ~20 GB of
    writable space at /kaggle/working, so nothing may be written next to the
    inputs.
    """
    if Path("/kaggle/working").exists():
        return {"data": Path("/kaggle/input"), "out": Path("/kaggle/working"),
                "env": "kaggle"}
    root = Path(os.environ.get("MEDLIFT3D_ROOT", "."))
    return {"data": root / "data", "out": root / "runs", "env": "local"}


# ----------------------------------------------------------------------------
# case I/O
# ----------------------------------------------------------------------------

def save_case(path, mu: np.ndarray, grid: Grid, nodule_mask: np.ndarray | None = None,
              projections: dict | None = None, meta: dict | None = None) -> Path:
    """Write one case. `projections` maps a track name to its array + geometry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mu": mu.astype(np.float32),
        "shape": np.asarray(grid.shape, dtype=np.int64),
        "spacing": np.asarray(grid.spacing, dtype=np.float64),
        "origin": np.asarray(grid.origin, dtype=np.float64),
        "meta": np.array(json.dumps(meta or {})),
    }
    if nodule_mask is not None:
        payload["nodule_mask"] = nodule_mask.astype(np.uint8)
    for name, rec in (projections or {}).items():
        payload[f"proj_{name}"] = np.asarray(rec["projections"], dtype=np.float32)
        payload[f"geom_{name}"] = np.array(json.dumps(rec["geometry"]))
    np.savez_compressed(path, **payload)
    return path


def load_case(path, with_projections: bool = True) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z]
        if missing:
            raise KeyError(f"{path.name} is missing {missing}; regenerate it")
        grid = Grid(tuple(int(v) for v in z["shape"]),
                    tuple(float(v) for v in z["spacing"]),
                    tuple(float(v) for v in z["origin"]))
        out = {"case_id": path.stem, "grid": grid,
               "mu": z["mu"].astype(np.float32),
               "meta": json.loads(str(z["meta"]))}
        if out["mu"].shape != grid.shape:
            raise ValueError(f"{path.name}: mu shape {out['mu'].shape} != grid {grid.shape}")
        if "nodule_mask" in z:
            out["nodule_mask"] = z["nodule_mask"]
        if with_projections:
            out["projections"] = {}
            for k in z.files:
                if k.startswith("proj_"):
                    name = k[5:]
                    out["projections"][name] = {
                        "projections": z[k].astype(np.float32),
                        "geometry": json.loads(str(z[f"geom_{name}"])),
                    }
    return out


# ----------------------------------------------------------------------------
# geometry (de)serialisation -- one source of truth
# ----------------------------------------------------------------------------

def geometry_to_dict(geom) -> dict:
    d = {"kind": geom.kind, "n_views": geom.n_views, "arc_deg": geom.arc_deg,
         "det_shape": list(geom.det_shape), "det_spacing": list(geom.det_spacing),
         "centre": list(geom.centre)}
    if geom.kind == "parallel":
        d["start_deg"] = geom.start_deg
    else:
        d["src_dist"] = geom.src_dist
        d["det_dist"] = geom.det_dist
    return d


def geometry_from_dict(d: dict):
    d = dict(d)
    kind = d.pop("kind")
    d["det_shape"] = tuple(d["det_shape"])
    d["det_spacing"] = tuple(d["det_spacing"])
    d["centre"] = tuple(d["centre"])
    return ParallelGeometry(**d) if kind == "parallel" else DTSGeometry(**d)


# ----------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------

def simulate(mu: np.ndarray, grid: Grid, geom, i0: float | None = 1e5,
             seed: int = 0, device="cpu", chunk_rays: int = 8192) -> np.ndarray:
    """Forward project with the SAME operator the solver optimises through.

    This is the invariant the FYDP-1 pipeline broke: it simulated polychromatic,
    flood-corrected, log-transformed projections of a density field and then
    compared them against monochromatic line integrals of a [0,1] field. Two
    incompatible physics models either side of one MSE. Simulate with the
    operator you optimise through, or accept a domain gap and measure it
    deliberately (see `scripts/simulate.py --polychromatic`).
    """
    proj = Projector(grid, geom, device=device, chunk_rays=chunk_rays)
    with torch.no_grad():
        p = proj.fp(torch.from_numpy(np.ascontiguousarray(mu)).to(device)).cpu().numpy()
    if i0:
        p = apply_poisson(p, i0=i0, rng=np.random.default_rng(seed))
    return p.astype(np.float32)


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------

def make_split(case_ids, fractions=(0.75, 0.125, 0.125), seed: int = 42) -> dict:
    """Deterministic patient-level split. Write it once and never regenerate."""
    ids = sorted(set(case_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    return {"train": sorted(ids[:n_tr]),
            "val": sorted(ids[n_tr:n_tr + n_va]),
            "test": sorted(ids[n_tr + n_va:]),
            "seed": seed}


def save_split(path, split: dict) -> Path:
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"case_id": c, "split": s} for s in ("train", "val", "test") for c in split[s]]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_split(path) -> dict:
    import pandas as pd
    df = pd.read_csv(path)
    return {s: sorted(df.loc[df["split"] == s, "case_id"].astype(str)) for s in
            ("train", "val", "test")}


def verify_split(split: dict) -> None:
    """Assert the three sets are disjoint.

    Data leakage here does not produce a visibly wrong number -- it produces a
    flatteringly right one. So it is checked, not trusted.
    """
    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    for a, b, na, nb in ((tr, va, "train", "val"), (tr, te, "train", "test"),
                         (va, te, "val", "test")):
        overlap = a & b
        if overlap:
            raise AssertionError(
                f"LEAKAGE: {len(overlap)} case(s) in both {na} and {nb}: "
                f"{sorted(overlap)[:5]}")
    if not te:
        raise AssertionError("empty test set")


# ----------------------------------------------------------------------------
# torch datasets
# ----------------------------------------------------------------------------

class CaseDataset(Dataset):
    """Whole cases, for reconstruction and evaluation."""

    def __init__(self, data_dir, case_ids=None, track: str | None = None):
        self.dir = Path(data_dir)
        files = sorted(self.dir.glob("*.npz"))
        if case_ids is not None:
            keep = set(str(c) for c in case_ids)
            files = [f for f in files if f.stem in keep]
        self.files = files
        self.track = track
        if not files:
            raise FileNotFoundError(f"no .npz cases in {self.dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return load_case(self.files[i])


class SliceDataset(Dataset):
    """Axial slices of `mu`, for training the 2-D prior.

    Slices are indexed lazily through an (case, z) table so memory stays flat
    regardless of dataset size, and `mmap` keeps startup fast.
    """

    def __init__(self, data_dir, case_ids=None, min_content: float = 1e-3,
                 normalise=True, cond_key: str | None = None):
        from .units import mu_to_net
        self.dir = Path(data_dir)
        self.normalise = normalise
        self._to_net = mu_to_net
        files = sorted(self.dir.glob("*.npz"))
        if case_ids is not None:
            keep = set(str(c) for c in case_ids)
            files = [f for f in files if f.stem in keep]
        if not files:
            raise FileNotFoundError(f"no .npz cases in {self.dir}")
        self.files = files
        self.index: list[tuple[int, int]] = []
        for fi, f in enumerate(files):
            with np.load(f, allow_pickle=False) as z:
                mu = z["mu"]
                # Skip near-empty slices: they teach the prior nothing and
                # inflate the epoch.
                keep_z = np.where(mu.reshape(mu.shape[0], -1).mean(1) > min_content)[0]
            self.index.extend((fi, int(zi)) for zi in keep_z)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, zi = self.index[i]
        with np.load(self.files[fi], allow_pickle=False, mmap_mode="r") as z:
            sl = np.asarray(z["mu"][zi], dtype=np.float32)
        x = self._to_net(sl) if self.normalise else sl
        return {"x": torch.from_numpy(np.ascontiguousarray(x))[None]}
