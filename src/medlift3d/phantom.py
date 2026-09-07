"""Synthetic chest phantoms.

Real work uses LIDC-IDRI (`scripts/prepare_lidc.py`). These phantoms exist so
that every gate test, the solver, and a smoke training run are executable with
no dataset download -- which is what makes the pipeline verifiable before any
data agreement is signed.

Structures are built in HU and converted once to `mu`. Nodules get a proper
partial-volume edge: a hard binary sphere would make volumetry artificially easy
and hide exactly the error the project is trying to measure.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import Grid
from .units import hu_to_mu

HU_AIR = -1000.0
HU_LUNG = -820.0
HU_SOFT = 40.0
HU_MUSCLE = 55.0
HU_BONE = 700.0
HU_SPINE = 320.0
HU_NODULE = 30.0
HU_VESSEL = 45.0


def _coords(grid: Grid):
    nz, ny, nx = grid.shape
    sz, sy, sx = grid.spacing
    oz, oy, ox = grid.origin
    z = np.arange(nz) * sz + oz
    y = np.arange(ny) * sy + oy
    x = np.arange(nx) * sx + ox
    return np.meshgrid(z, y, x, indexing="ij")


def _soft_ellipsoid(zz, yy, xx, centre, radii, voxel):
    """Partial-volume-aware ellipsoid occupancy in [0, 1]."""
    cz, cy, cx = centre
    rz, ry, rx = radii
    # Normalised radius, then rescale to an approximate signed distance in mm.
    q = np.sqrt(((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    scale = min(rz, ry, rx)
    sd = (q - 1.0) * scale
    return np.clip(0.5 - sd / voxel, 0.0, 1.0)


def make_water_cylinder(grid: Grid, radius_mm: float = 60.0, height_mm: float | None = None):
    """A uniform water cylinder along z. Used by the units gate test:
    the peak line integral through it must equal `mu_water * 2 * radius`."""
    zz, yy, xx = _coords(grid)
    cz, cy, cx = grid.centre_mm
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    inside = r <= radius_mm
    if height_mm is not None:
        inside &= np.abs(zz - cz) <= height_mm / 2.0
    mu = np.where(inside, hu_to_mu(0.0), hu_to_mu(HU_AIR))
    return mu.astype(np.float32)


def make_chest_phantom(grid: Grid, seed: int = 0, n_nodules: int = 3,
                       nodule_radius_mm: tuple[float, float] = (2.5, 11.0),
                       jitter: bool = True):
    """A randomised chest-like phantom.

    Returns `(mu, nodule_mask, meta)`:
      * `mu`          float32 [nz, ny, nx], mm^-1
      * `nodule_mask` uint8, labelled 1..n  (>= 50% partial-volume occupancy)
      * `meta`        dict with a `nodules` list of centre/radius/volume records
    """
    rng = np.random.default_rng(seed)
    zz, yy, xx = _coords(grid)
    cz, cy, cx = grid.centre_mm
    voxel = float(min(grid.spacing))
    ez, ey, ex = grid.extent_mm

    def jit(scale):
        return rng.uniform(-scale, scale) if jitter else 0.0

    hu = np.full(grid.shape, HU_AIR, dtype=np.float32)

    # --- torso: elliptical cylinder of soft tissue -------------------------
    body_ry = 0.30 * ey * (1 + 0.06 * jit(1))
    body_rx = 0.40 * ex * (1 + 0.06 * jit(1))
    body_cz, body_cy, body_cx = cz, cy + jit(4), cx + jit(4)
    body = _soft_ellipsoid(zz, yy, xx, (body_cz, body_cy, body_cx),
                           (ez * 10, body_ry, body_rx), voxel)   # ~infinite in z
    hu = hu * (1 - body) + HU_SOFT * body

    # --- chest wall musculature: a slightly denser shell -------------------
    inner = _soft_ellipsoid(zz, yy, xx, (body_cz, body_cy, body_cx),
                            (ez * 10, body_ry * 0.88, body_rx * 0.88), voxel)
    shell = np.clip(body - inner, 0, 1)
    hu = hu * (1 - shell) + HU_MUSCLE * shell

    # --- lungs -------------------------------------------------------------
    lung_rz = 0.34 * ez * (1 + 0.05 * jit(1))
    lung_ry = 0.19 * ey
    lung_rx = 0.15 * ex
    offx = 0.17 * ex
    lungs = np.zeros(grid.shape, dtype=np.float32)
    for sgn in (-1, +1):
        lungs = np.maximum(lungs, _soft_ellipsoid(
            zz, yy, xx,
            (body_cz + jit(6), body_cy - 0.04 * ey + jit(4), body_cx + sgn * offx + jit(4)),
            (lung_rz, lung_ry, lung_rx), voxel))
    hu = hu * (1 - lungs) + HU_LUNG * lungs
    lung_core = lungs > 0.5

    # --- spine -------------------------------------------------------------
    spine = _soft_ellipsoid(zz, yy, xx,
                            (body_cz, body_cy + 0.21 * ey, body_cx),
                            (ez * 10, 0.045 * ey, 0.05 * ex), voxel)
    hu = hu * (1 - spine) + HU_SPINE * spine

    # --- ribs: thin elliptical shells at regular z intervals ---------------
    rib_pitch = 22.0
    n_ribs = max(1, int(ez / rib_pitch))
    for i in range(n_ribs):
        z0 = grid.origin[0] + (i + 0.5) * ez / n_ribs + jit(2)
        band = np.exp(-0.5 * ((zz - z0) / 2.6) ** 2)
        outer = _soft_ellipsoid(zz, yy, xx, (body_cz, body_cy, body_cx),
                                (ez * 10, body_ry * 0.93, body_rx * 0.93), voxel)
        inner2 = _soft_ellipsoid(zz, yy, xx, (body_cz, body_cy, body_cx),
                                 (ez * 10, body_ry * 0.84, body_rx * 0.84), voxel)
        rib = np.clip(outer - inner2, 0, 1) * band
        hu = hu * (1 - rib) + HU_BONE * rib

    # --- pulmonary vessels: a few tapering tubes for texture ---------------
    n_vessel = rng.integers(5, 10)
    for _ in range(n_vessel):
        sgn = rng.choice([-1, 1])
        p0 = np.array([body_cz + jit(0.20 * ez),
                       body_cy - 0.04 * ey + jit(0.10 * ey),
                       body_cx + sgn * offx + jit(0.06 * ex)])
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        length = rng.uniform(0.10 * ez, 0.26 * ez)
        rad = rng.uniform(1.4, 3.0)
        t = np.stack([zz - p0[0], yy - p0[1], xx - p0[2]], -1)
        proj = np.clip((t * d).sum(-1), 0, length)
        perp = np.linalg.norm(t - proj[..., None] * d, axis=-1)
        taper = rad * (1 - 0.55 * proj / length)
        tube = np.clip(0.5 - (perp - taper) / voxel, 0, 1) * lungs
        hu = hu * (1 - tube) + HU_VESSEL * tube

    # --- nodules -----------------------------------------------------------
    # Place only where the sphere fits well inside lung parenchyma.
    dist = ndimage.distance_transform_edt(lung_core, sampling=grid.spacing)
    # `free` shrinks as nodules are placed, so they cannot overlap or merge.
    # Nodules that touch make per-nodule volumetry ill-posed -- they segment as
    # one blob -- and that is a modelling artefact, not a real evaluation
    # difficulty worth reproducing.
    free = lung_core.copy()
    mask = np.zeros(grid.shape, dtype=np.uint8)
    nodules = []
    for k in range(n_nodules):
        radius = float(rng.uniform(*nodule_radius_mm))
        ok = np.argwhere((dist > radius + 2.0 * voxel) & free)
        if ok.size == 0:
            break
        iz, iy, ix = ok[rng.integers(len(ok))]
        c = (grid.origin[0] + iz * grid.spacing[0],
             grid.origin[1] + iy * grid.spacing[1],
             grid.origin[2] + ix * grid.spacing[2])
        occ = _soft_ellipsoid(zz, yy, xx, c, (radius, radius, radius), voxel)
        hu = hu * (1 - occ) + HU_NODULE * occ
        mask[occ >= 0.5] = k + 1
        # Reserve a keep-out zone: this nodule's radius, plus room for the
        # largest nodule still to be placed, plus a separating margin.
        keep_out = radius + nodule_radius_mm[1] + 6.0
        free &= ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) > keep_out ** 2
        nodules.append({
            "label": k + 1,
            "centre_xyz": [float(c[2]), float(c[1]), float(c[0])],
            "radius_mm": radius,
            "true_volume_mm3": float(4.0 / 3.0 * np.pi * radius ** 3),
            "mask_volume_mm3": float((mask == k + 1).sum() * grid.voxel_volume_mm3),
        })

    mu = hu_to_mu(hu).astype(np.float32)
    meta = {"seed": int(seed), "nodules": nodules, "grid": grid.as_dict()}
    return mu, mask, meta


def grow_nodule(grid: Grid, mu: np.ndarray, nodule: dict, volume_gain: float):
    """Re-render one nodule with its volume scaled by `1 + volume_gain`.

    This is how the minimum-detectable-volume-change experiment is built: LIDC
    has no follow-up scans, so growth is synthesised on the ground truth and the
    projections are re-simulated.
    """
    zz, yy, xx = _coords(grid)
    voxel = float(min(grid.spacing))
    cx, cy, cz = nodule["centre_xyz"]
    r_old = nodule["radius_mm"]
    r_new = r_old * (1.0 + volume_gain) ** (1.0 / 3.0)

    out = mu.copy()
    # Erase the old nodule back to parenchyma, then draw the new one.
    old = _soft_ellipsoid(zz, yy, xx, (cz, cy, cx), (r_old, r_old, r_old), voxel)
    out = out * (1 - old) + hu_to_mu(HU_LUNG) * old
    new = _soft_ellipsoid(zz, yy, xx, (cz, cy, cx), (r_new, r_new, r_new), voxel)
    out = out * (1 - new) + hu_to_mu(HU_NODULE) * new

    mask = (new >= 0.5).astype(np.uint8)
    rec = dict(nodule)
    rec.update(radius_mm=r_new,
               true_volume_mm3=float(4.0 / 3.0 * np.pi * r_new ** 3),
               mask_volume_mm3=float(mask.sum() * grid.voxel_volume_mm3),
               volume_gain=float(volume_gain))
    return out.astype(np.float32), mask, rec


def erase_nodule(grid: Grid, mu: np.ndarray, nodule: dict):
    """Inpaint a nodule out of the ground truth, for the false-positive audit:
    if the prior invents it back, that is a hallucinated nodule."""
    zz, yy, xx = _coords(grid)
    voxel = float(min(grid.spacing))
    cx, cy, cz = nodule["centre_xyz"]
    r = nodule["radius_mm"]
    occ = _soft_ellipsoid(zz, yy, xx, (cz, cy, cx), (r * 1.05,) * 3, voxel)
    return (mu * (1 - occ) + hu_to_mu(HU_LUNG) * occ).astype(np.float32)


def lung_mask(mu: np.ndarray, grid: Grid, fill: bool = True) -> np.ndarray:
    """Lung *region* mask, used to restrict PSNR/SSIM and to bound nodule search.

    Holes are filled by default, which matters twice over. A raw
    air-density threshold excludes nodules and vessels, so (a) `psnr_lung` would
    be computed everywhere except the structures the project is about, and
    (b) a nodule would not lie inside its own lung mask, so restricting a
    segmentation to the mask would delete it. Filling makes this an anatomical
    region rather than a density band -- which is what standard lung
    segmentation produces.

    Whole-image PSNR is dominated by air, which every method reconstructs
    perfectly, so it flatters them all equally and discriminates nothing.
    """
    from .units import hu_to_mu as _h
    lo, hi = _h(-950.0), _h(-400.0)
    m = (mu > lo) & (mu < hi)
    m = ndimage.binary_closing(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n == 0:
        return m.astype(bool)
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:2] + 1
    m = np.isin(lbl, keep)
    if fill:
        m = ndimage.binary_fill_holes(m)
        # Fill slicewise too: a nodule touching the lung boundary leaves a
        # notch rather than an enclosed hole, which 3-D filling misses.
        for z in range(m.shape[0]):
            m[z] = ndimage.binary_fill_holes(m[z])
        m = ndimage.binary_closing(m, iterations=2)
    return m.astype(bool)
