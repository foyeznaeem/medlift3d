"""Evaluation metrics.

Two rules drive this module.

1. **Nodule metrics are computed inside a dilated bounding box, never over the
   whole volume.** Thresholding a whole chest at a soft-tissue level selects the
   body wall, the spine and every vessel, so a whole-volume Dice against a
   few-hundred-voxel nodule mask is structurally pinned near zero and carries no
   information.

2. **PSNR/SSIM are restricted to the lung mask.** Whole-image PSNR on a chest CT
   is dominated by air, which every method reconstructs perfectly, so it flatters
   all of them equally and discriminates nothing.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity

from .geometry import Grid
from .units import MU_MAX, NODULE_MU_THRESHOLD


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def bbox_of(mask: np.ndarray) -> tuple[slice, slice, slice]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError("empty mask has no bounding box")
    lo = idx.min(0)
    hi = idx.max(0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def dilate_bbox(box, grid: Grid, margin_mm: float = 10.0):
    out = []
    for ax, sl in enumerate(box):
        pad = int(np.ceil(margin_mm / grid.spacing[ax]))
        out.append(slice(max(0, sl.start - pad), min(grid.shape[ax], sl.stop + pad)))
    return tuple(out)


def _select_component(binary: np.ndarray, centre_idx, max_dist_vox: float = 1e9):
    """The connected component at `centre_idx`.

    A component that *contains* the query point wins outright; otherwise the
    nearest centroid within `max_dist_vox` is used. Containment has to come
    first: when a nodule merges with an adjacent vessel or a neighbouring
    nodule, the merged blob's centroid can sit well away from the query point
    even though the blob plainly is the structure at that location.

    Selection never consults the ground-truth mask -- doing so would leak the
    answer into the metric. Centring on a known nodule location is legitimate;
    that is what the reading workflow provides.
    """
    lbl, n = ndimage.label(binary)
    if n == 0:
        return np.zeros_like(binary, dtype=bool), None
    cents = np.array(ndimage.center_of_mass(binary, lbl, range(1, n + 1)))
    q = np.asarray(centre_idx, dtype=float)

    qi = tuple(int(round(v)) for v in q)
    if all(0 <= qi[a] < binary.shape[a] for a in range(3)) and lbl[qi] > 0:
        k = int(lbl[qi]) - 1
        return lbl == (k + 1), cents[k]

    d = np.linalg.norm(cents - q, axis=1)
    k = int(np.argmin(d))
    if d[k] > max_dist_vox:
        return np.zeros_like(binary, dtype=bool), None
    return lbl == (k + 1), cents[k]


def adaptive_threshold(roi: np.ndarray, core_frac: float = 0.02,
                       bg_quantile: float = 0.10) -> float:
    """Midpoint threshold between local background and nodule peak.

    A fixed -300 HU cut is not consistent with a ">= 50% partial-volume"
    reference mask: a 30 HU nodule in -820 HU parenchyma only crosses -300 HU at
    ~61% occupancy, so a fixed threshold systematically under-segments and
    charges reconstruction for a bias that belongs to the segmenter. The
    background/peak midpoint recovers the 50%-occupancy iso-surface, which is
    what clinical nodule volumetry uses adaptive thresholding to achieve.
    """
    flat = np.sort(roi.reshape(-1))
    bg = float(np.quantile(flat, bg_quantile))
    n_core = max(1, int(core_frac * flat.size))
    fg = float(flat[-n_core:].mean())
    return 0.5 * (bg + fg)


def segment_nodule(mu_roi: np.ndarray, centre_idx, threshold: float | None = None,
                   max_dist_vox: float = 1e9, allowed: np.ndarray | None = None):
    """Threshold + connected-component selection inside an ROI crop.

    `threshold=None` uses `adaptive_threshold`; `allowed` restricts candidate
    voxels before labelling, which is what stops a component from escaping into
    adjacent anatomy.
    """
    src = mu_roi if allowed is None else np.where(allowed, mu_roi, -np.inf)
    thr = adaptive_threshold(mu_roi if allowed is None else mu_roi[allowed]) \
        if threshold is None else threshold
    return _select_component(src > thr, centre_idx, max_dist_vox)


def _elongation(binary: np.ndarray) -> float:
    """Max/min bounding-box extent. Tubular structures (vessels) score high."""
    idx = np.argwhere(binary)
    if idx.size == 0:
        return np.inf
    ext = idx.max(0) - idx.min(0) + 1
    return float(ext.max() / max(1, ext.min()))


# ----------------------------------------------------------------------------
# global fidelity
# ----------------------------------------------------------------------------

def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None,
         data_range: float = MU_MAX) -> float:
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim3d(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None,
           data_range: float = MU_MAX) -> float:
    """True 3-D SSIM (not per-slice 2-D averaged).

    With a mask, the full SSIM map is computed and then averaged over masked
    voxels only -- SSIM is a local statistic, so cropping first would change it.
    """
    _, smap = structural_similarity(gt.astype(np.float64), pred.astype(np.float64),
                                    data_range=data_range, full=True)
    return float(smap[mask].mean()) if mask is not None else float(smap.mean())


def mae_hu(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    from .units import mu_to_hu
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    return float(np.mean(np.abs(mu_to_hu(pred) - mu_to_hu(gt))))


def global_metrics(pred, gt, grid: Grid, lung: np.ndarray | None = None) -> dict:
    out = {
        "psnr": psnr(pred, gt),
        "ssim": ssim3d(pred, gt),
    }
    if lung is not None and lung.any():
        out["psnr_lung"] = psnr(pred, gt, lung)
        out["ssim_lung"] = ssim3d(pred, gt, lung)
        out["mae_hu_lung"] = mae_hu(pred, gt, lung)
    return out


# ----------------------------------------------------------------------------
# nodule / clinical
# ----------------------------------------------------------------------------

def nodule_metrics(pred_mu: np.ndarray, gt_mu: np.ndarray, gt_mask: np.ndarray,
                   grid: Grid, label: int = 1, margin_mm: float = 10.0,
                   threshold: float | None = None,
                   region: np.ndarray | None = None) -> dict:
    """Dice and volume error for one nodule, inside a dilated bounding box.

    Two families of number are returned, and they answer different questions:

    * `nodule_dice` / `volume_ape_pct` apply the *same* segmentation operator to
      the reconstruction and to the ground-truth `mu`. Segmentation bias cancels,
      so these isolate reconstruction-induced error and a perfect reconstruction
      scores 1.0 / 0%. These are the numbers that compare methods.
    * `*_vs_consensus` compare the reconstruction's segmentation against the
      reference consensus mask. These include the segmenter's own partial-volume
      bias and are the numbers that describe absolute clinical agreement.

    Reporting only the second family conflates two error sources; reporting only
    the first hides the absolute floor. Report both.
    """
    m = gt_mask == label
    if not m.any():
        raise ValueError(f"no voxels with label {label}")
    box = dilate_bbox(bbox_of(m), grid, margin_mm)
    ref_roi = m[box]
    pred_roi = pred_mu[box]
    gt_roi = gt_mu[box]
    centre = np.array(ndimage.center_of_mass(ref_roi))

    # One threshold, derived from the ground truth, applied to both volumes so
    # the comparison is of geometry rather than of intensity calibration.
    thr = adaptive_threshold(gt_roi) if threshold is None else threshold
    allowed = region[box] if region is not None else None
    seg_pred, cent = segment_nodule(pred_roi, centre, thr, allowed=allowed)
    seg_gt, _ = segment_nodule(gt_roi, centre, thr, allowed=allowed)

    vox = grid.voxel_volume_mm3
    v_pred = float(seg_pred.sum() * vox)
    v_gt = float(seg_gt.sum() * vox)
    v_ref = float(ref_roi.sum() * vox)

    def _dice(a, b):
        return float(2.0 * (a & b).sum() / (a.sum() + b.sum() + 1e-8))

    res = {
        "label": int(label),
        "nodule_dice": _dice(seg_pred, seg_gt),
        "volume_ape_pct": abs(v_pred - v_gt) / (v_gt + 1e-8) * 100.0,
        "dice_vs_consensus": _dice(seg_pred, ref_roi),
        "ape_vs_consensus_pct": abs(v_pred - v_ref) / (v_ref + 1e-8) * 100.0,
        "pred_vol_mm3": v_pred,
        "gt_seg_vol_mm3": v_gt,
        "ref_vol_mm3": v_ref,
        "detected": bool(seg_pred.any()),
        "threshold_mu": float(thr),
        "gt_diameter_mm": float(2.0 * (3.0 * v_ref / (4.0 * np.pi)) ** (1.0 / 3.0)),
    }
    if cent is not None:
        err = (cent - centre) * np.array(grid.spacing)
        res["centroid_error_mm"] = float(np.linalg.norm(err))
    return res


def _sphere_mask(shape, centre_idx, radius_mm: float, spacing) -> np.ndarray:
    """Ball of `radius_mm` around `centre_idx`, in index space."""
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    r2 = sum(((g - c) * sp) ** 2 for g, c, sp in zip(grids, centre_idx, spacing))
    return r2 <= radius_mm ** 2


def _roi_box(centre_xyz, grid: Grid, margin_mm: float):
    idx = grid.world_to_index(np.asarray(centre_xyz)[None])[0]
    pad = [int(np.ceil(margin_mm / s)) for s in grid.spacing]
    box = tuple(slice(max(0, int(round(idx[a])) - pad[a]),
                      min(grid.shape[a], int(round(idx[a])) + pad[a] + 1)) for a in range(3))
    centre = np.array([idx[a] - box[a].start for a in range(3)])
    return box, centre


def measure_nodule_volume(pred_mu: np.ndarray, centre_xyz, grid: Grid,
                          max_diameter_mm: float = 40.0,
                          threshold: float | None = None,
                          region: np.ndarray | None = None,
                          seed_radius_mm: float = 3.0,
                          require_detection: bool = True) -> float:
    """Volume in mm^3 of the nodule at `centre_xyz`, using no ground truth.

    Used by the minimum-detectable-volume-change experiment, where the identical
    measurement must be applied to a baseline and a follow-up reconstruction.

    Three constraints keep this well-posed, and each was a bug first.

    * **Detection and measurement need different thresholds.** Adaptive
      thresholding presupposes a nodule is present; on pure parenchyma its
      background/peak midpoint lands inside the noise and it will segment
      nothing into something. Gate on the fixed clinical threshold, then measure
      adaptively. Returns 0.0 when nothing is detected.
    * **A geometric bound.** At a fixed soft-tissue threshold a nodule near the
      pleura is connected to the chest wall, so the component containing the
      seed can be the whole body. Candidates are confined to a ball of
      `max_diameter_mm / 2`: by definition a pulmonary nodule is <= 30 mm
      (larger is a mass), so the measurement cannot run away.
    * **An optional anatomical bound**, `region` -- normally a hole-filled lung
      mask. This severs the connection to the chest wall properly rather than
      merely bounding it. Pass the mask derived from the ground-truth anatomy:
      it is an anatomical region of interest, not the answer being measured.

    A fourth constraint: the selected component must actually touch the seed.
    Without it, erasing a nodule and re-measuring returns the volume of whatever
    vessel fragment happens to be nearest -- a false positive dressed up as a
    measurement.

    Nodules fused to a vessel of the same density (juxtavascular nodules) remain
    genuinely ambiguous for threshold-based volumetry; dedicated vessel-aware
    segmentation is out of scope and this is stated as a limitation.
    """
    half = max_diameter_mm / 2.0 + 4.0
    box, centre = _roi_box(centre_xyz, grid, half)
    roi = pred_mu[box]

    allowed = _sphere_mask(roi.shape, centre, max_diameter_mm / 2.0, grid.spacing)
    if region is not None:
        allowed &= region[box]
    seed_ball = _sphere_mask(roi.shape, centre, seed_radius_mm, grid.spacing)

    detect, _ = segment_nodule(roi, centre, NODULE_MU_THRESHOLD, allowed=allowed)
    if not detect.any() or not (detect & seed_ball).any():
        return 0.0
    seg, _ = segment_nodule(roi, centre, threshold, allowed=allowed)
    if not (seg & seed_ball).any():
        return 0.0
    return float(seg.sum() * grid.voxel_volume_mm3)


def hallucinated_at(pred_mu: np.ndarray, centre_xyz, grid: Grid,
                    min_diameter_mm: float = 4.0, max_elongation: float = 3.0,
                    margin_mm: float = 16.0, max_dist_mm: float = 6.0,
                    max_diameter_mm: float = 40.0,
                    region: np.ndarray | None = None) -> dict:
    """Did a nodule-like blob appear where the ground truth has none?

    Run on a reconstruction whose nodule was erased from the ground truth
    (`phantom.erase_nodule`). A blob found here is invented by the prior, and
    the rate over a test set is the false-positive nodule rate -- the number a
    clinician cares about, given the 96% LDCT false-positive rate this project
    is motivated by.

    Detection uses the FIXED clinical threshold, never the adaptive one: this is
    a detection question, and parenchyma at -820 HU does not cross -300 HU. The
    elongation filter and the distance gate suppress vessels, which share the
    nodule's density but neither its shape nor its exact location.
    """
    box, centre = _roi_box(centre_xyz, grid, min(margin_mm, max_diameter_mm / 2.0))
    roi = pred_mu[box]
    allowed = _sphere_mask(roi.shape, centre, max_diameter_mm / 2.0, grid.spacing)
    if region is not None:
        allowed &= region[box]
    # Distance-gated: a vessel 10 mm away is not a nodule *at this location*.
    seg, _ = segment_nodule(roi, centre, NODULE_MU_THRESHOLD, allowed=allowed,
                            max_dist_vox=max_dist_mm / min(grid.spacing))
    max_vol = 4.0 / 3.0 * np.pi * (max_diameter_mm / 2.0) ** 3
    min_vol = 4.0 / 3.0 * np.pi * (min_diameter_mm / 2.0) ** 3
    v = float(seg.sum() * grid.voxel_volume_mm3)
    elong = _elongation(seg)
    return {
        "hallucinated": bool(min_vol <= v <= max_vol and elong <= max_elongation),
        "blob_volume_mm3": v,
        "elongation": float(elong) if np.isfinite(elong) else None,
    }


def count_nodule_like(pred_mu: np.ndarray, grid: Grid, lung: np.ndarray,
                      min_diameter_mm: float = 4.0, max_elongation: float = 3.0,
                      threshold: float | None = None) -> int:
    """Count compact nodule-like blobs inside lung parenchyma.

    A whole-lung count, unlike `hallucinated_at`, and inherently noisier:
    vessels share nodule density, so the elongation filter does real work here.
    Prefer `hallucinated_at` for the controlled audit: it is localised, so its
    threshold is unambiguous and its false-positive rate is interpretable.
    """
    thr = NODULE_MU_THRESHOLD if threshold is None else threshold
    cand = ndimage.binary_opening((pred_mu > thr) & lung, iterations=1)
    lbl, n = ndimage.label(cand)
    if n == 0:
        return 0
    vox = grid.voxel_volume_mm3
    min_vol = 4.0 / 3.0 * np.pi * (min_diameter_mm / 2.0) ** 3
    count = 0
    for i in range(1, n + 1):
        comp = lbl == i
        if comp.sum() * vox >= min_vol and _elongation(comp) <= max_elongation:
            count += 1
    return count


# ----------------------------------------------------------------------------
# uncertainty calibration
# ----------------------------------------------------------------------------

def calibration(samples: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None,
                levels=(0.5, 0.68, 0.8, 0.9, 0.95)) -> dict:
    """Empirical coverage of the posterior-sample interval, plus its ECE.

    `samples` is [K, ...]. A well-calibrated uncertainty map has empirical
    coverage close to the nominal level at every level.
    """
    k = samples.shape[0]
    if k < 2:
        raise ValueError("calibration needs at least 2 posterior samples")
    flat = samples.reshape(k, -1)
    g = gt.reshape(-1)
    if mask is not None:
        sel = mask.reshape(-1)
        flat, g = flat[:, sel], g[sel]

    cov, ece = {}, 0.0
    for lv in levels:
        lo = np.quantile(flat, (1 - lv) / 2, axis=0)
        hi = np.quantile(flat, 1 - (1 - lv) / 2, axis=0)
        c = float(np.mean((g >= lo) & (g <= hi)))
        cov[f"coverage@{lv:.2f}"] = c
        ece += abs(c - lv)
    cov["ece"] = float(ece / len(levels))
    cov["mean_std"] = float(flat.std(0).mean())
    return cov


def relative_volume_change(v_base: float, v_follow: float) -> float:
    """Percent volume change, the quantity Volume Doubling Time is derived from."""
    return (v_follow - v_base) / (v_base + 1e-8) * 100.0
