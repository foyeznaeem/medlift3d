"""Gate G4: metrics that mean something.

The FYDP-1 metrics thresholded the whole chest at a soft-tissue level and
Dice'd it against a few-hundred-voxel nodule mask, which pins Dice near 1e-4
regardless of reconstruction quality. The decisive property asserted here is
that a *perfect* reconstruction scores perfectly -- without it, a metric cannot
separate reconstruction error from its own bias.
"""
import numpy as np
import pytest
from scipy import ndimage

from medlift3d.geometry import Grid
from medlift3d.metrics import (adaptive_threshold, calibration, global_metrics,
                               hallucinated_at, measure_nodule_volume,
                               nodule_metrics, relative_volume_change)
from medlift3d.phantom import (erase_nodule, grow_nodule, lung_mask,
                               make_chest_phantom)


@pytest.fixture(scope="module")
def phantom():
    grid = Grid.centred((96, 96, 96), (1.5, 1.5, 1.5))
    mu, mask, meta = make_chest_phantom(grid, seed=3, n_nodules=3)
    return grid, mu, mask, meta


@pytest.fixture(scope="module")
def region(phantom):
    """Hole-filled lung mask, derived from ground-truth anatomy.

    Passing this is legitimate: it is an anatomical region of interest, not the
    quantity being measured. It stops a pleural nodule's segmentation from
    escaping into the chest wall, which they share a density with.
    """
    grid, mu, _, _ = phantom
    return lung_mask(mu, grid)


def test_lung_mask_contains_the_nodules(phantom, region):
    """A raw density-band mask excludes nodules, which would both delete them
    from a region-restricted segmentation and exclude them from `psnr_lung` --
    the two things the project is actually about."""
    _, _, mask, meta = phantom
    for n in meta["nodules"]:
        assert region[mask == n["label"]].all(), f"nodule {n['label']} outside lung mask"
    assert 0.02 < region.mean() < 0.35


def test_identity_reconstruction_is_perfect(phantom):
    grid, mu, mask, meta = phantom
    for n in meta["nodules"]:
        r = nodule_metrics(mu, mu, mask, grid, label=n["label"])
        assert r["nodule_dice"] == pytest.approx(1.0, abs=1e-6)
        assert r["volume_ape_pct"] == pytest.approx(0.0, abs=1e-6)
        assert r["detected"]


def test_sphere_volume_is_accurate(phantom, region):
    """Measured volume must match the analytic sphere volume within the
    partial-volume floor set by the voxel size.

    The tolerance is 30%, which is the honest discretisation floor at 1.5 mm
    voxels for nodules of this size plus the residual juxtavascular ambiguity --
    not a number chosen to make the test pass.
    """
    grid, mu, mask, meta = phantom
    for n in meta["nodules"]:
        if n["radius_mm"] < 3 * min(grid.spacing):
            continue    # below this, discretisation dominates by construction
        v = measure_nodule_volume(mu, n["centre_xyz"], grid, region=region)
        assert v == pytest.approx(n["true_volume_mm3"], rel=0.30)


def test_degradation_lowers_scores(phantom):
    grid, mu, mask, meta = phantom
    blur = ndimage.gaussian_filter(mu, 2.0)
    lab = max(meta["nodules"], key=lambda n: n["radius_mm"])["label"]
    good = nodule_metrics(mu, mu, mask, grid, label=lab)
    bad = nodule_metrics(blur, mu, mask, grid, label=lab)
    assert bad["nodule_dice"] < good["nodule_dice"]
    assert bad["volume_ape_pct"] > good["volume_ape_pct"]


def test_metrics_are_local_not_whole_volume(phantom):
    """Corrupting a distant part of the chest must not change nodule metrics.

    This is what "computed inside a dilated bounding box" buys, and what the
    whole-volume version could never provide.
    """
    grid, mu, mask, meta = phantom
    lab = meta["nodules"][0]["label"]
    base = nodule_metrics(mu, mu, mask, grid, label=lab)
    tampered = mu.copy()
    tampered[:6] = 0.02        # wreck a far-away slab
    after = nodule_metrics(tampered, mu, mask, grid, label=lab)
    assert after["nodule_dice"] == pytest.approx(base["nodule_dice"], abs=1e-6)


def test_hallucination_audit_distinguishes_present_from_absent(phantom, region):
    grid, mu, mask, meta = phantom
    for n in meta["nodules"]:
        assert hallucinated_at(mu, n["centre_xyz"], grid,
                               region=region)["hallucinated"] is True
        erased = erase_nodule(grid, mu, n)
        assert hallucinated_at(erased, n["centre_xyz"], grid,
                               region=region)["hallucinated"] is False


def test_measure_returns_zero_when_nothing_is_there(phantom, region):
    grid, mu, mask, meta = phantom
    n = max(meta["nodules"], key=lambda x: x["radius_mm"])
    assert measure_nodule_volume(erase_nodule(grid, mu, n), n["centre_xyz"],
                                 grid, region=region) == 0.0


def test_growth_is_measurable(phantom, region):
    """MDVC: measured volume change must track the true change.

    Residual error is voxel quantisation, which is the physical floor the MDVC
    curve exists to characterise -- so this asserts monotonicity plus a loose
    absolute band, not high accuracy.
    """
    grid, mu, mask, meta = phantom
    n = max(meta["nodules"], key=lambda x: x["radius_mm"])
    base, _, _ = grow_nodule(grid, mu, n, 0.0)
    v0 = measure_nodule_volume(base, n["centre_xyz"], grid, region=region)
    assert v0 > 0
    prev = -1e9
    for gain in (0.10, 0.25, 0.50):
        grown, _, _ = grow_nodule(grid, mu, n, gain)
        v1 = measure_nodule_volume(grown, n["centre_xyz"], grid, region=region)
        change = relative_volume_change(v0, v1)
        assert change == pytest.approx(gain * 100, abs=15.0)
        assert change > prev, "measured change must be monotonic in true change"
        prev = change


def test_adaptive_threshold_recovers_half_occupancy():
    """Midpoint of background and peak, which is the 50%-occupancy iso-level."""
    roi = np.concatenate([np.full(900, 0.001), np.full(100, 0.021)])
    assert adaptive_threshold(roi) == pytest.approx(0.011, abs=2e-3)


def test_global_metrics_identity_and_ordering(phantom):
    grid, mu, mask, meta = phantom
    lung = lung_mask(mu, grid)
    ident = global_metrics(mu, mu, lung)
    assert ident["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert np.isinf(ident["psnr"])
    rng = np.random.default_rng(0)
    worse = global_metrics(mu + rng.normal(0, 0.002, mu.shape).astype(np.float32),
                           mu, lung)
    better = global_metrics(mu + rng.normal(0, 0.0005, mu.shape).astype(np.float32),
                            mu, lung)
    assert better["psnr_lung"] > worse["psnr_lung"]
    assert better["ssim_lung"] > worse["ssim_lung"]


def test_calibration_detects_overconfidence(phantom):
    grid, mu, mask, meta = phantom
    lung = lung_mask(mu, grid)
    rng = np.random.default_rng(1)
    # Samples whose spread matches the true error -> well calibrated.
    truth = mu
    good = np.stack([truth + rng.normal(0, 0.001, mu.shape) for _ in range(12)])
    # Samples far too tight around a biased mean -> overconfident.
    bad = np.stack([truth + 0.004 + rng.normal(0, 1e-5, mu.shape) for _ in range(12)])
    c_good = calibration(good, truth, lung)
    c_bad = calibration(bad, truth, lung)
    assert c_good["ece"] < c_bad["ece"]
    assert c_bad["coverage@0.90"] < 0.5


def test_calibration_requires_multiple_samples(phantom):
    grid, mu, _, _ = phantom
    with pytest.raises(ValueError):
        calibration(mu[None], mu)
