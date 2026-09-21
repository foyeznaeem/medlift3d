"""The two-scale ROI decomposition rests on `A` being linear. Assert it.

`mu = mu_bg + mu_roi` implies `A mu = A(mu_bg) + A(mu_roi)` exactly, so the
residual `p - A(mu_bg)` is precisely what the ROI has to explain. The same-grid
case is exact and is asserted; the fine-grid case additionally incurs resampling
error, which is *measured* here rather than assumed away.
"""
import pytest
import torch

from medlift3d.geometry import Grid, roi_grid_at
from medlift3d.phantom import make_chest_phantom
from medlift3d.projector import ParallelGeometry, Projector
from medlift3d.roi import (background_residual, resample, roi_index_box, zero_roi)


@pytest.fixture(scope="module")
def setup():
    grid = Grid.centred((32, 32, 32), (3.0, 3.0, 3.0))
    mu = torch.from_numpy(make_chest_phantom(grid, seed=0, n_nodules=1)[0])
    P = Projector(grid, ParallelGeometry.covering(grid, 8, 180.0, det_spacing=3.0))
    return grid, mu, P


def test_same_grid_decomposition_is_exact(setup):
    grid, mu, P = setup
    roi = roi_grid_at((0.0, 0.0, 0.0), (12, 12, 12), (3.0, 3.0, 3.0))
    box = roi_index_box(grid, roi)

    bg = zero_roi(mu, grid, roi)
    inside = torch.zeros_like(mu)
    inside[box] = mu[box]
    assert torch.equal(bg + inside, mu), "zero_roi must partition the volume"

    lhs = P.fp(mu)
    rhs = P.fp(bg) + P.fp(inside)
    tol = 1e-4 * float(lhs.abs().max())
    assert torch.allclose(lhs, rhs, atol=tol)

    resid = background_residual(lhs, bg, P)
    assert torch.allclose(resid, P.fp(inside), atol=tol), \
        "the residual IS the ROI's contribution"


def test_zero_roi_only_touches_the_box(setup):
    grid, mu, P = setup
    roi = roi_grid_at((0.0, 0.0, 0.0), (10, 10, 10), (3.0, 3.0, 3.0))
    box = roi_index_box(grid, roi)
    bg = zero_roi(mu, grid, roi)
    assert float(bg[box].abs().max()) == 0.0
    outside = torch.ones_like(mu, dtype=torch.bool)
    outside[box] = False
    assert torch.equal(bg[outside], mu[outside])


def test_roi_index_box_is_inside_the_grid(setup):
    grid, _, _ = setup
    for centre in [(0, 0, 0), (30, -30, 15), (-45, 45, -45)]:
        box = roi_index_box(grid, roi_grid_at(centre, (16, 16, 16), (1.5, 1.5, 1.5)))
        for ax, sl in enumerate(box):
            assert 0 <= sl.start < sl.stop <= grid.shape[ax]


def test_resample_preserves_values_on_identical_grid(setup):
    grid, mu, _ = setup
    assert torch.allclose(resample(mu, grid, grid), mu, atol=1e-5)


def test_resampler_is_accurate_on_a_smooth_field(setup):
    """Resampling accuracy, isolated from box-edge effects.

    Note this error does NOT enter the decomposition: `mu_bg` stays on the chest
    grid and is never resampled, so the residual is exact (see the test above).
    Resampling is used only to *initialise* the fine ROI. Measured on a smooth
    field the resampler is accurate to a few percent; on a box-truncated field
    the sharp edge dominates, which is a property of the edge, not the operator.
    """
    grid, mu, P = setup
    fine = roi_grid_at((0.0, 0.0, 0.0), (24, 24, 24), (1.5, 1.5, 1.5))

    # A smooth bump that fits comfortably inside the ROI: sigma = 4 mm, so
    # 3 sigma = 12 mm against the ROI's 34 mm extent. A wider bump would be
    # truncated by the ROI and the test would measure truncation, not the
    # resampler.
    c = grid.voxel_centres_xyz()
    r2 = (c ** 2).sum(-1)
    smooth = (0.02 * torch.exp(-r2 / (2 * 4.0 ** 2))).to(mu.dtype)

    fine_proj = Projector(fine, P.geometry, device=P.device, dtype=P.dtype)
    got = fine_proj.fp(resample(smooth, grid, fine))
    want = P.fp(smooth)
    rel = float((got - want).abs().max() / want.abs().max().clamp_min(1e-12))
    assert rel < 0.15, f"resampler inaccurate on a smooth field: {rel:.4f}"
