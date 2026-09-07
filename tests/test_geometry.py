"""Gate G1: one canonical grid, and an affine that is never silently dropped."""
import numpy as np
import pytest
import torch

from medlift3d.geometry import CHEST, ROI, Grid, roi_grid_at, world_to_index_torch


def test_affine_and_spacing_round_trip(small_grid):
    assert Grid.spacing_from_affine(small_grid.affine) == pytest.approx(small_grid.spacing)


def test_spacing_from_affine_handles_oblique():
    """np.diag() is wrong for oblique affines; column norms are not."""
    theta = np.deg2rad(30.0)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    a = np.eye(4)
    a[:3, :3] = rot @ np.diag([0.7, 0.7, 2.5])   # (x, y, z)
    got = Grid.spacing_from_affine(a)
    assert got == pytest.approx((2.5, 0.7, 0.7), abs=1e-9)
    assert not np.allclose(np.abs(np.diag(a)[:3]), [0.7, 0.7, 2.5])  # the old bug


def test_world_to_norm_maps_corners_to_pm1(small_grid):
    c = small_grid.voxel_centres_xyz()
    n = small_grid.world_to_norm(torch.stack([c[0, 0, 0], c[-1, -1, -1]]))
    assert torch.allclose(n[0], torch.full((3,), -1.0), atol=1e-6)
    assert torch.allclose(n[1], torch.full((3,), 1.0), atol=1e-6)


def test_world_to_index_matches_grid_sample_convention(small_grid):
    """`world_to_index_torch` must agree with align_corners=True indexing."""
    c = small_grid.voxel_centres_xyz()
    idx = world_to_index_torch(small_grid, c.reshape(-1, 3)).reshape(*c.shape[:3], 3)
    nz, ny, nx = small_grid.shape
    assert idx[0, 0, 0].tolist() == pytest.approx([0, 0, 0], abs=1e-5)
    assert idx[-1, -1, -1].tolist() == pytest.approx([nz - 1, ny - 1, nx - 1], abs=1e-4)


def test_assert_matches_rejects_different_grid(small_grid):
    other = Grid(small_grid.shape, (1.0, 1.0, 1.0), small_grid.origin)
    with pytest.raises(AssertionError):
        small_grid.assert_matches(other)
    wrong_shape = Grid((10, 10, 10), small_grid.spacing, small_grid.origin)
    with pytest.raises(AssertionError):
        small_grid.assert_matches(wrong_shape)


def test_assert_matches_catches_the_fydp1_shape_bug():
    """A 324x65x94 reconstruction vs a 512x512x184 ground truth must not pass."""
    gt = Grid((184, 512, 512), (1.25, 0.7, 0.7))
    bad = Grid((324, 65, 94), (1.25, 5.0, 1.25))
    with pytest.raises(AssertionError, match="shape"):
        gt.assert_matches(bad)


def test_canonical_grids_are_centred():
    for g in (CHEST, ROI):
        assert g.centre_mm == pytest.approx((0.0, 0.0, 0.0))


def test_roi_grid_centres_on_request():
    g = roi_grid_at((10.0, -20.0, 30.0), (16, 16, 16), (1.0, 1.0, 1.0))
    cz, cy, cx = g.centre_mm
    assert (cx, cy, cz) == pytest.approx((10.0, -20.0, 30.0))


def test_grid_rejects_degenerate_input():
    with pytest.raises(ValueError):
        Grid((1, 10, 10), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        Grid((10, 10, 10), (0.0, 1.0, 1.0))


def test_serialisation_round_trip(small_grid):
    assert Grid.from_dict(small_grid.as_dict()) == small_grid
