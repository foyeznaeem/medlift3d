"""Gate G3: physical units. Catches every scale error in one assertion."""
import numpy as np
import pytest
import torch

from medlift3d.geometry import Grid
from medlift3d.phantom import make_water_cylinder
from medlift3d.projector import ParallelGeometry, Projector
from medlift3d.units import (HU_CLIP, MU_MAX, MU_WATER, hu_to_mu, mu_to_hu,
                             mu_to_net, net_to_mu)


def test_hu_mu_round_trip():
    for hu in (-1000.0, -800.0, -300.0, 0.0, 400.0):
        assert mu_to_hu(hu_to_mu(hu)) == pytest.approx(hu, abs=1e-6)


def test_air_is_zero_attenuation():
    assert hu_to_mu(-1000.0) == pytest.approx(0.0, abs=1e-12)


def test_water_is_mu_water():
    assert hu_to_mu(0.0) == pytest.approx(MU_WATER)


def test_hu_window_is_enforced():
    assert hu_to_mu(np.array([5000.0])) == pytest.approx(hu_to_mu(HU_CLIP[1]))
    assert hu_to_mu(np.array([-5000.0])) == pytest.approx(hu_to_mu(HU_CLIP[0]))


def test_network_normalisation_is_physical():
    """mu in [0, MU_MAX] maps exactly onto [-1, 1], which is what makes
    clamping a network output to [-1, 1] a physical constraint rather than an
    arbitrary one."""
    assert mu_to_net(0.0) == pytest.approx(-1.0)
    assert mu_to_net(MU_MAX) == pytest.approx(1.0)
    for mu in (0.0, 0.005, 0.02, MU_MAX):
        assert net_to_mu(mu_to_net(mu)) == pytest.approx(mu, abs=1e-9)


def test_water_cylinder_line_integral():
    """Peak line integral through a water cylinder must be mu_water * 2R.

    This single test catches HU/density confusion, spacing errors, ray-step
    normalisation errors and detector-scale errors together.
    """
    grid = Grid.centred((32, 64, 64), (2.0, 1.5, 1.5))
    radius = 30.0
    mu = torch.from_numpy(make_water_cylinder(grid, radius_mm=radius))
    P = Projector(grid, ParallelGeometry.covering(grid, 4, 180.0, det_spacing=1.0))
    p_max = float(P.fp(mu).max())
    assert p_max == pytest.approx(MU_WATER * 2 * radius, rel=0.02)


def test_poisson_noise_is_unbiased_ish():
    from medlift3d.units import apply_poisson
    p = np.full((4, 8, 8), 1.5, dtype=np.float32)
    noisy = apply_poisson(p, i0=1e6, rng=np.random.default_rng(0))
    assert noisy.mean() == pytest.approx(1.5, abs=0.01)
    assert noisy.std() > 0
