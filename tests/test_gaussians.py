"""The Gaussian ROI parameterisation: correctness and the two FYDP-1 fixes."""
import numpy as np
import pytest
import torch

from medlift3d.gaussians import GaussianConfig, GaussianField, quat_to_rot
from medlift3d.geometry import Grid
from medlift3d.phantom import make_chest_phantom


@pytest.fixture(scope="module")
def field():
    grid = Grid.centred((24, 24, 24), (2.0, 2.0, 2.0))
    mu = torch.from_numpy(make_chest_phantom(grid, seed=0, n_nodules=1)[0])
    g = GaussianField(grid, GaussianConfig(n_init=1500, window=5)).init_from_volume(mu)
    return grid, mu, g


def test_quat_to_rot_is_a_rotation():
    torch.manual_seed(0)
    q = torch.randn(16, 4)
    R = quat_to_rot(q)
    assert torch.allclose(torch.det(R), torch.ones(16), atol=1e-5)
    eye = torch.eye(3).expand(16, 3, 3)
    assert torch.allclose(R @ R.transpose(-2, -1), eye, atol=1e-5)
    assert torch.allclose(quat_to_rot(torch.tensor([[1.0, 0, 0, 0]]))[0],
                          torch.eye(3), atol=1e-6)


def test_attenuation_is_non_negative_and_unbounded(field):
    """FYDP-1 used sigmoid, capping mu at 1.0 and saturating its own gradient."""
    _, _, g = field
    assert float(g.amp.detach().min()) >= 0.0
    g.raw_amp.data.fill_(20.0)
    assert float(g.amp.detach().min()) > 10.0, "softplus must not saturate"


def test_rasterize_is_non_negative_and_differentiable(field):
    grid, mu, g = field
    r = g.rasterize()
    assert r.shape == grid.shape
    assert bool((r >= 0).all())
    loss = (r - mu).pow(2).mean()
    loss.backward()
    for name, p in (("xyz", g.xyz), ("raw_amp", g.raw_amp), ("log_scale", g.log_scale)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, f"no grad for {name}"


def test_amplitude_calibration_reduces_error(field):
    grid, mu, _ = field
    g = GaussianField(grid, GaussianConfig(n_init=1500, window=5))
    g.init_from_volume(mu)          # calibration runs inside init
    after = float((g.rasterize() - mu).pow(2).mean().sqrt().detach())
    g.raw_amp.data = torch.log(torch.expm1((g.amp * 6.0).clamp_min(1e-8)))
    worse = float((g.rasterize() - mu).pow(2).mean().sqrt().detach())
    assert after < worse


def test_optimisation_converges(field):
    grid, mu, _ = field
    g = GaussianField(grid, GaussianConfig(n_init=1500, window=5)).init_from_volume(mu)
    opt = g.make_optimizer()
    start = float((g.rasterize() - mu).pow(2).mean().sqrt().detach())
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        ((g.rasterize() - mu).pow(2).mean() * 1e4 + g.regularisation()).backward()
        opt.step()
    end = float((g.rasterize() - mu).pow(2).mean().sqrt().detach())
    assert end < start, f"fit did not improve: {start:.5g} -> {end:.5g}"


def test_regularisation_penalises_anisotropy(field):
    grid, mu, _ = field
    g = GaussianField(grid, GaussianConfig(n_init=200, window=5)).init_from_volume(mu)
    iso = float(g.regularisation().detach())
    g.log_scale.data[:, 0] += 1.5      # stretch one axis
    assert float(g.regularisation().detach()) > iso


def test_prune_removes_dead_primitives(field):
    grid, mu, _ = field
    g = GaussianField(grid, GaussianConfig(n_init=400, window=5)).init_from_volume(mu)
    n0 = g.n
    g.raw_amp.data[:100] = -30.0       # softplus -> ~0
    assert g.prune(min_amp=1e-4) >= 100
    assert g.n < n0
