"""The data-consistency step actually enforces the measurements.

The solver's whole claim is that the prior supplies anatomy and the measurement
operator supplies patient specificity. That second half is easy to break without
breaking anything visible: a step size that is too small by orders of magnitude
leaves the data term fully present in the code, running every iteration, and
inert in effect. These tests pin the effect, not the presence.
"""
import pytest
import torch

from medlift3d.geometry import Grid
from medlift3d.phantom import make_chest_phantom
from medlift3d.projector import ParallelGeometry, Projector
from medlift3d.solver import DiffusionSolver, SolverConfig, estimate_lipschitz
from medlift3d.units import MU_MAX, mu_to_net, net_to_mu


@pytest.fixture(scope="module")
def setup():
    grid = Grid.centred((16, 24, 24), (3.0, 3.0, 3.0))
    mu = torch.from_numpy(make_chest_phantom(grid, seed=0, n_nodules=1)[0])
    P = Projector(grid, ParallelGeometry.covering(grid, 12, 180.0, det_spacing=3.0))
    return grid, mu, P, P.fp(mu)


class _NullModel(torch.nn.Module):
    cfg = None


class _NullDiffusion(torch.nn.Module):
    """Stands in for the prior: the data term is what is under test."""
    model = _NullModel()


def test_data_consistency_drives_the_residual_down(setup):
    """The step size must be scaled by (dmu/dx)^2, not by dmu/dx.

    `mu = c*x + const` with `c = MU_MAX/2 ~ 0.0135`, so the objective's Hessian
    in network units is `c^2 A^T A`. Using `dc_step / L` instead of
    `dc_step / (c^2 L)` under-relaxes by ~5500x and the residual barely moves.
    """
    _, mu, P, p = setup
    s = DiffusionSolver(_NullDiffusion(), P, SolverConfig(dc_steps=100, progress=False))
    x0 = mu_to_net(torch.zeros_like(mu)).clamp(-1, 1)
    r0 = float((P.fp(net_to_mu(x0)) - p).pow(2).mean().sqrt())
    x1 = s._data_consistency(x0, p).clamp(-1, 1)
    r1 = float((P.fp(net_to_mu(x1)) - p).pow(2).mean().sqrt())
    assert r1 < 0.2 * r0, f"data consistency is inert: {r0:.4g} -> {r1:.4g}"


def test_data_consistency_is_stable(setup):
    """dc_step < 2 must not diverge: that is what deriving it from L buys."""
    _, mu, P, p = setup
    s = DiffusionSolver(_NullDiffusion(), P, SolverConfig(dc_steps=200, dc_step=1.5,
                                                         progress=False))
    x = s._data_consistency(mu_to_net(torch.zeros_like(mu)).clamp(-1, 1), p)
    assert torch.isfinite(x).all(), "data consistency diverged"


def test_lipschitz_bounds_the_operator(setup):
    """Power iteration must return an upper bound on ||A x||^2 / ||x||^2."""
    grid, mu, P, _ = setup
    L = estimate_lipschitz(P)
    x = torch.rand(grid.shape)
    assert float(P.fp(x).pow(2).sum()) <= L * float(x.pow(2).sum()) * 1.01


def test_solver_declines_conditioning_an_unconditional_prior(setup):
    """A prior with no cond channel must not be handed one silently."""
    _, _, P, _ = setup
    s = DiffusionSolver(_NullDiffusion(), P, SolverConfig(use_cond=True, progress=False))
    assert s.cond_ch == 0
    assert s.cfg.use_cond is False


def test_mu_max_is_the_clamp_the_solver_uses():
    assert SolverConfig().mu_max == MU_MAX
