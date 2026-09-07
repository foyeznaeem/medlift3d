"""Regression test for showstopper S2.

FYDP-1 trained with `objective='pred_noise'` and sampled with the constructor
default `objective='pred_x0'`, so the sampler fed noise into the slot where the
clean signal belonged, 1000 times per reconstruction, silently.
"""
import pytest
import torch

from medlift3d.diffusion import DiffusionConfig, GaussianDiffusion
from medlift3d.prior2d import UNet2D, UNetConfig


def _make(objective="eps", timesteps=100):
    return GaussianDiffusion(UNet2D(UNetConfig(base_dim=8, dim_mults=(1, 2))),
                             DiffusionConfig(timesteps=timesteps, objective=objective))


def test_objective_must_be_stated_and_validated():
    with pytest.raises(ValueError):
        DiffusionConfig(objective="pred_noise")   # the FYDP-1 spelling
    with pytest.raises(ValueError):
        DiffusionConfig(schedule="quadratic")
    with pytest.raises(ValueError):
        DiffusionConfig(loss_type="mse")


def test_matching_config_loads():
    src = _make("eps")
    _make("eps").load_state(src.state())


def test_objective_mismatch_is_refused():
    """THE regression test for S2."""
    ck = _make("eps").state()
    with pytest.raises(ValueError, match="mismatch"):
        _make("x0").load_state(ck)


def test_timestep_mismatch_is_refused():
    ck = _make("eps", timesteps=100).state()
    with pytest.raises(ValueError, match="mismatch"):
        _make("eps", timesteps=250).load_state(ck)


def test_from_checkpoint_rebuilds_everything():
    src = _make("eps", 100)
    got = GaussianDiffusion.from_checkpoint(src.state())
    assert got.cfg == src.cfg
    assert got.model.cfg.to_dict() == src.model.cfg.to_dict()
    for a, b in zip(got.model.state_dict().values(), src.model.state_dict().values()):
        assert torch.equal(a, b)


def test_eps_x0_conversions_are_inverse():
    d = _make()
    x0 = torch.rand(2, 1, 16, 16) * 2 - 1
    eps = torch.randn_like(x0)
    t = torch.tensor([10, 60])
    x_t = d.q_sample(x0, t, eps)
    assert torch.allclose(d.eps_to_x0(x_t, t, eps), x0, atol=1e-5)
    assert torch.allclose(d.x0_to_eps(x_t, t, x0), eps, atol=1e-5)


def test_x0_clip_is_the_physical_range():
    d = _make()
    assert d.cfg.x0_clip == (-1.0, 1.0)
    assert float(d.clip_x0(torch.tensor([-5.0, 5.0])).abs().max()) == 1.0


def test_ddim_timesteps_are_descending_and_unique():
    d = _make(timesteps=100)
    ts = d.ddim_timesteps(20)
    assert ts == sorted(set(ts), reverse=True)
    assert len(ts) <= 20 and ts[0] < 100
