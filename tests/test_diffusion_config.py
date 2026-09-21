"""Train/sample agreement: objective, schedule and conditioning.

Two mismatches in this family are silent and ruin a whole training run. The
objective one (train on eps, sample as x0) is caught by refusing a checkpoint
whose config differs. The conditioning one -- training on the clean target while
sampling on a blurry initialisation -- is caught by making the clean target
unreachable as a conditioning source and defaulting the prior to unconditional.
"""
import numpy as np
import pytest
import torch

from medlift3d.datasets import SliceDataset, save_case
from medlift3d.diffusion import DiffusionConfig, GaussianDiffusion
from medlift3d.geometry import Grid
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


def test_prior_is_unconditional_by_default():
    """Conditioning must be opted into, with something that exists at inference.

    The default used to be `cond_ch=1`, and the trainer filled that channel with
    the clean slice while the solver filled it with a blurry CGLS volume. The
    network learns to copy the channel, training loss looks excellent, and at
    inference it reproduces CGLS.
    """
    assert UNetConfig().cond_ch == 0
    net = UNet2D(UNetConfig(base_dim=8, dim_mults=(1, 2)))
    assert net.cfg.cond_ch == 0
    out = net(torch.randn(2, 1, 16, 16), torch.tensor([3, 7]))
    assert out.shape == (2, 1, 16, 16)


def _write_case(path, cond=None):
    grid = Grid.centred((4, 8, 8), (1.0, 1.0, 1.0))
    mu = np.full(grid.shape, 0.02, dtype=np.float32)
    save_case(path, mu, grid, conditioning=cond)


def test_slice_dataset_is_unconditional_unless_asked(tmp_path):
    _write_case(tmp_path / "c0.npz")
    assert "cond" not in SliceDataset(tmp_path)[0]


def test_slice_dataset_refuses_a_missing_conditioning_volume(tmp_path):
    """Better a loud KeyError than a silent fallback to the clean target."""
    _write_case(tmp_path / "c0.npz")
    with pytest.raises(KeyError, match="cond_cgls"):
        SliceDataset(tmp_path, cond_key="cgls")


def test_slice_dataset_serves_a_stored_conditioning_volume(tmp_path):
    grid = Grid.centred((4, 8, 8), (1.0, 1.0, 1.0))
    init = np.full(grid.shape, 0.01, dtype=np.float32)
    _write_case(tmp_path / "c0.npz", cond={"cgls": init})
    item = SliceDataset(tmp_path, cond_key="cgls")[0]
    assert item["cond"].shape == item["x"].shape
    assert not torch.equal(item["cond"], item["x"]), "conditioning must not be the target"
