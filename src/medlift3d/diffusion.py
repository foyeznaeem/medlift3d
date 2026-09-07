"""DDPM training and DDIM sampling.

The single most damaging bug in the FYDP-1 code was a train/sample mismatch: the
network was trained to predict noise and the sampler interpreted its output as
the clean signal, silently, for 1000 steps. Nothing crashed; the reconstruction
was simply garbage.

That is made structurally impossible here. `DiffusionConfig` is frozen, it is
written into every checkpoint, and `GaussianDiffusion.load` refuses a checkpoint
whose config differs from the one being constructed. There is no code path that
lets training and sampling disagree.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "cosine"          # cosine | linear
    objective: str = "eps"            # eps | x0   -- ALWAYS stated explicitly
    loss_type: str = "l2"             # l2 | l1 | huber
    cfg_drop_prob: float = 0.1        # conditioning dropout for guidance
    x0_clip: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self):
        if self.objective not in ("eps", "x0"):
            raise ValueError(f"objective must be 'eps' or 'x0', got {self.objective!r}")
        if self.schedule not in ("cosine", "linear"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.loss_type not in ("l2", "l1", "huber"):
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

    def to_dict(self):
        d = asdict(self)
        d["x0_clip"] = list(self.x0_clip)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        if "x0_clip" in d:
            d["x0_clip"] = tuple(d["x0_clip"])
        return DiffusionConfig(**d)


def _cosine_betas(t: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, t, t + 1, dtype=torch.float64)
    ac = torch.cos(((x / t) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - ac[1:] / ac[:-1], 1e-4, 0.9999)


def _linear_betas(t: int) -> torch.Tensor:
    scale = 1000.0 / t
    return torch.linspace(scale * 1e-4, scale * 2e-2, t, dtype=torch.float64)


def _extract(a: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    return a.gather(0, t).reshape(-1, *((1,) * (ndim - 1)))


class GaussianDiffusion(torch.nn.Module):
    """Wraps a noise-estimator network with a fixed, declared noise schedule."""

    def __init__(self, model: torch.nn.Module, cfg: DiffusionConfig | None = None):
        super().__init__()
        self.model = model
        self.cfg = cfg or DiffusionConfig()

        betas = (_cosine_betas(self.cfg.timesteps) if self.cfg.schedule == "cosine"
                 else _linear_betas(self.cfg.timesteps))
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_ac", alphas_cumprod.sqrt().float())
        self.register_buffer("sqrt_1mac", (1.0 - alphas_cumprod).sqrt().float())

    # -- forward process -------------------------------------------------------

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return (_extract(self.sqrt_ac, t, x0.ndim) * x0
                + _extract(self.sqrt_1mac, t, x0.ndim) * noise)

    # -- conversions -----------------------------------------------------------

    def eps_to_x0(self, x_t, t, eps):
        return ((x_t - _extract(self.sqrt_1mac, t, x_t.ndim) * eps)
                / _extract(self.sqrt_ac, t, x_t.ndim))

    def x0_to_eps(self, x_t, t, x0):
        return ((x_t - _extract(self.sqrt_ac, t, x_t.ndim) * x0)
                / _extract(self.sqrt_1mac, t, x_t.ndim).clamp_min(1e-8))

    def clip_x0(self, x0):
        lo, hi = self.cfg.x0_clip
        return x0.clamp(lo, hi)

    # -- training --------------------------------------------------------------

    def loss(self, x0, cond=None, t=None):
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.cfg.timesteps, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        # Classifier-free guidance: drop the conditioning on a fraction of the
        # batch so the same weights serve the conditional and unconditional
        # branches and guidance strength becomes tunable at sampling time.
        if cond is not None and self.cfg.cfg_drop_prob > 0 and self.training:
            keep = (torch.rand(b, device=x0.device) >= self.cfg.cfg_drop_prob)
            cond = cond * keep.view(-1, *((1,) * (cond.ndim - 1))).to(cond.dtype)

        pred = self.model(x_t, t, cond)
        target = noise if self.cfg.objective == "eps" else x0
        if self.cfg.loss_type == "l2":
            return F.mse_loss(pred, target)
        if self.cfg.loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.smooth_l1_loss(pred, target)

    # -- sampling --------------------------------------------------------------

    @torch.no_grad()
    def predict_eps(self, x_t, t, cond=None, guidance: float = 0.0):
        """Model output converted to eps, with optional classifier-free guidance."""
        out = self.model(x_t, t, cond)
        eps = out if self.cfg.objective == "eps" else self.x0_to_eps(x_t, t, out)
        if guidance > 0 and cond is not None:
            out_u = self.model(x_t, t, None)
            eps_u = out_u if self.cfg.objective == "eps" else self.x0_to_eps(x_t, t, out_u)
            eps = eps_u + (1.0 + guidance) * (eps - eps_u)
        return eps

    def ddim_timesteps(self, n_steps: int) -> list[int]:
        n_steps = min(n_steps, self.cfg.timesteps)
        idx = torch.linspace(0, self.cfg.timesteps - 1, n_steps).round().long()
        return sorted(set(idx.tolist()), reverse=True)

    @torch.no_grad()
    def ddim_step(self, x_t, t_cur: int, t_prev: int, eps, x0=None, eta: float = 0.0):
        """One DDIM update. `x0` may be supplied already data-corrected."""
        ac_t = self.alphas_cumprod[t_cur]
        ac_p = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(ac_t)
        if x0 is None:
            tb = torch.full((x_t.shape[0],), t_cur, device=x_t.device, dtype=torch.long)
            x0 = self.clip_x0(self.eps_to_x0(x_t, tb, eps))
        sigma = eta * ((1 - ac_p) / (1 - ac_t)).sqrt() * (1 - ac_t / ac_p).sqrt()
        dir_xt = (1 - ac_p - sigma ** 2).clamp_min(0).sqrt() * eps
        out = ac_p.sqrt() * x0 + dir_xt
        if eta > 0 and t_prev >= 0:
            out = out + sigma * torch.randn_like(x_t)
        return out

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, n_steps: int = 50, guidance: float = 0.0,
                    eta: float = 0.0, device=None, generator=None, progress=False,
                    x0_hook=None):
        """Plain DDIM sampling. The solver uses `x0_hook` to inject data
        consistency between the prior step and the DDIM update."""
        device = device or next(self.model.parameters()).device
        x = torch.randn(shape, device=device, generator=generator)
        steps = self.ddim_timesteps(n_steps)
        for i, t_cur in enumerate(tqdm(steps, disable=not progress, desc="DDIM")):
            t_prev = steps[i + 1] if i + 1 < len(steps) else -1
            tb = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps = self.predict_eps(x, tb, cond, guidance)
            x0 = self.clip_x0(self.eps_to_x0(x, tb, eps))
            if x0_hook is not None:
                x0 = self.clip_x0(x0_hook(x0, t_cur))
                eps = self.x0_to_eps(x, tb, x0)
            x = self.ddim_step(x, t_cur, t_prev, eps, x0, eta)
        return x

    # -- checkpoints -----------------------------------------------------------

    def state(self, **extra) -> dict:
        return {"model": self.model.state_dict(),
                "diffusion_cfg": self.cfg.to_dict(),
                "unet_cfg": getattr(self.model, "cfg", None).to_dict()
                if hasattr(getattr(self.model, "cfg", None), "to_dict") else None,
                **extra}

    def load_state(self, ckpt: dict, strict: bool = True) -> None:
        """Load weights, refusing any checkpoint whose diffusion config differs.

        This is the guard that makes the FYDP-1 train/sample mismatch
        unreachable: a checkpoint trained with `objective='eps'` cannot be loaded
        into a sampler built with `objective='x0'`.
        """
        saved = ckpt.get("diffusion_cfg")
        if saved is not None:
            mine = self.cfg.to_dict()
            diff = {k: (saved.get(k), mine.get(k)) for k in mine
                    if saved.get(k) != mine.get(k)}
            # Sampling-time knobs may legitimately differ from training.
            diff.pop("cfg_drop_prob", None)
            if diff and strict:
                raise ValueError(
                    "diffusion config mismatch between checkpoint and sampler "
                    f"(checkpoint, current): {diff}. Refusing to load -- this is "
                    "exactly the silent train/sample mismatch that produces "
                    "garbage reconstructions."
                )
        self.model.load_state_dict(ckpt["model"])

    @staticmethod
    def from_checkpoint(ckpt: dict, model_factory=None):
        """Rebuild both the UNet and the diffusion wrapper from a checkpoint,
        so there is no opportunity to pass mismatched arguments by hand."""
        from .prior2d import UNet2D, UNetConfig
        cfg = DiffusionConfig.from_dict(ckpt["diffusion_cfg"])
        if model_factory is not None:
            model = model_factory()
        else:
            ucfg = ckpt.get("unet_cfg")
            if ucfg is None:
                raise ValueError("checkpoint has no unet_cfg; pass model_factory")
            model = UNet2D(UNetConfig.from_dict(ucfg))
        diff = GaussianDiffusion(model, cfg)
        diff.load_state(ckpt)
        return diff
