"""Reconstruction: a 2-D diffusion prior alternated with hard data consistency.

Structure follows DiffusionMBIR (Chung et al., CVPR 2023): a purely 2-D slice
prior supplies anatomy, the measurement operator supplies patient specificity,
and a z-direction TV term supplies inter-slice coherence. That is what lets the
whole thing fit on one 16 GB card.

Each reverse-diffusion step is:

    eps  <- prior(x_t, t, cond)             # slicewise, batched over z
    x0   <- (x_t - sqrt(1-a_t) eps) / sqrt(a_t)
    for j in 1..M:                          # data consistency
        x0 <- x0 - lr * (A^T(A x0 - p) + lam_z * dTV_z(x0))
    x0   <- clamp(x0, physical range)
    x_t-1 <- ddim(x_t, x0, eps)

Two properties make the data term real rather than decorative:

* `Projector.bp` is the exact adjoint of `Projector.fp` (asserted by
  `tests/test_adjoint.py`), so `A^T(A x0 - p)` is the true gradient of
  `0.5||A x0 - p||^2`. Without that the projection loss is a constant and the
  regulariser is the whole objective, which converges very stably to mush.
* The step size is derived, not guessed. `L` is the largest eigenvalue of
  `A^T A` from power iteration, so one `dc_step` behaves consistently across
  grid sizes, view counts and geometries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from tqdm.auto import tqdm

from .diffusion import GaussianDiffusion
from .projector import Projector
from .units import MU_MAX, mu_to_net, net_to_mu
from .utils import tv_grad_z


@dataclass
class SolverConfig:
    n_steps: int = 50              # DDIM steps (not 1000 -- see docs/REVIEW.md C2)
    dc_steps: int = 2              # data-consistency steps per diffusion step
    dc_step: float = 0.9           # relative to 1/L, so ~1 means a full CG-ish step
    lam_z: float = 0.03            # z-direction TV weight, relative to the data grad
    guidance: float = 0.0          # classifier-free guidance strength
    eta: float = 0.0               # DDIM stochasticity; > 0 diversifies samples
    slice_batch: int = 16          # z-slices per prior forward pass
    n_posterior: int = 8           # K posterior samples -> mean + per-voxel std
    warm_start_t: float = 0.7      # begin at this fraction of the schedule
    use_cond: bool = True          # feed the initialisation as a channel, if the
                                   # prior was trained with one
    mu_max: float = MU_MAX
    progress: bool = True

    def to_dict(self):
        return asdict(self)


def estimate_lipschitz(projector: Projector, n_iter: int = 12, seed: int = 0) -> float:
    """Largest eigenvalue of `A^T A` by power iteration.

    Sets the data-consistency step size so `dc_step` is dimensionless and
    transfers across geometries instead of needing a per-experiment retune.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(projector.grid.shape, generator=g).to(projector.device, projector.dtype)
    v /= v.norm().clamp_min(1e-12)
    lam = 1.0
    with torch.no_grad():
        for _ in range(n_iter):
            w = projector.bp(projector.fp(v))
            lam = float(w.norm())
            if lam < 1e-20:
                return 1.0
            v = w / lam
    return lam


class DiffusionSolver:
    def __init__(self, diffusion: GaussianDiffusion, projector: Projector,
                 cfg: SolverConfig | None = None, lipschitz: float | None = None):
        self.diffusion = diffusion
        self.projector = projector
        self.cfg = cfg or SolverConfig()
        self.device = projector.device
        self.L = lipschitz if lipschitz is not None else estimate_lipschitz(projector)

        # The prior decides whether conditioning exists; the solver may only
        # decline it. Handing a cond channel to a network that never had one is
        # silent, so take the answer from the checkpoint rather than the config.
        model_cfg = getattr(getattr(diffusion, "model", None), "cfg", None)
        self.cond_ch = int(getattr(model_cfg, "cond_ch", 0))
        if self.cfg.use_cond and not self.cond_ch:
            self.cfg = replace(self.cfg, use_cond=False)

    # -- the prior, applied slicewise -----------------------------------------

    @torch.no_grad()
    def _prior_eps(self, x: torch.Tensor, t: int, cond: torch.Tensor | None):
        """x: [nz, ny, nx] in network units. Returns eps of the same shape.

        The 2-D prior is run on axial slices batched over z, so peak memory is
        set by `slice_batch`, not by the volume.
        """
        nz = x.shape[0]
        out = torch.empty_like(x)
        for i in range(0, nz, self.cfg.slice_batch):
            j = min(i + self.cfg.slice_batch, nz)
            xb = x[i:j].unsqueeze(1)                      # [b,1,ny,nx]
            cb = cond[i:j].unsqueeze(1) if cond is not None else None
            tb = torch.full((j - i,), t, device=x.device, dtype=torch.long)
            out[i:j] = self.diffusion.predict_eps(xb, tb, cb, self.cfg.guidance).squeeze(1)
        return out

    # -- data consistency ------------------------------------------------------

    def _data_consistency(self, x0_net: torch.Tensor, projs: torch.Tensor):
        """Gradient steps on 0.5||A mu - p||^2 + lam_z * TV_z, in network units.

        The chain rule has to be applied to the step size as well as to the
        gradient. With `mu = c * x + const` and `c = MU_MAX / 2`, the objective's
        Hessian in network units is `c^2 A^T A`, so the stable step is
        `dc_step / (c^2 L)` -- not `dc_step / L`. Using the latter under-relaxes
        by 1/c^2 ~ 5500x, which leaves the data term visibly present in the code
        and inert in effect: 100 steps then cut the projection residual by ~1%
        where a correctly scaled step cuts it by ~99%.
        """
        c = self.cfg.mu_max / 2.0
        lr = self.cfg.dc_step / max(self.L * c * c, 1e-12)
        x = x0_net
        for _ in range(self.cfg.dc_steps):
            g = self.projector.bp(self.projector.fp(net_to_mu(x)) - projs) * c
            if self.cfg.lam_z > 0:
                gz = tv_grad_z(x, axis=0)
                scale = g.abs().mean().clamp_min(1e-12) / gz.abs().mean().clamp_min(1e-12)
                g = g + self.cfg.lam_z * scale * gz
            x = x - lr * g
        return x

    # -- one posterior sample --------------------------------------------------

    @torch.no_grad()
    def sample_once(self, projs: torch.Tensor, init_mu: torch.Tensor,
                    seed: int = 0, progress: bool | None = None) -> torch.Tensor:
        """One posterior sample. Returns `mu` on the projector's grid."""
        d = self.diffusion
        cfg = self.cfg
        progress = cfg.progress if progress is None else progress

        cond = mu_to_net(init_mu).clamp(-1, 1) if cfg.use_cond else None

        steps = d.ddim_timesteps(cfg.n_steps)
        # Warm start: begin partway down the schedule from the classical
        # reconstruction rather than from pure noise.
        t_start = int(cfg.warm_start_t * (d.cfg.timesteps - 1))
        steps = [s for s in steps if s <= t_start] or [steps[-1]]

        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(init_mu.shape, generator=gen).to(self.device, self.projector.dtype)
        t0 = torch.full((1,), steps[0], device=self.device, dtype=torch.long)
        x = d.q_sample(mu_to_net(init_mu).clamp(-1, 1)[None], t0, noise[None])[0]

        x0 = x
        bar = tqdm(range(len(steps)), desc="solve", disable=not progress, leave=False)
        for i in bar:
            t_cur = steps[i]
            t_prev = steps[i + 1] if i + 1 < len(steps) else -1
            eps = self._prior_eps(x, t_cur, cond)

            tb = torch.full((1,), t_cur, device=self.device, dtype=torch.long)
            x0 = d.clip_x0(d.eps_to_x0(x[None], tb, eps[None])[0])
            x0 = d.clip_x0(self._data_consistency(x0, projs))
            eps = d.x0_to_eps(x[None], tb, x0[None])[0]

            x = d.ddim_step(x[None], t_cur, t_prev, eps[None], x0[None], cfg.eta)[0]
            if progress:
                with torch.no_grad():
                    r = float((self.projector.fp(net_to_mu(x0)) - projs).pow(2).mean().sqrt())
                bar.set_postfix(t=t_cur, rmse=f"{r:.4g}")

        return net_to_mu(d.clip_x0(x0)).clamp(0.0, cfg.mu_max)

    # -- posterior -------------------------------------------------------------

    def reconstruct(self, projs: torch.Tensor, init_mu: torch.Tensor | None = None,
                    seed: int = 0, n_posterior: int | None = None) -> dict:
        """K posterior samples -> mean reconstruction plus per-voxel uncertainty.

        The mean is the reconstruction; the standard deviation is the uncertainty
        map. It is close to free (K samples at 256^3 fp32 is ~270 MB) and it is
        the only thing that lets a reader tell measured structure from structure
        the prior invented.
        """
        k = n_posterior if n_posterior is not None else self.cfg.n_posterior
        projs = projs.to(self.device, self.projector.dtype)
        if init_mu is None:
            from .baselines.iterative import cgls
            init_mu = cgls(projs, self.projector, n_iter=20)
        init_mu = init_mu.to(self.device, self.projector.dtype)

        samples = []
        outer = tqdm(range(k), desc="posterior", disable=not self.cfg.progress)
        for i in outer:
            samples.append(self.sample_once(projs, init_mu, seed=seed + i,
                                            progress=False))
        s = torch.stack(samples, 0)
        mean = s.mean(0)
        with torch.no_grad():
            resid = float((self.projector.fp(mean) - projs).pow(2).mean().sqrt())
        return {
            "mean": mean,
            "std": s.std(0) if k > 1 else torch.zeros_like(mean),
            "samples": s,
            "init": init_mu,
            "proj_rmse": resid,
            "n_posterior": k,
            "lipschitz": self.L,
        }
