"""Iterative reconstruction baselines.

These minimise `||A x - p||^2` (plus a regulariser) using only `A` and `A^T`, so
unlike FBP they carry no discretisation-dependent normalisation constant.

`sirt_tv` is the classical strong baseline for sparse-view CT and is what a
learned prior has to beat to justify its existence.
"""
from __future__ import annotations

import torch
from tqdm.auto import tqdm

from ..projector import Projector
from ..units import MU_MAX
from ..utils import tv_grad


def _sirt_weights(projector: Projector):
    """Row and column sums of |A|, used to precondition SIRT."""
    with torch.no_grad():
        ones_vol = torch.ones(projector.grid.shape, device=projector.device,
                              dtype=projector.dtype)
        row = projector.fp(ones_vol).clamp_min(1e-6)          # A 1
        ones_proj = torch.ones(projector.proj_shape, device=projector.device,
                               dtype=projector.dtype)
        col = projector.bp(ones_proj).clamp_min(1e-6)          # A^T 1
    return row, col


def sirt(projs: torch.Tensor, projector: Projector, n_iter: int = 50,
         x0: torch.Tensor | None = None, relax: float = 1.0,
         mu_max: float = MU_MAX, progress: bool = False) -> torch.Tensor:
    """Simultaneous Iterative Reconstruction Technique with a box constraint."""
    row, col = _sirt_weights(projector)
    x = torch.zeros(projector.grid.shape, device=projector.device,
                    dtype=projector.dtype) if x0 is None else x0.clone()
    it = tqdm(range(n_iter), desc="SIRT", disable=not progress)
    with torch.no_grad():
        for _ in it:
            x = x + relax * projector.bp((projs - projector.fp(x)) / row) / col
            x.clamp_(0.0, mu_max)
    return x


def sirt_tv(projs: torch.Tensor, projector: Projector, n_iter: int = 60,
            tv_weight: float = 2e-3, tv_steps: int = 2, x0: torch.Tensor | None = None,
            relax: float = 1.0, mu_max: float = MU_MAX,
            progress: bool = False) -> torch.Tensor:
    """SIRT with interleaved total-variation descent.

    `tv_weight` is relative: the TV gradient is rescaled to `tv_weight` times the
    magnitude of the data update, so the same value behaves consistently across
    grid sizes and view counts instead of needing to be retuned every time.
    """
    row, col = _sirt_weights(projector)
    x = torch.zeros(projector.grid.shape, device=projector.device,
                    dtype=projector.dtype) if x0 is None else x0.clone()
    it = tqdm(range(n_iter), desc="SIRT-TV", disable=not progress)
    with torch.no_grad():
        for _ in it:
            upd = projector.bp((projs - projector.fp(x)) / row) / col
            x = x + relax * upd
            if tv_weight > 0:
                ref = upd.abs().mean().clamp_min(1e-12)
                for _ in range(tv_steps):
                    g = tv_grad(x)
                    step = tv_weight * ref / g.abs().mean().clamp_min(1e-12)
                    x = x - step * g
            x.clamp_(0.0, mu_max)
    return x


def cgls(projs: torch.Tensor, projector: Projector, n_iter: int = 30,
         x0: torch.Tensor | None = None, mu_max: float = MU_MAX,
         progress: bool = False) -> torch.Tensor:
    """Conjugate gradients on the normal equations.

    Fast and calibration-free, which makes it the default initialiser for the
    diffusion solver. Semi-convergent: it starts fitting noise if run too long,
    so keep `n_iter` modest.
    """
    with torch.no_grad():
        x = torch.zeros(projector.grid.shape, device=projector.device,
                        dtype=projector.dtype) if x0 is None else x0.clone()
        r = projs - projector.fp(x)
        s = projector.bp(r)
        p = s.clone()
        gamma = float((s * s).sum())
        it = tqdm(range(n_iter), desc="CGLS", disable=not progress)
        for _ in it:
            q = projector.fp(p)
            denom = float((q * q).sum())
            if denom < 1e-20 or gamma < 1e-20:
                break
            alpha = gamma / denom
            x = x + alpha * p
            r = r - alpha * q
            s = projector.bp(r)
            gamma_new = float((s * s).sum())
            p = s + (gamma_new / gamma) * p
            gamma = gamma_new
        return x.clamp_(0.0, mu_max)
