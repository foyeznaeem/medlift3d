"""Filtered back-projection.

Baseline and (optionally) initialiser. Note the distinction this module exists
to keep straight: `Projector.bp` is the plain adjoint `A^T`, which is the right
gradient for a data-fidelity term. FBP is `A^T` composed with a ramp filter,
which is a *reconstruction* operator. Substituting one for the other gives
plausible-looking but wrong results, so they live in separate places.

The discrete ramp filter carries a normalisation that depends on ray sampling
and detector spacing. Rather than hand-tuning a constant, `estimate_fbp_scale`
recovers it once per (grid, geometry) by reconstructing a uniform disc of known
attenuation. Iterative methods in `iterative.py` need no such calibration
because they minimise `||A x - p||^2` directly.
"""
from __future__ import annotations

import numpy as np
import torch

from ..projector import Projector


def ramp_filter(projs: torch.Tensor, du: float, window: str = "hann") -> torch.Tensor:
    """Filter each detector row along u (the last axis)."""
    n = projs.shape[-1]
    pad = int(2 ** np.ceil(np.log2(max(64, 2 * n))))
    x = torch.zeros(*projs.shape[:-1], pad, dtype=projs.dtype, device=projs.device)
    x[..., :n] = projs

    freq = torch.fft.rfftfreq(pad, d=du, device=projs.device, dtype=projs.dtype)
    h = freq.abs().clone()
    if window == "hann":
        h = h * (0.5 + 0.5 * torch.cos(np.pi * freq / freq.max().clamp_min(1e-12)))
    elif window not in (None, "none", "ramlak"):
        raise ValueError(f"unknown window {window!r}")

    out = torch.fft.irfft(torch.fft.rfft(x, dim=-1) * h, n=pad, dim=-1)
    return out[..., :n].contiguous()


def estimate_fbp_scale(projector: Projector, window: str = "hann",
                       radius_frac: float = 0.35) -> float:
    """Empirical normalisation for the discrete FBP of this projector.

    Projects a uniform disc of unit attenuation, reconstructs it, and returns the
    factor that restores unit amplitude at the centre.
    """
    from ..phantom import make_water_cylinder  # local import: avoids a cycle
    from ..units import MU_WATER

    grid = projector.grid
    radius = radius_frac * min(grid.extent_mm[1], grid.extent_mm[2])
    mu = torch.from_numpy(make_water_cylinder(grid, radius_mm=radius)).to(projector.device)
    p = projector.fp(mu)
    raw = projector.bp(ramp_filter(p, projector.geometry.det_spacing[1], window))

    # Sample the interior only, well away from the disc edge.
    nz, ny, nx = grid.shape
    hz, hy, hx = nz // 2, ny // 2, nx // 2
    r = max(2, int(0.3 * radius / min(grid.spacing)))
    core = raw[hz - 1:hz + 2, hy - r:hy + r, hx - r:hx + r]
    amp = float(core.median())
    if abs(amp) < 1e-20:
        return 1.0
    return MU_WATER / amp


def fbp(projs: torch.Tensor, projector: Projector, window: str = "hann",
        scale: float | None = None, non_negative: bool = True) -> torch.Tensor:
    """Reconstruct `mu` from projections. Returns [nz, ny, nx]."""
    if scale is None:
        scale = estimate_fbp_scale(projector, window)
    filt = ramp_filter(projs.to(projector.dtype), projector.geometry.det_spacing[1], window)
    out = projector.bp(filt) * scale
    return out.clamp_min(0.0) if non_negative else out
