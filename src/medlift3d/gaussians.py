"""Anisotropic Gaussian parameterisation of attenuation, for ROI refinement.

This is an **adaptive, edge-aware parameterisation of a voxel field**: Gaussians
are evaluated onto the ROI grid and the resulting volume is line-integrated by
the ordinary projector, so the physics is identical to every other stage. It is
*not* a splat rasteriser and it does not make rendering free -- the dense ROI
volume is materialised on every iteration. The claim it can support is that
Gaussians are a compact adaptive basis; whether that helps is what the O5
ablation measures. It is affordable only on a 128^3 ROI (~8 MB), never on a
whole chest.

Two invariants:
  * amplitude uses `softplus`, so attenuation is non-negative and **unbounded**.
    A sigmoid would cap mu at 1.0 mm^-1 and saturate its own gradient.
  * regularisation acts on the **primitives** (scale, anisotropy), never on a
    rasterised dense volume -- that would reintroduce the cost the
    parameterisation exists to avoid.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import Grid, world_to_index_torch


@dataclass
class GaussianConfig:
    n_init: int = 60_000
    window: int = 7               # odd; local support in voxels
    init_scale_vox: float = 0.9   # initial sigma in voxels
    min_scale_mm: float = 0.15
    max_scale_mm: float = 6.0
    lr_xyz: float = 2e-3
    lr_amp: float = 5e-2
    lr_scale: float = 5e-3
    lr_quat: float = 1e-3
    lam_scale: float = 1e-4       # L1 on sigma: suppresses needle artefacts
    lam_aniso: float = 1e-3       # penalise sigma_max/sigma_min: fights the
                                  # missing-wedge elongation in Track B
    chunk: int = 8192

    def to_dict(self):
        return asdict(self)


_PARAM_NAMES = ("xyz", "raw_amp", "log_scale", "quat")


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of `softplus`, so an amplitude can be set to an exact value."""
    return torch.log(torch.expm1(y.clamp_min(1e-6)).clamp_min(1e-12))


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Unit-normalised quaternion (w, x, y, z) -> [N, 3, 3] rotation."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


class GaussianField(nn.Module):
    """A sum of anisotropic Gaussians, rasterisable onto a `Grid`."""

    def __init__(self, grid: Grid, cfg: GaussianConfig | None = None,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        self.grid = grid
        self.cfg = cfg or GaussianConfig()
        self.device = torch.device(device)
        self.dtype = dtype
        self.xyz = nn.Parameter(torch.zeros(0, 3, device=device, dtype=dtype))
        self.raw_amp = nn.Parameter(torch.zeros(0, device=device, dtype=dtype))
        self.log_scale = nn.Parameter(torch.zeros(0, 3, device=device, dtype=dtype))
        self.quat = nn.Parameter(torch.zeros(0, 4, device=device, dtype=dtype))
        self._register_window()

    def _register_window(self):
        w = self.cfg.window
        if w % 2 == 0:
            raise ValueError("window must be odd")
        r = w // 2
        a = torch.arange(-r, r + 1, device=self.device)
        zz, yy, xx = torch.meshgrid(a, a, a, indexing="ij")
        self.register_buffer("_off", torch.stack([zz, yy, xx], -1).reshape(-1, 3).long(),
                             persistent=False)

    # -- properties -------------------------------------------------------------

    @property
    def n(self) -> int:
        return self.xyz.shape[0]

    @property
    def amp(self) -> torch.Tensor:
        """Peak attenuation per primitive: non-negative, unbounded."""
        return F.softplus(self.raw_amp)

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(self.cfg.min_scale_mm, self.cfg.max_scale_mm)

    # -- initialisation ---------------------------------------------------------

    @torch.no_grad()
    def init_from_volume(self, mu: torch.Tensor, n: int | None = None,
                         threshold: float = 1e-4, seed: int = 0):
        """Seed primitives where there is structure to represent.

        Sampling probability is `mu + |grad mu|`, so both bulk tissue and edges
        get covered. Sampling by gradient alone leaves flat interiors unmodelled;
        sampling uniformly wastes primitives on air.
        """
        n = n or self.cfg.n_init
        mu = mu.to(self.device, self.dtype)
        g = torch.zeros_like(mu)
        for ax in range(3):
            d = torch.diff(mu, dim=ax, append=mu.narrow(ax, mu.shape[ax] - 1, 1))
            g = g + d.abs()
        w = (mu + g).clamp_min(0.0)
        w[mu < threshold] = 0.0
        flat = w.reshape(-1)
        if float(flat.sum()) <= 0:
            raise ValueError("volume has no structure above threshold to initialise from")

        gen = torch.Generator(device="cpu").manual_seed(seed)
        n = min(n, int((flat > 0).sum()))
        probs = flat.detach().cpu()
        idx = torch.multinomial(probs / probs.sum(), n, replacement=False,
                                generator=gen).to(self.device)

        nz, ny, nx = self.grid.shape
        iz = idx // (ny * nx)
        iy = (idx % (ny * nx)) // nx
        ix = idx % nx
        sz, sy, sx = self.grid.spacing
        oz, oy, ox = self.grid.origin
        xyz = torch.stack([ix * sx + ox, iy * sy + oy, iz * sz + oz], -1).to(self.dtype)

        vals = mu.reshape(-1)[idx].clamp_min(1e-6)
        s0 = self.cfg.init_scale_vox * float(min(self.grid.spacing))
        q0 = torch.zeros(n, 4, device=self.device, dtype=self.dtype)
        q0[:, 0] = 1.0

        # Invert softplus so the initial amplitude reproduces the sampled value.
        raw = _inv_softplus(vals)

        self.xyz = nn.Parameter(xyz)
        self.raw_amp = nn.Parameter(raw)
        self.log_scale = nn.Parameter(torch.full((n, 3), float(np.log(s0)),
                                                 device=self.device, dtype=self.dtype))
        self.quat = nn.Parameter(q0)
        self.calibrate_amplitude(mu)
        return self

    @torch.no_grad()
    def calibrate_amplitude(self, mu: torch.Tensor) -> float:
        """Rescale all amplitudes by the least-squares optimal global factor.

        Primitives overlap, so seeding each amplitude with the local `mu` value
        overshoots by the local packing density (measured ~6x at default
        settings). One closed-form factor `<r, mu> / <r, r>` removes that bias
        and gives the optimiser a far better starting point than letting it
        discover the scale by gradient descent.
        """
        r = self.rasterize()
        denom = float((r * r).sum())
        if denom <= 0:
            return 1.0
        alpha = float((r * mu.to(r.device, r.dtype)).sum() / denom)
        alpha = max(alpha, 1e-6)
        self.raw_amp.data = _inv_softplus(self.amp * alpha)
        return alpha

    # -- rasterisation ----------------------------------------------------------

    def rasterize(self, grid: Grid | None = None) -> torch.Tensor:
        """Evaluate the Gaussian sum onto a voxel grid. Differentiable."""
        grid = grid or self.grid
        nz, ny, nx = grid.shape
        acc = torch.zeros(nz * ny * nx, device=self.device, dtype=self.dtype)
        if self.n == 0:
            return acc.view(nz, ny, nx)

        sp = torch.tensor([grid.spacing[2], grid.spacing[1], grid.spacing[0]],
                          device=self.device, dtype=self.dtype)          # (x,y,z)
        org = torch.tensor([grid.origin[2], grid.origin[1], grid.origin[0]],
                           device=self.device, dtype=self.dtype)         # (x,y,z)
        rot = quat_to_rot(self.quat)                                     # [N,3,3]
        scale = self.scale
        amp = self.amp

        for i in range(0, self.n, self.cfg.chunk):
            j = min(i + self.cfg.chunk, self.n)
            xyz = self.xyz[i:j]
            idx_zyx = world_to_index_torch(grid, xyz)                    # [n,3] (z,y,x)
            base = idx_zyx.round().long()[:, None, :] + self._off[None]  # [n,W3,3]

            vox_xyz = base.flip(-1).to(self.dtype) * sp + org            # [n,W3,3] (x,y,z)
            delta = vox_xyz - xyz[:, None, :]
            local = torch.einsum("nab,nkb->nka", rot[i:j].transpose(-2, -1), delta)
            q = ((local / scale[i:j][:, None, :]) ** 2).sum(-1)
            val = amp[i:j][:, None] * torch.exp(-0.5 * q)

            iz, iy, ix = base[..., 0], base[..., 1], base[..., 2]
            ok = ((iz >= 0) & (iz < nz) & (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx))
            lin = (iz.clamp(0, nz - 1) * ny + iy.clamp(0, ny - 1)) * nx + ix.clamp(0, nx - 1)
            acc = acc.index_add(0, lin.reshape(-1), (val * ok).reshape(-1))
        return acc.view(nz, ny, nx)

    # -- regularisation ---------------------------------------------------------

    def regularisation(self) -> torch.Tensor:
        """Primitive-space penalties. Never regularise the rasterised volume."""
        if self.n == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        s = self.scale
        l_scale = s.mean()
        aniso = (s.amax(-1) / s.amin(-1).clamp_min(1e-8) - 1.0).clamp_min(0.0).mean()
        return self.cfg.lam_scale * l_scale + self.cfg.lam_aniso * aniso

    # -- optimiser --------------------------------------------------------------

    def make_optimizer(self) -> torch.optim.Optimizer:
        c = self.cfg
        return torch.optim.Adam([
            {"params": [self.xyz], "lr": c.lr_xyz},
            {"params": [self.raw_amp], "lr": c.lr_amp},
            {"params": [self.log_scale], "lr": c.lr_scale},
            {"params": [self.quat], "lr": c.lr_quat},
        ], eps=1e-15)

    @torch.no_grad()
    def prune(self, min_amp: float = 1e-4, optimizer=None) -> int:
        """Drop primitives that contribute nothing. Returns the number removed.

        Pruning replaces every `Parameter`, which orphans an optimizer that is
        already holding the old tensors: it goes on stepping detached buffers
        while the live parameters never move again. Pass the optimizer and its
        param groups and per-parameter state are re-bound and sliced to match.
        """
        if self.n == 0:
            return 0
        keep = self.amp > min_amp
        removed = int((~keep).sum())
        if not removed:
            return 0
        old = {name: getattr(self, name) for name in _PARAM_NAMES}
        for name, p in old.items():
            setattr(self, name, nn.Parameter(p.data[keep]))
        if optimizer is not None:
            self._rebind(optimizer, old, keep)
        return removed

    def _rebind(self, optimizer, old, keep) -> None:
        """Point `optimizer` at the new Parameters, carrying its state across."""
        new_for = {id(p): getattr(self, name) for name, p in old.items()}
        for group in optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                new = new_for.get(id(p))
                if new is None:
                    continue
                state = optimizer.state.pop(p, None)
                if state is not None:
                    # Slice the per-primitive moments; leave scalars (step) alone.
                    optimizer.state[new] = {
                        k: v[keep] if torch.is_tensor(v) and v.shape == p.shape else v
                        for k, v in state.items()}
                group["params"][i] = new
