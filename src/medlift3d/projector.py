"""Differentiable X-ray forward projection.

The forward model is the Beer-Lambert line integral of the linear attenuation
coefficient, `p = integral(mu dl)`, discretised by trilinear ray sampling
(Joseph's method). It is built entirely from `torch` ops, so autograd supplies
the backward pass -- there is no opaque library call in the middle of the graph
to sever it.

`A` is linear, so its vector-Jacobian product *is* its adjoint. `Projector.bp`
therefore computes `A^T` by a VJP rather than by a second hand-written kernel,
which makes an fp/bp inconsistency structurally impossible.

`bp` is the plain adjoint back-projection, NOT filtered back-projection. FBP
applies a ramp filter and is a *reconstruction* operator; using it as a gradient
gives wrong updates that still look like they are converging. Filtered
back-projection lives in `medlift3d.baselines.fbp`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import Grid, world_to_index_torch

_EPS = 1e-8


# ----------------------------------------------------------------------------
# Acquisition geometries
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ParallelGeometry:
    """Sparse-view parallel-beam CT: `n_views` equispaced over `arc_deg`.

    Rotation is about the z (cranio-caudal) axis. The detector's u axis lies in
    the transverse plane and its v axis is along z.
    """
    n_views: int
    arc_deg: float
    det_shape: tuple[int, int]              # (nv, nu)
    det_spacing: tuple[float, float]        # (dv, du) mm
    start_deg: float = 0.0
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)   # world (x, y, z)

    kind = "parallel"

    @property
    def proj_shape(self) -> tuple[int, int, int]:
        return (self.n_views, *self.det_shape)

    @property
    def angles_rad(self) -> np.ndarray:
        # Endpoint excluded: over a 180 deg arc the last view would duplicate the
        # first for parallel beam.
        return np.deg2rad(self.start_deg + np.linspace(0.0, self.arc_deg, self.n_views, endpoint=False))

    def rays(self, grid: Grid, device=None, dtype=torch.float32):
        nv, nu = self.det_shape
        dv, du = self.det_spacing
        reach = grid.diagonal_mm()  # push origins safely outside the volume

        ang = torch.tensor(self.angles_rad, device=device, dtype=dtype)          # [V]
        d = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)   # [V,3]
        u = torch.stack([-torch.sin(ang), torch.cos(ang), torch.zeros_like(ang)], -1)  # [V,3]
        v = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(self.n_views, 3)

        iu = (torch.arange(nu, device=device, dtype=dtype) - (nu - 1) / 2.0) * du
        iv = (torch.arange(nv, device=device, dtype=dtype) - (nv - 1) / 2.0) * dv
        c = torch.tensor(self.centre, device=device, dtype=dtype)

        # [V, nv, nu, 3]
        pts = (c
               + u[:, None, None, :] * iu[None, None, :, None]
               + v[:, None, None, :] * iv[None, :, None, None])
        dirs = d[:, None, None, :].expand(self.n_views, nv, nu, 3).contiguous()
        origins = pts - dirs * reach
        return origins, dirs

    @staticmethod
    def covering(grid: Grid, n_views: int, arc_deg: float = 180.0,
                 det_spacing: float | tuple[float, float] | None = None,
                 start_deg: float = 0.0, margin: float = 1.06) -> "ParallelGeometry":
        """Build a geometry whose detector is guaranteed to cover `grid`.

        A detector that under-covers the object truncates the line integrals and
        no reconstruction method can recover from it, so size it from the grid
        rather than hardcoding numbers.
        """
        ez, ey, ex = grid.extent_mm
        if det_spacing is None:
            dv = du = float(min(grid.spacing))
        elif isinstance(det_spacing, (int, float)):
            dv = du = float(det_spacing)
        else:
            dv, du = (float(v) for v in det_spacing)
        # A rotating parallel beam must cover the in-plane diagonal.
        in_plane = math.hypot(ex, ey) * margin
        nu = int(math.ceil(in_plane / du))
        nv = int(math.ceil(ez * margin / dv))
        return ParallelGeometry(n_views, arc_deg, (nv, nu), (dv, du), start_deg,
                                centre=(grid.centre_mm[2], grid.centre_mm[1], grid.centre_mm[0]))


@dataclass(frozen=True)
class DTSGeometry:
    """Limited-angle digital tomosynthesis.

    The source translates along x while a stationary detector sits behind the
    patient. `arc_deg` is the total angle subtended *at the object*, which is how
    tomosynthesis angular range is conventionally quoted: the source offset is
    `src_dist * tan(arc/2)`.

    The missing wedge is `180 - arc_deg` degrees wide, which is why depth
    resolution here is poor and why `medlift3d` reports Track B separately.
    """
    n_views: int
    arc_deg: float
    det_shape: tuple[int, int]              # (nv, nu)
    det_spacing: tuple[float, float]        # (dv, du) mm
    src_dist: float = 1000.0                # source -> object centre, mm
    det_dist: float = 400.0                 # object centre -> detector, mm
    centre: tuple[float, float, float] = (0.0, 0.0, 0.0)

    kind = "dts"

    @property
    def proj_shape(self) -> tuple[int, int, int]:
        return (self.n_views, *self.det_shape)

    @property
    def source_offsets(self) -> np.ndarray:
        half = self.src_dist * math.tan(math.radians(self.arc_deg) / 2.0)
        if self.n_views == 1:
            return np.zeros(1)
        return np.linspace(-half, half, self.n_views)

    @property
    def missing_wedge_deg(self) -> float:
        return 180.0 - self.arc_deg

    def rays(self, grid: Grid, device=None, dtype=torch.float32):
        nv, nu = self.det_shape
        dv, du = self.det_spacing
        c = torch.tensor(self.centre, device=device, dtype=dtype)

        off = torch.tensor(self.source_offsets, device=device, dtype=dtype)       # [V]
        src = c + torch.stack(
            [off, torch.full_like(off, self.src_dist), torch.zeros_like(off)], -1)  # [V,3]

        iu = (torch.arange(nu, device=device, dtype=dtype) - (nu - 1) / 2.0) * du
        iv = (torch.arange(nv, device=device, dtype=dtype) - (nv - 1) / 2.0) * dv
        zeros = torch.zeros(nv, nu, device=device, dtype=dtype)
        det = torch.stack([
            iu[None, :].expand(nv, nu),
            zeros - self.det_dist,
            iv[:, None].expand(nv, nu),
        ], -1) + c                                                                # [nv,nu,3]

        origins = src[:, None, None, :].expand(self.n_views, nv, nu, 3).contiguous()
        dirs = det[None] - origins
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        return origins, dirs

    @staticmethod
    def covering(grid: Grid, n_views: int, arc_deg: float = 30.0,
                 det_spacing: float | tuple[float, float] | None = None,
                 src_dist: float = 1000.0, det_dist: float = 400.0,
                 margin: float = 1.15) -> "DTSGeometry":
        ez, ey, ex = grid.extent_mm
        if det_spacing is None:
            dv = du = float(min(grid.spacing))
        elif isinstance(det_spacing, (int, float)):
            dv = du = float(det_spacing)
        else:
            dv, du = (float(v) for v in det_spacing)
        # Cone magnification at the detector plane, plus the source sweep.
        mag = (src_dist + det_dist) / src_dist
        half = src_dist * math.tan(math.radians(arc_deg) / 2.0)
        nu = int(math.ceil((ex * mag + 2 * half * det_dist / src_dist) * margin / du))
        nv = int(math.ceil(ez * mag * margin / dv))
        return DTSGeometry(n_views, arc_deg, (nv, nu), (dv, du), src_dist, det_dist,
                           centre=(grid.centre_mm[2], grid.centre_mm[1], grid.centre_mm[0]))


Geometry = ParallelGeometry | DTSGeometry


# ----------------------------------------------------------------------------
# Projector
# ----------------------------------------------------------------------------

def _ray_box_t(origins, dirs, lo, hi):
    """Slab-method ray/AABB intersection.

    Returns (t_near, t_far), both zero for rays that miss the box -- so their
    step length `dt` is zero and they contribute nothing.
    """
    inv = 1.0 / torch.where(dirs.abs() < _EPS, torch.full_like(dirs, _EPS), dirs)
    t1 = (lo - origins) * inv
    t2 = (hi - origins) * inv
    t_near = torch.minimum(t1, t2).amax(dim=-1)
    t_far = torch.maximum(t1, t2).amin(dim=-1)
    valid = t_far > t_near
    t_near = torch.where(valid, t_near, torch.zeros_like(t_near))
    t_far = torch.where(valid, t_far, torch.zeros_like(t_far))
    return t_near, t_far


class Projector:
    """`A` and `A^T` for one (grid, geometry) pair.

    Rays are precomputed once; sampling is chunked so peak memory stays bounded
    regardless of detector size.
    """

    def __init__(self, grid: Grid, geometry: Geometry, n_samples: int | None = None,
                 chunk_rays: int = 8192, device="cpu", dtype=torch.float32,
                 bp_impl: str = "vjp"):
        self.grid = grid
        self.geometry = geometry
        self.device = torch.device(device)
        self.dtype = dtype
        self.proj_shape = geometry.proj_shape
        self.n_rays = int(np.prod(self.proj_shape))
        self.chunk_rays = int(chunk_rays)
        if bp_impl not in ("vjp", "scatter"):
            raise ValueError(f"bp_impl must be 'vjp' or 'scatter', got {bp_impl!r}")
        self.bp_impl = bp_impl

        # Sample at roughly one voxel along the ray: the standard Joseph rate.
        if n_samples is None:
            n_samples = int(math.ceil(grid.diagonal_mm() / min(grid.spacing)))
        self.n_samples = int(n_samples)

        origins, dirs = geometry.rays(grid, device=self.device, dtype=dtype)
        self.origins = origins.reshape(-1, 3)
        self.dirs = dirs.reshape(-1, 3)

        lo_np, hi_np = grid.bounds_xyz()
        lo = torch.tensor(lo_np, device=self.device, dtype=dtype)
        hi = torch.tensor(hi_np, device=self.device, dtype=dtype)
        t_near, t_far = _ray_box_t(self.origins, self.dirs, lo, hi)
        self.t_near, self.t_far = t_near, t_far
        self.dt = (t_far - t_near) / self.n_samples          # 0 for rays that miss

        self._frac = ((torch.arange(self.n_samples, device=self.device, dtype=dtype)
                       + 0.5) / self.n_samples)

    # -- helpers ---------------------------------------------------------------

    def coverage(self) -> float:
        """Fraction of rays that intersect the volume. Near 1.0 means the
        detector is over-sized; very low means it is mis-aimed."""
        return float((self.dt > 0).float().mean())

    def _ray_points(self, sl: slice):
        """World-space sample points [R, S, 3] and per-ray step for a ray slice."""
        o = self.origins[sl]
        d = self.dirs[sl]
        tn = self.t_near[sl]
        span = self.t_far[sl] - tn
        t = tn[:, None] + self._frac[None, :] * span[:, None]   # [R,S]
        pts = o[:, None, :] + d[:, None, :] * t[..., None]      # [R,S,3] world xyz
        return pts, self.dt[sl]

    def _sample_chunk(self, vol5: torch.Tensor, sl: slice) -> torch.Tensor:
        """Line integral for a slice of rays. `vol5` is [1, 1, nz, ny, nx]."""
        pts, dt = self._ray_points(sl)
        g = self.grid.world_to_norm(pts).view(1, pts.shape[0], self.n_samples, 1, 3)
        vals = F.grid_sample(vol5, g, mode="bilinear",
                             padding_mode="zeros", align_corners=True)
        return vals.view(pts.shape[0], self.n_samples).sum(1) * dt

    # -- the operator ----------------------------------------------------------

    def fp(self, mu: torch.Tensor) -> torch.Tensor:
        """Forward project. `mu` is [nz, ny, nx]; returns [V, nv, nu].

        Differentiable in `mu` -- this is the whole point of the module.
        """
        if tuple(mu.shape) != tuple(self.grid.shape):
            raise ValueError(f"mu shape {tuple(mu.shape)} != grid {self.grid.shape}")
        vol5 = mu.to(self.dtype)[None, None]
        out = [self._sample_chunk(vol5, slice(i, min(i + self.chunk_rays, self.n_rays)))
               for i in range(0, self.n_rays, self.chunk_rays)]
        return torch.cat(out, 0).view(self.proj_shape)

    def bp_scatter(self, projs: torch.Tensor) -> torch.Tensor:
        """Adjoint by explicit trilinear scatter.

        The weights reproduce `grid_sample(align_corners=True,
        padding_mode="zeros")` exactly -- out-of-range corners get zero weight
        rather than being clamped -- and `tests/test_adjoint.py` asserts this
        agrees with `bp`.

        Kept as an alternative rather than the default: measured on CPU it is
        several times slower than routing through `grid_sample`'s fused backward
        kernel. Benchmark both on your hardware via `Projector.benchmark_bp`
        before choosing, then pass `bp_impl="scatter"` to select it.
        """
        if tuple(projs.shape) != tuple(self.proj_shape):
            raise ValueError(f"projs shape {tuple(projs.shape)} != {self.proj_shape}")
        nz, ny, nx = self.grid.shape
        flat_proj = projs.reshape(-1).to(self.dtype)
        acc = torch.zeros(nz * ny * nx, device=self.device, dtype=self.dtype)

        for i in range(0, self.n_rays, self.chunk_rays):
            sl = slice(i, min(i + self.chunk_rays, self.n_rays))
            pts, dt = self._ray_points(sl)
            idx = world_to_index_torch(self.grid, pts)              # [R,S,3] (z,y,x)
            base = torch.floor(idx)
            frac = idx - base
            base = base.long()
            # Per-sample contribution, shared by all 8 corners.
            val = (flat_proj[sl] * dt)[:, None].expand(-1, self.n_samples)

            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        iz = base[..., 0] + dz
                        iy = base[..., 1] + dy
                        ix = base[..., 2] + dx
                        wz = frac[..., 0] if dz else 1.0 - frac[..., 0]
                        wy = frac[..., 1] if dy else 1.0 - frac[..., 1]
                        wx = frac[..., 2] if dx else 1.0 - frac[..., 2]
                        ok = ((iz >= 0) & (iz < nz) & (iy >= 0) & (iy < ny)
                              & (ix >= 0) & (ix < nx))
                        w = wz * wy * wx * val * ok
                        lin = (iz.clamp(0, nz - 1) * ny
                               + iy.clamp(0, ny - 1)) * nx + ix.clamp(0, nx - 1)
                        acc.scatter_add_(0, lin.reshape(-1), w.reshape(-1))
        return acc.view(nz, ny, nx)

    def bp_vjp(self, projs: torch.Tensor) -> torch.Tensor:
        """Adjoint as the vector-Jacobian product of `fp`.

        `A` is linear, so its VJP *is* `A^T`: an fp/bp inconsistency is
        structurally impossible, which is why this is the default. Gradients
        accumulate into one `probe.grad` buffer across chunks, so peak memory is
        a single volume regardless of ray count.
        """
        if tuple(projs.shape) != tuple(self.proj_shape):
            raise ValueError(f"projs shape {tuple(projs.shape)} != {self.proj_shape}")
        flat = projs.reshape(-1).to(self.dtype)
        probe = torch.zeros(self.grid.shape, device=self.device,
                            dtype=self.dtype, requires_grad=True)
        with torch.enable_grad():
            for i in range(0, self.n_rays, self.chunk_rays):
                sl = slice(i, min(i + self.chunk_rays, self.n_rays))
                self._sample_chunk(probe[None, None], sl).backward(gradient=flat[sl])
        return probe.grad.detach()

    def bp(self, projs: torch.Tensor) -> torch.Tensor:
        """Adjoint back-projection `A^T p`. Returns [nz, ny, nx].

        NOT filtered back-projection: there is no ramp filter here. See
        `medlift3d.baselines.fbp` for that.
        """
        if self.bp_impl == "scatter":
            return self.bp_scatter(projs)
        return self.bp_vjp(projs)

    def benchmark_bp(self, repeats: int = 1) -> dict:
        """Time both adjoint implementations on this projector.

        Which one wins depends on grid size, ray count and device, so measure
        rather than assume.
        """
        y = torch.rand(self.proj_shape, device=self.device, dtype=self.dtype)
        out = {}
        for name, fn in (("vjp", self.bp_vjp), ("scatter", self.bp_scatter)):
            fn(y)  # warm up
            t0 = time.perf_counter()
            for _ in range(repeats):
                fn(y)
            out[name] = (time.perf_counter() - t0) / repeats
        out["faster"] = min(out, key=out.get)
        return out

    def residual(self, mu: torch.Tensor, projs: torch.Tensor) -> torch.Tensor:
        return self.fp(mu) - projs

    def grad_data_term(self, mu: torch.Tensor, projs: torch.Tensor) -> torch.Tensor:
        """grad of 0.5 * ||A mu - p||^2  =  A^T (A mu - p). No autograd needed."""
        with torch.no_grad():
            return self.bp(self.fp(mu) - projs)

    def __repr__(self):
        return (f"Projector(grid={self.grid.shape}, geom={self.geometry.kind}, "
                f"views={self.proj_shape[0]}, det={self.proj_shape[1:]}, "
                f"samples={self.n_samples})")
