"""Voxel grids that know where they are in space.

The deepest failure mode in volumetric reconstruction is an array that has lost
its physical frame: a reconstruction and its ground truth end up on different
grids and every metric computed between them is meaningless. `Grid` makes the
frame part of the type, and `assert_matches` makes a mismatch a loud error.

Conventions, without exception:
  * index order is (z, y, x) -- z is cranio-caudal, y anterior-posterior, x left-right
  * `origin` is the world position of the CENTRE of voxel [0, 0, 0], in mm, RAS
  * `spacing` is mm per voxel, same axis order as `shape`
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]        # (nz, ny, nx)
    spacing: tuple[float, float, float]  # (sz, sy, sx) mm
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (oz, oy, ox) mm

    def __post_init__(self):
        if len(self.shape) != 3 or len(self.spacing) != 3 or len(self.origin) != 3:
            raise ValueError("shape, spacing and origin must all be 3-tuples")
        if any(n < 2 for n in self.shape):
            raise ValueError(f"every axis needs >= 2 voxels, got {self.shape}")
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    # ---- physical description -------------------------------------------------

    @property
    def affine(self) -> np.ndarray:
        """4x4 index->world matrix in (x, y, z) world order, NIfTI convention."""
        sz, sy, sx = self.spacing
        oz, oy, ox = self.origin
        a = np.eye(4, dtype=np.float64)
        a[:3, :3] = np.diag([sx, sy, sz])   # columns map (i_x, i_y, i_z)
        a[:3, 3] = [ox, oy, oz]
        return a

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        """Physical span from first to last voxel centre, in (z, y, x) order."""
        return tuple(s * (n - 1) for s, n in zip(self.spacing, self.shape))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    @property
    def centre_mm(self) -> tuple[float, float, float]:
        return tuple(o + e / 2.0 for o, e in zip(self.origin, self.extent_mm))

    def bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as (lo, hi) in world (x, y, z) order."""
        oz, oy, ox = self.origin
        ez, ey, ex = self.extent_mm
        lo = np.array([ox, oy, oz], dtype=np.float64)
        return lo, lo + np.array([ex, ey, ez], dtype=np.float64)

    def diagonal_mm(self) -> float:
        return float(np.linalg.norm(self.extent_mm))

    # ---- coordinate transforms ------------------------------------------------

    def world_to_norm(self, pts_xyz: torch.Tensor) -> torch.Tensor:
        """World (x, y, z) mm -> grid_sample coordinates in [-1, 1].

        Returns the last axis ordered (x, y, z), which is what `F.grid_sample`
        expects for a 5-D input laid out [N, C, z, y, x] (align_corners=True).
        """
        oz, oy, ox = self.origin
        sz, sy, sx = self.spacing
        nz, ny, nx = self.shape
        lo = torch.tensor([ox, oy, oz], dtype=pts_xyz.dtype, device=pts_xyz.device)
        span = torch.tensor(
            [sx * (nx - 1), sy * (ny - 1), sz * (nz - 1)],
            dtype=pts_xyz.dtype, device=pts_xyz.device,
        )
        return 2.0 * (pts_xyz - lo) / span - 1.0

    def voxel_centres_xyz(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """[nz, ny, nx, 3] world coordinates in (x, y, z) order."""
        nz, ny, nx = self.shape
        sz, sy, sx = self.spacing
        oz, oy, ox = self.origin
        z = torch.arange(nz, device=device, dtype=dtype) * sz + oz
        y = torch.arange(ny, device=device, dtype=dtype) * sy + oy
        x = torch.arange(nx, device=device, dtype=dtype) * sx + ox
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1)

    def world_to_index(self, pts_xyz: np.ndarray) -> np.ndarray:
        """World (x, y, z) mm -> continuous index (z, y, x)."""
        pts = np.atleast_2d(np.asarray(pts_xyz, dtype=np.float64))
        oz, oy, ox = self.origin
        sz, sy, sx = self.spacing
        idx = np.stack([
            (pts[:, 2] - oz) / sz,
            (pts[:, 1] - oy) / sy,
            (pts[:, 0] - ox) / sx,
        ], axis=-1)
        return idx

    # ---- discipline -----------------------------------------------------------

    def assert_matches(self, other: "Grid", tol: float = 1e-4, what: str = "") -> None:
        prefix = f"{what}: " if what else ""
        if tuple(self.shape) != tuple(other.shape):
            raise AssertionError(f"{prefix}shape {self.shape} != {other.shape}")
        if not np.allclose(self.affine, other.affine, atol=tol):
            raise AssertionError(f"{prefix}affine mismatch\n{self.affine}\nvs\n{other.affine}")

    def as_dict(self) -> dict:
        return {"shape": list(self.shape), "spacing": list(self.spacing), "origin": list(self.origin)}

    @staticmethod
    def from_dict(d: dict) -> "Grid":
        return Grid(tuple(int(v) for v in d["shape"]),
                    tuple(float(v) for v in d["spacing"]),
                    tuple(float(v) for v in d.get("origin", (0.0, 0.0, 0.0))))

    @staticmethod
    def spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
        """Robust spacing extraction, valid for oblique affines.

        `np.diag(affine)` is only correct for axis-aligned volumes -- use the
        column norms instead. Returned in (z, y, x) order.
        """
        cols = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)  # (x, y, z)
        return (float(cols[2]), float(cols[1]), float(cols[0]))

    @staticmethod
    def centred(shape, spacing, origin=None) -> "Grid":
        """Grid centred on the world origin unless `origin` is given."""
        shape = tuple(int(v) for v in shape)
        spacing = tuple(float(v) for v in spacing)
        if origin is None:
            origin = tuple(-s * (n - 1) / 2.0 for s, n in zip(spacing, shape))
        return Grid(shape, spacing, tuple(float(v) for v in origin))


# The two grids this project uses. Anything else is a bug.
CHEST = Grid.centred((256, 256, 256), (1.5, 1.5, 1.5))
ROI = Grid.centred((128, 128, 128), (1.0, 1.0, 1.0))


def roi_grid_at(centre_xyz, shape=(128, 128, 128), spacing=(1.0, 1.0, 1.0)) -> Grid:
    """A fine ROI grid centred on a world point -- e.g. a nodule centroid."""
    cx, cy, cz = (float(v) for v in centre_xyz)
    extent = [s * (n - 1) / 2.0 for s, n in zip(spacing, shape)]
    origin = (cz - extent[0], cy - extent[1], cx - extent[2])
    return Grid(tuple(int(v) for v in shape), tuple(float(v) for v in spacing), origin)


def world_to_index_torch(grid: Grid, pts_xyz: torch.Tensor) -> torch.Tensor:
    """World (x, y, z) mm -> continuous index (z, y, x), as a tensor.

    Matches `F.grid_sample(..., align_corners=True)` exactly: normalised
    coordinate `n` corresponds to index `(n + 1) / 2 * (size - 1)`, which is
    `(world - origin) / spacing`.
    """
    oz, oy, ox = grid.origin
    sz, sy, sx = grid.spacing
    lo = torch.tensor([ox, oy, oz], dtype=pts_xyz.dtype, device=pts_xyz.device)
    sp = torch.tensor([sx, sy, sz], dtype=pts_xyz.dtype, device=pts_xyz.device)
    idx_xyz = (pts_xyz - lo) / sp
    return idx_xyz.flip(-1)  # (x, y, z) -> (z, y, x)
