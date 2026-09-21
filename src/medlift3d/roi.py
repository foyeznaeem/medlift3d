"""Two-scale region-of-interest refinement.

A 1 mm whole-chest grid is unaffordable on a 16 GB card, and it is also
unnecessary: nodules are at most ~30 mm across and the reading workflow already
localises them. So reconstruct the chest coarsely, then refine a small fine-grid
box around the nodule.

The decomposition is **exact**, because `A` is linear:

    mu           = mu_bg            +  mu_roi
    A mu         = A_chest(mu_bg)   +  A_roi(mu_roi)
    residual r   = p - A_chest(mu_bg)

where `mu_bg` is the coarse reconstruction with the ROI box zeroed and `mu_roi`
lives on the fine grid. Fitting `A_roi(mu_roi) ~ r` therefore introduces no
approximation beyond resampling the background at the box boundary.
`tests/test_roi_exact.py` asserts the exact same-grid case and measures the
fine-grid resampling error rather than assuming it away.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .gaussians import GaussianConfig, GaussianField
from .geometry import Grid, roi_grid_at
from .projector import Projector
from .units import MU_MAX


def resample(mu: torch.Tensor, src: Grid, dst: Grid,
             mode: str = "bilinear") -> torch.Tensor:
    """Trilinear resample a volume from one grid onto another (world-aligned)."""
    pts = dst.voxel_centres_xyz(device=mu.device, dtype=mu.dtype)   # [nz,ny,nx,3]
    g = src.world_to_norm(pts.reshape(-1, 3)).view(1, *dst.shape, 3)
    out = F.grid_sample(mu[None, None], g, mode=mode,
                        padding_mode="zeros", align_corners=True)
    return out.view(dst.shape)


def roi_index_box(chest: Grid, roi: Grid) -> tuple[slice, slice, slice]:
    """Index range of the ROI's world extent within the chest grid (clipped)."""
    lo_w = np.array(roi.origin, dtype=np.float64)                       # (z,y,x)
    hi_w = lo_w + np.array(roi.extent_mm, dtype=np.float64)
    out = []
    for ax in range(3):
        lo = (lo_w[ax] - chest.origin[ax]) / chest.spacing[ax]
        hi = (hi_w[ax] - chest.origin[ax]) / chest.spacing[ax]
        a = max(0, int(np.floor(lo)))
        b = min(chest.shape[ax], int(np.ceil(hi)) + 1)
        out.append(slice(a, max(a + 1, b)))
    return tuple(out)


def zero_roi(mu: torch.Tensor, chest: Grid, roi: Grid) -> torch.Tensor:
    """Copy of `mu` with the ROI's world extent zeroed on the chest grid."""
    out = mu.clone()
    out[roi_index_box(chest, roi)] = 0.0
    return out


def background_residual(projs: torch.Tensor, mu_bg: torch.Tensor,
                        chest_projector: Projector) -> torch.Tensor:
    """`r = p - A_chest(mu_bg)`: what the ROI still has to explain."""
    with torch.no_grad():
        return projs - chest_projector.fp(mu_bg)


def refine_roi(projs: torch.Tensor, mu_chest: torch.Tensor,
               chest_projector: Projector, centre_xyz,
               roi_shape=(128, 128, 128), roi_spacing=(1.0, 1.0, 1.0),
               n_iter: int = 800, gcfg: GaussianConfig | None = None,
               mu_max: float = MU_MAX, progress: bool = True,
               use_gaussians: bool = True, lr_voxel: float = 5e-2,
               seed: int = 0) -> dict:
    """Refine a fine-grid ROI against the background-subtracted residual.

    `use_gaussians=False` runs the identical optimisation on a plain voxel grid.
    It is not a fallback but the control arm for the O5 ablation: same residual,
    same iteration count, different parameterisation.
    """
    device = chest_projector.device
    chest = chest_projector.grid
    roi = roi_grid_at(centre_xyz, roi_shape, roi_spacing)

    mu_bg = zero_roi(mu_chest.to(device), chest, roi)
    resid = background_residual(projs.to(device), mu_bg, chest_projector)

    roi_proj = Projector(roi, chest_projector.geometry,
                         chunk_rays=chest_projector.chunk_rays,
                         device=device, dtype=chest_projector.dtype,
                         bp_impl=chest_projector.bp_impl)

    init = resample(mu_chest.to(device), chest, roi).clamp(0.0, mu_max)

    if use_gaussians:
        field = GaussianField(roi, gcfg or GaussianConfig(), device=device,
                              dtype=chest_projector.dtype)
        field.init_from_volume(init, seed=seed)
        opt = field.make_optimizer()
        params = None
    else:
        field = None
        vox = init.clone().requires_grad_(True)
        opt = torch.optim.Adam([vox], lr=lr_voxel * mu_max)
        params = vox

    # Scale the data term so the loss is O(1) regardless of geometry.
    norm = max(float(resid.pow(2).mean()), 1e-20)
    history = []
    bar = tqdm(range(n_iter), desc="ROI refine", disable=not progress, leave=False)
    for it in bar:
        opt.zero_grad(set_to_none=True)
        mu_roi = field.rasterize() if use_gaussians else params.clamp(0.0, mu_max)
        data = (roi_proj.fp(mu_roi) - resid).pow(2).mean() / norm
        loss = data + (field.regularisation() if use_gaussians else 0.0)
        loss.backward()
        opt.step()
        if use_gaussians and (it + 1) % 200 == 0:
            field.prune(optimizer=opt)
        if (it + 1) % max(1, n_iter // 10) == 0:
            history.append({"iter": it + 1, "data": float(data.detach()),
                             "loss": float(loss.detach())})
            bar.set_postfix(data=f"{float(data.detach()):.4g}")

    with torch.no_grad():
        mu_roi = (field.rasterize() if use_gaussians else params).clamp(0.0, mu_max)
    return {
        "mu_roi": mu_roi.detach(),
        "roi_grid": roi,
        "init": init,
        "residual_rmse": float((roi_proj.fp(mu_roi.detach()) - resid).pow(2).mean().sqrt()),
        "history": history,
        "n_primitives": (field.n if use_gaussians else int(np.prod(roi.shape))),
        "parameterisation": "gaussian" if use_gaussians else "voxel",
    }
