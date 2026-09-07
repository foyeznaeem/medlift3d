"""MedLift-3D: volumetric lung nodule assessment from sparse-view and
limited-angle chest projections.

Clean-room implementation. See `docs/IMPLEMENTATION.md` for the build order and
`docs/CODE_AUDIT.md` for the failure modes this design exists to avoid.
"""
__version__ = "0.1.0"

from .geometry import CHEST, ROI, Grid, roi_grid_at
from .projector import DTSGeometry, ParallelGeometry, Projector
from .units import HU_CLIP, MU_MAX, MU_WATER, hu_to_mu, mu_to_hu, mu_to_net, net_to_mu

__all__ = [
    "__version__",
    "Grid", "CHEST", "ROI", "roi_grid_at",
    "Projector", "ParallelGeometry", "DTSGeometry",
    "MU_WATER", "MU_MAX", "HU_CLIP", "hu_to_mu", "mu_to_hu", "mu_to_net", "net_to_mu",
]
