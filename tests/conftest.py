import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch.set_num_threads(2)

from medlift3d.geometry import Grid
from medlift3d.projector import DTSGeometry, ParallelGeometry, Projector


@pytest.fixture(scope="session")
def small_grid():
    """Deliberately tiny: these tests assert correctness, not quality, and must
    run on a CPU box in seconds."""
    return Grid.centred((20, 24, 24), (3.0, 3.0, 3.0))


@pytest.fixture(scope="session")
def parallel_projector(small_grid):
    return Projector(small_grid,
                     ParallelGeometry.covering(small_grid, 8, 180.0, det_spacing=3.0))


@pytest.fixture(scope="session")
def dts_projector(small_grid):
    return Projector(small_grid,
                     DTSGeometry.covering(small_grid, 9, 30.0, det_spacing=3.0))


@pytest.fixture(params=["parallel", "dts"])
def any_projector(request, parallel_projector, dts_projector):
    return parallel_projector if request.param == "parallel" else dts_projector
