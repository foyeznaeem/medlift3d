"""Shared numerics: total variation, seeding, checkpoint plumbing."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# total variation
# ---------------------------------------------------------------------------

def _fwd_diff(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Forward difference with a zero-flux (replicate) boundary."""
    return torch.diff(x, dim=axis, append=x.narrow(axis, x.shape[axis] - 1, 1))


def _bwd_div(g: torch.Tensor, axis: int) -> torch.Tensor:
    """Adjoint of `_fwd_diff` up to sign, i.e. the backward difference."""
    return torch.diff(g, dim=axis, prepend=torch.zeros_like(g.narrow(axis, 0, 1)))


def tv(x: torch.Tensor, eps: float = 1e-8, axes=(0, 1, 2)) -> torch.Tensor:
    """Isotropic total variation over `axes`. Differentiable."""
    sq = sum(_fwd_diff(x, a) ** 2 for a in axes)
    return torch.sqrt(sq + eps ** 2).sum()


def tv_grad(x: torch.Tensor, eps: float = 1e-6, axes=(0, 1, 2)) -> torch.Tensor:
    """Analytic gradient of smoothed isotropic TV: -div(grad x / |grad x|).

    Written out rather than obtained by autograd because the solver calls it
    inside `no_grad` blocks many times per reverse-diffusion step.
    """
    d = [_fwd_diff(x, a) for a in axes]
    norm = torch.sqrt(sum(di ** 2 for di in d) + eps ** 2)
    return -sum(_bwd_div(di / norm, a) for di, a in zip(d, axes))


def tv_grad_z(x: torch.Tensor, eps: float = 1e-6, axis: int = 0) -> torch.Tensor:
    """Gradient of anisotropic TV along one axis only.

    This is the inter-slice coupling that lets a purely 2-D prior produce a
    coherent 3-D volume: the projector couples slices physically, and this term
    suppresses the residual slice-to-slice flicker.
    """
    return tv_grad(x, eps, axes=(axis,))


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path, **payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
