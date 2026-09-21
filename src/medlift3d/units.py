"""Attenuation units.

One rule, enforced everywhere in this package: the quantity that models predict,
projectors integrate, and losses compare is the **linear attenuation coefficient**
`mu`, in mm^-1, non-negative and *unbounded above*.

Never apply a bounded activation (sigmoid, tanh) to mu, and never re-normalise a
reconstruction with the HU affine map -- that silently collapses a density-unit
volume to a near-constant field.
"""
from __future__ import annotations

import numpy as np

# Linear attenuation of water at ~70 keV effective energy (mm^-1).
# Mass attenuation 0.1929 cm^2/g at rho = 1 g/cm^3 -> 0.1929 cm^-1 -> 0.01929 mm^-1.
MU_WATER: float = 0.0193

# The single diagnostic HU window for this project. Imported by preprocessing,
# simulation and evaluation alike so they cannot drift apart.
HU_CLIP: tuple[float, float] = (-1000.0, 400.0)

# mu at the top of the window; use as the physical clamp in solvers.
MU_MAX: float = MU_WATER * (1.0 + HU_CLIP[1] / 1000.0)

# Nodule segmentation threshold. Solid pulmonary nodules are soft-tissue density
# against roughly -800 HU parenchyma, so -300 HU separates them robustly.
NODULE_HU_THRESHOLD: float = -300.0


def hu_to_mu(hu):
    """HU -> linear attenuation (mm^-1). Clipped to HU_CLIP, so mu >= 0.

    `np.clip` handles scalars, lists and arrays alike; branching on
    `isinstance(hu, np.ndarray)` silently failed on everything else.
    """
    return MU_WATER * (1.0 + np.clip(hu, *HU_CLIP) / 1000.0)


def mu_to_hu(mu):
    """Linear attenuation (mm^-1) -> HU."""
    return (mu / MU_WATER - 1.0) * 1000.0


NODULE_MU_THRESHOLD: float = float(hu_to_mu(NODULE_HU_THRESHOLD))


def apply_poisson(projections, i0: float = 1e5, rng=None):
    """Add photon noise to *post-log* line integrals.

    Noise is Poisson on the detected counts, so it must be applied in intensity
    space and transformed back:  I = I0 exp(-p),  I~ ~ Poisson(I),  p~ = -log(I~/I0).
    """
    rng = rng or np.random.default_rng(0)
    intensity = i0 * np.exp(-np.asarray(projections, dtype=np.float64))
    noisy = rng.poisson(np.clip(intensity, 0.0, None))
    noisy = np.clip(noisy, 1.0, None)  # guard log(0)
    return (-np.log(noisy / i0)).astype(np.float32)


# ---------------------------------------------------------------------------
# network normalisation
# ---------------------------------------------------------------------------
# Networks see mu mapped onto [-1, 1]. This is a *physical* rescaling: mu is
# bounded below by 0 (air) and above by MU_MAX (the top of the diagnostic HU
# window), so clamping a network output to [-1, 1] is exactly clamping mu to its
# physical range. Clamping an unregularised latent with no known range is not.

def mu_to_net(mu):
    return mu / MU_MAX * 2.0 - 1.0


def net_to_mu(x):
    return (x + 1.0) * 0.5 * MU_MAX
