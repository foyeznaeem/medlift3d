"""Compatibility shims for unmaintained dependencies.

Kept in one place so the same fix applies wherever the dependency is imported,
rather than being rediscovered one traceback at a time.
"""
from __future__ import annotations


def patch_pylidc() -> None:
    """Restore the APIs pylidc needs but Python and NumPy have since removed.

    pylidc was last released in 2020 and calls several names that have gone
    away since. This is the complete set, found by grepping the installed
    package rather than by waiting for each to raise:

      configparser.SafeConfigParser   Scan.py            (gone in Python 3.12)
      np.int                          Contour.to_matrix, Annotation.bbox
      np.float                        utils, Annotation diameter/volume
      np.bool                         Annotation.boolean_mask   (all NumPy 1.24)

    Every one was a deprecated alias of a builtin, so aliasing them back is
    what the code already expects and changes no behaviour.
    """
    import configparser
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

    # Only the three pylidc uses: probing for np.object or np.str would itself
    # emit a FutureWarning.
    import numpy as np
    for name, builtin in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, name):
            setattr(np, name, builtin)


def import_pylidc():
    """`patch_pylidc()` then import, with an error that says what to do."""
    patch_pylidc()
    try:
        import pylidc as pl
    except ModuleNotFoundError as e:
        raise SystemExit(
            "pylidc is not installed, and the LIDC-IDRI route needs it: it "
            "ships the nodule annotations (1,018 scans, 6,859 readings, "
            "41,406 contours) that become the ground-truth masks.\n"
            "    pip install pylidc"
        ) from e
    return pl
