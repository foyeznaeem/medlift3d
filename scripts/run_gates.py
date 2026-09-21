#!/usr/bin/env python
"""Run the four correctness gates and print a pass/fail table.

    python scripts/run_gates.py

Gates G1-G4 from `docs/IMPLEMENTATION.md`. Each one guards a failure mode that
is silent -- it produces plausible loss curves and wrong reconstructions -- so
they are checked before any GPU time is spent, not after.

  G1  geometry      one canonical grid, affine never dropped
  G2  projector     differentiable, and bp is the exact adjoint of fp
  G3  units         mu in mm^-1; water cylinder line integral = mu_water * 2R
  G4  metrics       a perfect reconstruction scores perfectly
  +   leakage, diffusion config, ROI exactness, Gaussian field
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GATES = [
    ("G1  geometry / affine discipline", "tests/test_geometry.py"),
    ("G2  differentiable projector + adjoint", "tests/test_adjoint.py"),
    ("G3  physical units", "tests/test_units.py"),
    ("G4  metrics that mean something", "tests/test_metrics.py"),
    ("--  no data leakage", "tests/test_no_leakage.py"),
    ("--  diffusion train/sample config", "tests/test_diffusion_config.py"),
    ("--  ROI decomposition exactness", "tests/test_roi_exact.py"),
    ("--  Gaussian field", "tests/test_gaussians.py"),
    ("--  solver data consistency", "tests/test_solver.py"),
]


def main():
    print(f"MedLift-3D gates  ({ROOT})\n" + "=" * 62)
    results = []
    for name, path in GATES:
        r = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "--no-header",
                            "-p", "no:warnings"], cwd=ROOT, capture_output=True, text=True)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()]
        summary = tail[-1] if tail else "no output"
        ok = r.returncode == 0
        results.append((name, ok, summary))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:42s} {summary[:40]}")
        if not ok:
            print("\n".join("        " + l for l in tail[-25:]))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print("=" * 62)
    print(f"{n_ok}/{len(results)} gate suites passed")
    if n_ok != len(results):
        print("\nDo not start training until every gate passes: each of these\n"
              "failure modes is silent and invalidates every downstream number.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
