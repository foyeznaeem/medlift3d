"""Offline-safe run logging.

Kaggle notebooks have no internet by default, so wandb is not an option. Metrics
go to a CSV next to the checkpoints, which also means a killed 12-hour session
leaves its history behind for the next one to append to.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path


class CsvLogger:
    def __init__(self, path, fieldnames=None, resume: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames) if fieldnames else None
        self._start = time.time()
        if not resume and self.path.exists():
            self.path.unlink()
        self._has_header = self.path.exists() and self.path.stat().st_size > 0

    def log(self, **row):
        row.setdefault("wall_s", round(time.time() - self._start, 2))
        if self.fieldnames is None:
            self.fieldnames = list(row)
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._has_header:
                w.writeheader()
                self._has_header = True
            w.writerow(row)

    def plot(self, x: str, ys, out_png=None):
        """Render the logged curves; useful as a notebook cell output."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.read_csv(self.path)
        ys = [y for y in ([ys] if isinstance(ys, str) else ys) if y in df.columns]
        if not ys:
            return None
        fig, a = plt.subplots(figsize=(6, 3.4))
        for y in ys:
            a.plot(df[x], df[y], label=y, lw=1.2)
        a.set_xlabel(x)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
        fig.tight_layout()
        out = Path(out_png or self.path.with_suffix(".png"))
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out
