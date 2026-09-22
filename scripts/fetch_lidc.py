#!/usr/bin/env python
"""Download LIDC-IDRI CT series from TCIA, in the layout pylidc expects.

TCIA's public NBIA API needs no key for LIDC-IDRI. `getSeries` lists the
collection (so no `.tcia` manifest is needed) and `getImage` returns a zip of
one series.

The layout matters and is not negotiable. `pylidc.Scan.get_path_to_dicom_files`
does:

    base = <configured path>/<PatientID>          # must exist, or RuntimeError
    path = base/<StudyInstanceUID>/<SeriesInstanceUID>

falling back to a recursive walk under `base` that matches on DICOM headers. So
files are written to exactly that nested path.

Annotations are NOT downloaded: pylidc ships them in its own 25 MB SQLite
database (1,018 scans, 6,859 annotations, 41,406 contours). Only pixel data is
missing, and that is what this fetches.

    python scripts/fetch_lidc.py --batch 2 --out /kaggle/working/lidc_dicom
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
NEEDED = ("SeriesInstanceUID", "StudyInstanceUID", "PatientID")


def list_series(collection: str, modality: str, timeout: float = 120) -> list[dict]:
    r = requests.get(f"{NBIA}/getSeries",
                     params={"Collection": collection, "Modality": modality},
                     timeout=timeout)
    r.raise_for_status()
    series = r.json()
    if not series:
        raise SystemExit(f"getSeries returned nothing for {collection}/{modality}")
    missing = [k for k in NEEDED if k not in series[0]]
    if missing:
        raise SystemExit(f"getSeries did not return {missing}; got "
                         f"{sorted(series[0])}")
    return series


def series_dir(root: Path, meta: dict) -> Path:
    return root / meta["PatientID"] / meta["StudyInstanceUID"] / meta["SeriesInstanceUID"]


def download_series(meta: dict, root: Path, timeout: float = 300) -> Path:
    dest = series_dir(root, meta)
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "_series.zip"
    with requests.get(f"{NBIA}/getImage",
                      params={"SeriesInstanceUID": meta["SeriesInstanceUID"]},
                      stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(zpath, "wb") as out:
            shutil.copyfileobj(r.raw, out)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    zpath.unlink()
    return dest


def write_pylidcrc(root: Path) -> Path:
    """pylidc reads exactly one key, and only from the user's home directory."""
    rc = Path.home() / ".pylidcrc"
    rc.write_text(f"[dicom]\npath = {root}\n")
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/lidc_dicom"))
    ap.add_argument("--batch", type=int, default=2,
                    help="series to fetch this run (~43 s each)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many series first, to fetch a later slice")
    ap.add_argument("--collection", default="LIDC-IDRI")
    ap.add_argument("--modality", default="CT")
    ap.add_argument("--no-pylidcrc", action="store_true",
                    help="do not write ~/.pylidcrc")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    series = list_series(args.collection, args.modality)
    print(f"{len(series)} {args.modality} series in {args.collection}")

    chosen = series[args.skip:args.skip + args.batch]
    print(f"fetching {len(chosen)} (skipping first {args.skip}) -> {args.out}")

    ok, failed = 0, []
    for meta in tqdm(chosen, desc="series"):
        dest = series_dir(args.out, meta)
        if dest.exists() and any(dest.glob("*.dcm")):
            ok += 1
            continue
        try:
            download_series(meta, args.out)
            ok += 1
        except Exception as e:                                      # noqa: BLE001
            failed.append((meta["SeriesInstanceUID"], f"{type(e).__name__}: {e}"))

    print(f"\n{ok}/{len(chosen)} series present under {args.out}")
    for p in sorted(args.out.glob("*/*/*"))[:3]:
        print(f"  {p.relative_to(args.out)}  ({len(list(p.glob('*.dcm')))} .dcm)")
    if failed:
        print(f"{len(failed)} failed, first: {failed[0]}")
        print("A 403/404 means the public getImage endpoint changed or now "
              "needs auth; check TCIA's REST API docs.")

    if not args.no_pylidcrc:
        print(f"wrote {write_pylidcrc(args.out)}:")
        print("  " + (Path.home() / ".pylidcrc").read_text().replace("\n", "\n  "))

    if ok == 0:
        raise SystemExit("nothing downloaded")
    patients = sorted(p.name for p in args.out.iterdir() if p.is_dir())
    print(f"{len(patients)} patient folders, e.g. {patients[:3]}")


if __name__ == "__main__":
    main()
