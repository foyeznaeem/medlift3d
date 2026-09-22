#!/usr/bin/env python
"""Download LIDC-IDRI CT series from TCIA, in the layout pylidc expects.

The series list comes from **pylidc's own database**, not from TCIA's
`getSeries`, and with the same `slice_thickness` filter `prepare_lidc.py`
applies. That coordination is the point: driving the two scripts from
different lists means downloading one patient and then asking pylidc to ingest
a different one, which fails with "Couldn't find DICOM files" for a scan that
was never requested.

Only pixel data is fetched. pylidc ships the annotations itself (1,018 scans,
6,859 readings, 41,406 contours in a 25 MB SQLite file), so there is nothing
else to download.

Layout is fixed by `pylidc.Scan.get_path_to_dicom_files`:

    <root>/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm

    python scripts/fetch_lidc.py --batch 2 --out /kaggle/working/lidc_dicom
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medlift3d.compat import import_pylidc

NBIA = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def select_scans(pl, limit: int, skip: int, max_slice_thickness: float):
    """The same query prepare_lidc.py runs, so the two cannot diverge."""
    scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= max_slice_thickness)
    return list(scans[skip:skip + limit])


def series_dir(root: Path, scan) -> Path:
    return root / scan.patient_id / scan.study_instance_uid / scan.series_instance_uid


def download_series(scan, root: Path, timeout: float = 300) -> Path:
    dest = series_dir(root, scan)
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "_series.zip"
    with requests.get(f"{NBIA}/getImage",
                      params={"SeriesInstanceUID": scan.series_instance_uid},
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
                    help="scans to fetch this run (a few minutes each)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many scans first; pass the same value to "
                         "prepare_lidc.py --skip so both see the same set")
    ap.add_argument("--max-slice-thickness", type=float, default=1.5,
                    help="must match prepare_lidc.py, or the two disagree on "
                         "which scans they are working with")
    ap.add_argument("--no-pylidcrc", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_pylidcrc:
        # Written before the query: pylidc needs a config file to exist even to
        # answer questions about scans whose images are not here yet.
        print(f"wrote {write_pylidcrc(args.out)}")

    pl = import_pylidc()
    scans = select_scans(pl, args.batch, args.skip, args.max_slice_thickness)
    if not scans:
        raise SystemExit(f"no scans with slice_thickness <= "
                         f"{args.max_slice_thickness} at offset {args.skip}")
    print(f"{len(scans)} scan(s) selected (skip={args.skip}, "
          f"slice_thickness <= {args.max_slice_thickness} mm):")
    for s in scans:
        print(f"  {s.patient_id}  {s.slice_thickness} mm")

    ok, failed = 0, []
    for scan in tqdm(scans, desc="downloading"):
        dest = series_dir(args.out, scan)
        if dest.exists() and any(dest.glob("*.dcm")):
            ok += 1
            continue
        try:
            download_series(scan, args.out)
            ok += 1
        except Exception as e:                                      # noqa: BLE001
            failed.append((scan.patient_id, f"{type(e).__name__}: {e}"))

    print(f"\n{ok}/{len(scans)} scans present under {args.out}")
    for p in sorted(args.out.glob("*/*/*"))[:3]:
        print(f"  {p.relative_to(args.out)}  ({len(list(p.glob('*.dcm')))} .dcm)")
    if failed:
        print(f"{len(failed)} failed, first: {failed[0]}")
        print("A 403/404 means TCIA's public getImage changed or now needs "
              "auth; check their REST API docs.")
    if ok == 0:
        raise SystemExit("nothing downloaded")

    print(f"\nnow run:  python scripts/prepare_lidc.py --source pylidc "
          f"--limit {args.batch} --skip {args.skip} "
          f"--max-slice-thickness {args.max_slice_thickness} --device cuda")


if __name__ == "__main__":
    main()
