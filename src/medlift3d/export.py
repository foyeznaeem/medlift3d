"""Export reconstructions for clinical tooling.

NIfTI always. DICOM SEG + SR when `pydicom`/`highdicom` are available: a nodule
mask plus its volume measurement in a standard object is real interoperability,
which is what the report's compliance section claims but the FYDP-1 code did not
deliver (it produced only `.nii.gz`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import Grid
from .units import mu_to_hu


def save_nifti(path, volume: np.ndarray, grid: Grid, as_hu: bool = True) -> Path:
    """Write a volume with its physical frame intact.

    The affine is not optional decoration: without it a viewer renders the wrong
    aspect ratio and any downstream measurement is wrong.
    """
    import nibabel as nib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vol = mu_to_hu(volume) if as_hu else volume
    # Grid indexes (z, y, x); NIfTI wants (x, y, z).
    img = nib.Nifti1Image(np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32),
                          grid.affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))
    return path


def save_uncertainty(path, std: np.ndarray, grid: Grid) -> Path:
    """Per-voxel posterior standard deviation, in mm^-1 (not HU)."""
    return save_nifti(path, std, grid, as_hu=False)


def save_dicom_seg(path, mask: np.ndarray, grid: Grid, series_description: str,
                   source_dicom_dir=None) -> Path | None:
    """DICOM Segmentation object for the nodule mask.

    Requires `highdicom` and reference source DICOM instances; returns None when
    either is unavailable so the pipeline degrades rather than fails.
    """
    try:
        import highdicom as hd
        import pydicom
    except ImportError:
        return None
    if source_dicom_dir is None:
        return None
    src = [pydicom.dcmread(str(f)) for f in sorted(Path(source_dicom_dir).glob("*.dcm"))]
    if not src:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = hd.seg.SegmentDescription(
        segment_number=1, segment_label="pulmonary nodule",
        segmented_property_category=hd.sr.CodedConcept("M-01000", "SRT", "Morphologically Abnormal Structure"),
        segmented_property_type=hd.sr.CodedConcept("M-03010", "SRT", "Nodule"),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name="MedLift-3D", version="0.1.0", family=hd.sr.CodedConcept(
                "111067", "DCM", "Deep Learning")),
    )
    seg = hd.seg.Segmentation(
        source_images=src,
        pixel_array=(mask > 0).astype(np.uint8),
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[desc],
        series_instance_uid=hd.UID(), series_number=99,
        sop_instance_uid=hd.UID(), instance_number=1,
        manufacturer="MedLift-3D", manufacturer_model_name="MedLift-3D",
        software_versions="0.1.0", device_serial_number="0",
        series_description=series_description,
    )
    seg.save_as(str(path))
    return path


def write_measurement_report(path, records: list[dict], grid: Grid) -> Path:
    """Volume measurements as JSON, mirroring what a DICOM SR would carry.

    Written unconditionally so the numbers are always available even when
    `highdicom` is absent.
    """
    from .utils import write_json
    write_json(path, {"grid": grid.as_dict(),
                      "voxel_volume_mm3": grid.voxel_volume_mm3,
                      "measurements": records})
    return Path(path)


def save_triplanar_png(path, volume: np.ndarray, overlay: np.ndarray | None = None,
                       title: str = "") -> Path:
    """Mid-slice axial/coronal/sagittal preview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = volume.shape
    views = [volume[nz // 2], volume[:, ny // 2], volume[:, :, nx // 2]]
    ovs = ([overlay[nz // 2], overlay[:, ny // 2], overlay[:, :, nx // 2]]
           if overlay is not None else [None] * 3)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    for a, v, o, nm in zip(ax, views, ovs, ("axial", "coronal", "sagittal")):
        a.imshow(v, cmap="gray", origin="lower")
        if o is not None:
            a.contour(o > 0, levels=[0.5], colors="r", linewidths=0.8)
        a.set_title(nm, fontsize=9)
        a.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
