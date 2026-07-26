"""Stage `ingest`: DICOM -> NIfTI, PHI audit, sequence inventory and picks."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from ..dicom_audit import audit
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..sequences import build_picks, describe_series
from ..utils import clean_reason, read_json, write_json


def _extract_zip(zip_path: Path, target: Path) -> dict:
    """Extract, verifying completeness instead of trusting a non-empty directory.

    ``needs_extract = not any(dir.rglob('*'))`` (the old check) treats a run that
    was interrupted half-way through unzip as "already extracted", and the
    pipeline then analyses a partial study without saying so.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        expected = len(members)
        present = sum(1 for p in target.rglob("*") if p.is_file())
        if present >= expected and expected > 0:
            return {"extracted": False, "files": present, "expected": expected,
                    "note": "archive already fully extracted"}
        target.mkdir(parents=True, exist_ok=True)
        zf.extractall(target)
        present = sum(1 for p in target.rglob("*") if p.is_file())
    return {"extracted": True, "files": present, "expected": expected,
            "complete": present >= expected}


def _run_dcm2niix(dicom_dir: Path, nifti_dir: Path) -> tuple[int, str, str]:
    nifti_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix",
        "-z", "y",          # gzip
        "-b", "y",          # BIDS sidecar — the pipeline needs TE/TR/TI/ScanOptions
        "-ba", "n",         # keep the sidecar unanonymised so the PHI audit is honest
        "-f", "%p_%s_%d",   # protocol_series_description: unique per series
        "-o", str(nifti_dir),
        str(dicom_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    source = Path(cfg.dicom_source) if cfg.dicom_source else None
    if source is None or not source.exists():
        raise SkipStage(f"dicom_source not found: {cfg.dicom_source!r}")

    extract_info: dict = {}
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_info = _extract_zip(source, cfg.dicom_dir)
        dicom_root = cfg.dicom_dir
    elif source.is_dir():
        dicom_root = source
    else:
        raise SkipStage(f"dicom_source must be a directory or .zip, got {source}")

    phi = audit(dicom_root)
    write_json(cfg.results_dir / "phi_audit.json", phi)

    if shutil.which("dcm2niix") is None:
        raise SkipStage("dcm2niix is not installed (apt-get install -y dcm2niix)")

    existing = sorted(cfg.nifti_dir.glob("*.nii*"))
    if existing:
        rc, out, err = 0, f"reusing {len(existing)} existing NIfTI volumes", ""
    else:
        rc, out, err = _run_dcm2niix(dicom_root, cfg.nifti_dir)

    nifti_files = sorted(cfg.nifti_dir.glob("*.nii.gz")) + sorted(cfg.nifti_dir.glob("*.nii"))
    if not nifti_files:
        return StageResult(
            name="ingest", status=Status.FAILED, evidence=Evidence.MEASUREMENT,
            reason=f"dcm2niix produced no volumes (rc={rc}): {clean_reason(err or out)}",
            data={"phi_audit": phi, "extract": extract_info},
        )

    series = []
    for path in nifti_files:
        sidecar = Path(str(path).replace(".nii.gz", ".json").replace(".nii", ".json"))
        meta = read_json(sidecar, {}) or {}
        img = _load_header(path)
        series.append(describe_series(
            meta,
            path=str(path),
            name=path.name,
            shape=img["shape"],
            voxel_mm=img["zooms"],
            normal=img["slice_normal"],
        ))

    picks = build_picks(series, min_slices=cfg.min_series_slices)
    inventory = [s.to_dict() for s in series]
    write_json(cfg.intermediate_dir / "sequence_inventory.json", inventory)
    write_json(cfg.intermediate_dir / "sequence_picks.json", picks)

    status = Status.OK if picks.get("T2_SAG") else Status.PARTIAL
    reason = None if status is Status.OK else "no usable sagittal T2 in this study"
    return StageResult(
        name="ingest",
        status=status,
        evidence=Evidence.MEASUREMENT,
        reason=reason,
        data={
            "n_series": len(series),
            "series": inventory,
            "picks": picks,
            "phi_audit": phi,
            "extract": extract_info,
            "dcm2niix_returncode": rc,
        },
        artifacts=[str(cfg.intermediate_dir / "sequence_inventory.json")],
    )


def _load_header(path: Path) -> dict:
    """Shape, voxel size and slice-normal direction in RAS, without loading data.

    Deliberately does NOT reorient to canonical: after reorientation the third
    axis is always the one closest to superior-inferior, so every study would
    look axial. The plane is a property of the acquisition, so it must be read
    from the original affine.
    """
    import nibabel as nib
    import numpy as np

    img = nib.load(str(path))
    affine = np.asarray(img.affine, dtype=float)
    # Third column = voxel step along the slice axis, in RAS world coordinates.
    step = affine[:3, 2]
    norm = float(np.linalg.norm(step))
    normal = (step / norm).tolist() if norm > 1e-9 else None
    return {"shape": tuple(int(s) for s in img.shape[:3]),
            "zooms": tuple(float(z) for z in img.header.get_zooms()[:3]),
            "slice_normal": normal}
