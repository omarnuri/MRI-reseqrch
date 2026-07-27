"""Stage `ingest`: DICOM -> NIfTI, PHI audit, sequence inventory and picks."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from ..dicom_audit import audit
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..runlog import event, get_logger, log_command
from ..sequences import build_picks, describe_series
from ..utils import clean_reason, read_json, write_json

log = get_logger(__name__)


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
    import time

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
    log.info("converting DICOM -> NIfTI: %s", dicom_dir)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_command("ingest", cmd, proc, seconds=time.time() - t0)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _fetch_url(url: str, dest_dir: Path) -> Path:
    """Download an archive into the workspace.

    Convenience for Colab, where the runtime's connection is fast and the operator's
    is not: the study can be pulled straight into the VM instead of being uploaded
    from a slow local link. The archive still must not live in a public repository —
    see docs/PRIVACY.md — and once that repository is private this needs a token,
    at which point Drive is the simpler path.
    """
    import urllib.parse
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.parse.urlparse(url).path).name or "study.zip"
    target = dest_dir / urllib.parse.unquote(name)
    if target.exists() and target.stat().st_size > 0:
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - explicit user-provided URL
    tmp.replace(target)
    return target


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    # Nobody should have to type a path. If the configured source is missing or is
    # still the placeholder, find the study instead of failing.
    from ..discover import discover, is_placeholder

    raw_source = cfg.dicom_source or ""
    if is_placeholder(raw_source) or not (
            raw_source.startswith(("http://", "https://")) or Path(raw_source).exists()):
        found = discover(raw_source, cache_dir=cfg.cache_dir)
        event("study_discovery", **found.to_dict())
        if not found.source:
            raise SkipStage(
                f"no study found. dicom_source={raw_source!r} ({found.how}). Put the "
                "archive in Drive (…/MyDrive/mri/study.zip), or write its path or URL "
                "into study_source.txt in the cache directory, or set SPINELAB_STUDY.")
        log.info("study discovered: %s — %s", found.source, found.how)
        raw_source = found.source
        cfg.dicom_source = found.source

    log.info("dicom_source=%r", raw_source)
    if raw_source.startswith(("http://", "https://")):
        log.info("source is a URL — downloading into the workspace")
        source = _fetch_url(raw_source, cfg.data_dir / "download")
        log.info("downloaded %s (%.1f MB)", source.name, source.stat().st_size / 1e6)
        event("source_downloaded", path=str(source), bytes=source.stat().st_size)
    else:
        source = Path(raw_source) if raw_source else None
    if source is None or not source.exists():
        raise SkipStage(f"dicom_source not found: {cfg.dicom_source!r}")

    extract_info: dict = {}
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_info = _extract_zip(source, cfg.dicom_dir)
        log.info("archive: %s", extract_info)
        event("archive_extracted", **extract_info)
        dicom_root = cfg.dicom_dir
    elif source.is_dir():
        dicom_root = source
    else:
        raise SkipStage(f"dicom_source must be a directory or .zip, got {source}")

    phi = audit(dicom_root)
    write_json(cfg.results_dir / "phi_audit.json", phi)
    log.info("PHI audit: %s (%d files sampled, tags: %s)", phi["verdict"],
             phi["files_sampled"], ", ".join(phi["phi_tags_present"]) or "none")
    event("phi_audit", verdict=phi["verdict"], files_sampled=phi["files_sampled"],
          tags=phi["phi_tags_present"])

    if shutil.which("dcm2niix") is None:
        raise SkipStage("dcm2niix is not installed (apt-get install -y dcm2niix)")

    existing = sorted(cfg.nifti_dir.glob("*.nii*"))
    if existing:
        log.info("reusing %d existing NIfTI volumes in %s", len(existing), cfg.nifti_dir)
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

    for s in series:
        log.info("series: %-28s %-14s %-9s %3d slices %s",
                 (s.description or s.name)[:28], s.sequence_label, s.plane, s.n_slices,
                 "[localizer]" if s.localizer else "")
    picks = build_picks(series, min_slices=cfg.min_series_slices)
    log.info("picks: %s", {k: (Path(v).name if isinstance(v, str) and v.endswith(".gz") else v)
                           for k, v in picks.items() if k != "limitations"})
    for limitation in picks.get("limitations", []):
        log.warning("study limitation: %s", limitation)
        event("problem", stage="ingest", detail=limitation)
    event("series_inventory", n_series=len(series),
          series=[[s.description, s.sequence_label, s.plane, s.n_slices, s.localizer]
                  for s in series],
          picks={k: v for k, v in picks.items() if k != "limitations"})
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
