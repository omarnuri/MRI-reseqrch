"""DICOM PHI audit and de-identification.

The source archive in this repository is named
``OMER_NURIYEV (RAMIN)_dcm_<uuid>.zip`` and its headers carry the patient name,
birth date, institution and accession number. That archive is committed to a
**public** GitHub repository (see docs/PRIVACY.md). This module makes that
visible instead of implicit, and can write a de-identified copy.

The tag list follows the DICOM PS3.15 Basic Application Level Confidentiality
Profile for the tags that actually appear in routine MR studies. It is not a
certified de-identification implementation — private vendor tags and burned-in
annotations are out of scope, which is why remove_private_tags defaults to True
and the report names what it could not check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

#: Tags that directly identify a person or an encounter.
PHI_TAGS = (
    "PatientName", "PatientID", "PatientBirthDate", "PatientAddress",
    "PatientTelephoneNumbers", "PatientMotherBirthName", "OtherPatientIDs",
    "OtherPatientNames", "PatientBirthName", "PatientAge", "PatientSex",
    "PatientWeight", "PatientSize", "AccessionNumber", "StudyID",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "PhysiciansOfRecord", "RequestingPhysician", "StudyDescription",
    "RequestedProcedureDescription", "AdmittingDiagnosesDescription",
    "PatientComments", "AdditionalPatientHistory", "DeviceSerialNumber",
    "StationName",
)

#: Tags whose value is replaced rather than emptied, so the study still loads.
REPLACEMENTS = {
    "PatientName": "ANON",
    "PatientID": "ANON",
    "PatientBirthDate": "",
    "PatientSex": "O",
    "AccessionNumber": "",
    "StudyID": "",
}

#: Kept on purpose: these drive sequence identification and geometry.
KEEP_TAGS = (
    "SeriesDescription", "ProtocolName", "SequenceName", "ScanOptions",
    "EchoTime", "RepetitionTime", "InversionTime", "ImageType",
    "ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing",
    "SliceThickness", "MagneticFieldStrength", "Manufacturer",
)


def iter_dicom_files(root: str | Path, limit: int | None = None) -> Iterable[Path]:
    root = Path(root)
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in (".zip", ".json", ".nii", ".gz"):
            continue
        yield path
        count += 1
        if limit is not None and count >= limit:
            return


def audit(root: str | Path, sample: int = 40) -> dict:
    """Report which PHI tags are present in a sample of the study's files."""
    import pydicom

    found: dict[str, set[str]] = {}
    n_read = 0
    n_failed = 0
    private_tags = 0
    for path in iter_dicom_files(root, limit=sample):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception:
            n_failed += 1
            continue
        n_read += 1
        for tag in PHI_TAGS:
            value = getattr(ds, tag, None)
            if value in (None, ""):
                continue
            found.setdefault(tag, set()).add(str(value)[:64])
        private_tags += sum(1 for el in ds if el.tag.is_private)

    return {
        "files_sampled": n_read,
        "files_unreadable": n_failed,
        "phi_tags_present": sorted(found),
        "phi_examples": {k: sorted(v)[:2] for k, v in sorted(found.items())},
        "private_tag_elements_seen": private_tags,
        "not_checked": [
            "burned-in annotations in pixel data",
            "vendor private tags beyond element count",
            "file and directory names",
        ],
        "verdict": ("PHI present — do not publish this study or its archive"
                    if found else "no standard PHI tags found in the sample"),
    }


def deidentify(src: str | Path, dst: str | Path, *, remove_private_tags: bool = True) -> dict:
    """Write a de-identified copy of a DICOM tree. Returns a summary."""
    import pydicom

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    cleared: set[str] = set()
    for path in iter_dicom_files(src):
        try:
            ds = pydicom.dcmread(str(path))
        except Exception:
            failed += 1
            continue
        for tag in PHI_TAGS:
            if tag in ds:
                if tag in REPLACEMENTS:
                    setattr(ds, tag, REPLACEMENTS[tag])
                else:
                    delattr(ds, tag)
                cleared.add(tag)
        if remove_private_tags:
            ds.remove_private_tags()
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "spinelab.dicom_audit.deidentify (PS3.15 basic subset)"
        out = dst / f"{written:05d}.dcm"
        ds.save_as(str(out), enforce_file_format=False)
        written += 1
    return {
        "files_written": written,
        "files_unreadable": failed,
        "tags_cleared": sorted(cleared),
        "output_dir": str(dst),
        "caveat": "Not certified de-identification. Pixel-data annotations are NOT removed.",
    }
