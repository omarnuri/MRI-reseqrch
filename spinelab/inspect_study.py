"""Read the series list straight from DICOM headers — no dcm2niix, no GPU.

Answers "what is actually in this study?" in a few seconds on any machine. That
question comes first: which sequences exist decides which stages of the pipeline
can run at all, and on this study it decides whether the facet question is
answerable. Running it before a GPU session is cheap and prevents an hour of
compute spent on a study that never had the needed series.

Reuses the same classification code as the pipeline (`spinelab.sequences`), so a
disagreement between this listing and the pipeline's own inventory would be a bug
in one of them, not two independent guesses.
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from .sequences import (
    Series,
    classify_contrast,
    group_stations,
    has_fat_saturation,
    is_localizer,
    plane_from_direction,
)


@dataclass
class SeriesSummary:
    uid: str
    number: int | None = None
    description: str = ""
    n_instances: int = 0
    contrast: str = ""
    fat_sat: bool = False
    plane: str = ""
    echo_time_ms: float | None = None
    repetition_time_ms: float | None = None
    inversion_time_ms: float | None = None
    slice_thickness_mm: float | None = None
    pixel_spacing_mm: tuple[float, float] | None = None
    rows: int | None = None
    columns: int | None = None
    localizer: bool = False
    field_strength_t: float | None = None
    z_range_mm: tuple[float, float] | None = None
    station: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.contrast}{' FS' if self.fat_sat else ''} {self.plane}".strip()

    def to_dict(self) -> dict:
        return {
            "series_number": self.number,
            "description": self.description,
            "sequence_label": self.label,
            "contrast": self.contrast,
            "fat_saturated": self.fat_sat,
            "plane": self.plane,
            "n_instances": self.n_instances,
            "echo_time_ms": self.echo_time_ms,
            "repetition_time_ms": self.repetition_time_ms,
            "inversion_time_ms": self.inversion_time_ms,
            "slice_thickness_mm": self.slice_thickness_mm,
            "pixel_spacing_mm": list(self.pixel_spacing_mm) if self.pixel_spacing_mm else None,
            "matrix": [self.rows, self.columns] if self.rows else None,
            "field_strength_t": self.field_strength_t,
            "localizer": self.localizer,
            "z_range_mm": [round(v, 1) for v in self.z_range_mm] if self.z_range_mm else None,
            "station": self.station,
            "notes": self.notes,
        }


def _slice_normal(orientation) -> list[float] | None:
    """Cross product of the row and column direction cosines (6-value IOP tag)."""
    try:
        values = [float(v) for v in orientation]
    except (TypeError, ValueError):
        return None
    if len(values) < 6:
        return None
    r, c = values[0:3], values[3:6]
    return [
        r[1] * c[2] - r[2] * c[1],
        r[2] * c[0] - r[0] * c[2],
        r[0] * c[1] - r[1] * c[0],
    ]


def _as_float(value) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _z_range(datasets) -> tuple[float, float] | None:
    """Superior-inferior extent of a series in patient coordinates.

    Every slice contributes its four corners, built from ImagePositionPatient plus
    the two direction cosines scaled by the pixel spacing — so a sagittal stack,
    whose slices all share one z position but span 300 mm within the plane, gets
    its real extent rather than a single number.
    """
    lo = hi = None
    for ds in datasets:
        ipp, iop = getattr(ds, "ImagePositionPatient", None), getattr(ds, "ImageOrientationPatient", None)
        spacing = getattr(ds, "PixelSpacing", None)
        rows, cols = getattr(ds, "Rows", None), getattr(ds, "Columns", None)
        if not ipp or not iop or not spacing or not rows or not cols:
            continue
        try:
            z0 = float(ipp[2])
            row_dz, col_dz = float(iop[2]), float(iop[5])
            dr, dc = float(spacing[0]), float(spacing[1])
        except (TypeError, ValueError, IndexError):
            continue
        # z of the four in-plane corners, relative to the slice origin
        offsets = (0.0, row_dz * (int(cols) - 1) * dc, col_dz * (int(rows) - 1) * dr,
                   row_dz * (int(cols) - 1) * dc + col_dz * (int(rows) - 1) * dr)
        zs = [z0 + o for o in offsets]
        lo = min(zs) if lo is None else min(lo, *zs)
        hi = max(zs) if hi is None else max(hi, *zs)
    return None if lo is None else (lo, hi)


def inspect(source: str | Path) -> dict:
    """Summarise a DICOM study given a directory or a .zip archive."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with TemporaryDirectory() as tmp:
            with zipfile.ZipFile(source) as zf:
                zf.extractall(tmp)
            return _inspect_dir(Path(tmp), origin=str(source))
    return _inspect_dir(source, origin=str(source))


def _inspect_dir(root: Path, *, origin: str) -> dict:
    import pydicom

    groups: dict[str, list] = defaultdict(list)
    unreadable = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            unreadable += 1
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", path.parent.name))
        groups[uid].append(ds)

    series: list[SeriesSummary] = []
    for uid, datasets in groups.items():
        head = datasets[0]
        meta = {
            "SeriesDescription": getattr(head, "SeriesDescription", ""),
            "ProtocolName": getattr(head, "ProtocolName", ""),
            "SequenceName": getattr(head, "SequenceName", ""),
            "ScanOptions": getattr(head, "ScanOptions", ""),
            "ImageType": list(getattr(head, "ImageType", []) or []),
            "EchoTime": getattr(head, "EchoTime", None),
            "RepetitionTime": getattr(head, "RepetitionTime", None),
            "InversionTime": getattr(head, "InversionTime", None),
            "SliceThickness": getattr(head, "SliceThickness", None),
        }
        normal = _slice_normal(getattr(head, "ImageOrientationPatient", None))
        spacing = getattr(head, "PixelSpacing", None)
        summary = SeriesSummary(
            uid=uid,
            number=int(getattr(head, "SeriesNumber", 0) or 0) or None,
            description=str(meta["SeriesDescription"] or ""),
            n_instances=len(datasets),
            contrast=classify_contrast(meta),
            fat_sat=has_fat_saturation(meta),
            plane=plane_from_direction(normal) if normal else "unknown",
            echo_time_ms=_as_float(meta["EchoTime"]),
            repetition_time_ms=_as_float(meta["RepetitionTime"]),
            inversion_time_ms=_as_float(meta["InversionTime"]),
            slice_thickness_mm=_as_float(getattr(head, "SliceThickness", None)),
            pixel_spacing_mm=(tuple(float(v) for v in spacing[:2]) if spacing else None),
            rows=int(getattr(head, "Rows", 0) or 0) or None,
            columns=int(getattr(head, "Columns", 0) or 0) or None,
            localizer=is_localizer(meta, n_slices=len(datasets)),
            field_strength_t=_as_float(getattr(head, "MagneticFieldStrength", None)),
            z_range_mm=_z_range(datasets),
        )
        if normal is None:
            summary.notes.append("no ImageOrientationPatient — plane unknown")
        series.append(summary)

    series.sort(key=lambda s: (s.number or 0, s.description))
    stations = _assign_stations(series)
    usable = [s for s in series if not s.localizer]
    fat_sat = [s for s in usable if s.fat_sat]

    findings = []
    if not any(s.contrast == "T2" and s.plane == "sagittal" for s in usable):
        findings.append("no sagittal T2 — the segmentation stages cannot run")
    if not fat_sat:
        findings.append("no fat-suppressed series — bone-marrow and peri-facet oedema "
                        "are not assessable with this study")
    else:
        planes = sorted({s.plane for s in fat_sat})
        findings.append(f"fat-suppressed series present in: {', '.join(planes)}")
        if "sagittal" not in planes:
            findings.append("the fat-suppressed series is not sagittal — masks must be "
                            "registered onto it, and its coverage of the posterior "
                            "elements decides whether a left/right comparison is valid")
    if not any(s.plane == "axial" for s in usable):
        findings.append("no axial series — thoracic facet joints are close to the coronal "
                        "plane and are poorly seen on sagittal images alone")
    if len(stations) > 1:
        findings.append(
            f"the study covers {len(stations)} craniocaudal stations "
            + "; ".join(f"({st['label']}: {st['extent_mm']:.0f} mm, "
                        f"{len(st['series'])} series)" for st in stations)
            + " — one run analyses one station, select with `spinelab run --station N`")

    return {
        "source": origin,
        "n_series": len(series),
        "n_files_unreadable": unreadable,
        "series": [s.to_dict() for s in series],
        "stations": stations,
        "what_this_means": findings,
    }


def _assign_stations(series: list[SeriesSummary]) -> list[dict]:
    """Group the series into craniocaudal blocks and stamp each with its station.

    Uses the pipeline's own grouping so this listing and the run cannot disagree.
    """
    stubs = [
        Series(path=str(s.number), name=str(s.number), contrast=s.contrast,
               fat_sat=s.fat_sat, plane=s.plane, n_slices=s.n_instances,
               localizer=s.localizer, z_range_mm=s.z_range_mm)
        for s in series
    ]
    stations = group_stations(stubs)
    by_name = {st_series.name: st.id for st in stations for st_series in st.series}
    for s in series:
        s.station = by_name.get(str(s.number))
    return [st.to_dict() for st in stations]


def format_table(report: dict) -> str:
    """Human-readable one-line-per-series listing."""
    lines = [
        f"{'#':>3}  {'description':<32} {'label':<18} {'slices':>6}  {'TE':>6} {'TR':>7} "
        f"{'TI':>6}  {'thk':>5}  {'stn':>3}  matrix",
        "-" * 113,
    ]
    for s in report["series"]:
        matrix = "x".join(str(v) for v in (s["matrix"] or [])) or "-"
        lines.append(
            f"{str(s['series_number'] or '-'):>3}  {(s['description'] or '')[:32]:<32} "
            f"{s['sequence_label']:<18} {s['n_instances']:>6}  "
            f"{_fmt(s['echo_time_ms']):>6} {_fmt(s['repetition_time_ms']):>7} "
            f"{_fmt(s['inversion_time_ms']):>6}  {_fmt(s['slice_thickness_mm']):>5}  "
            f"{(s.get('station') or '-'):>3}  {matrix}"
            + ("   [localizer]" if s["localizer"] else "")
        )
    lines.append("")
    for note in report["what_this_means"]:
        lines.append(f"  * {note}")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "-" if value is None else f"{float(value):g}"
