"""Sequence identification and selection.

Two real bugs from the previous version are fixed here.

1. Fat saturation was detected with ``'fs' in desc``. Every ordinary "T2 TSE FSE"
   contains ``fs``, so plain T2 fast-spin-echo series were labelled ``T2_FS`` —
   i.e. the pipeline believed it had a fat-saturated sequence when it did not,
   and then reported bright-voxel counts as if they meant oedema. Token matching
   now uses word boundaries and prefers the DICOM InversionTime/ScanOptions
   evidence over free-text.

2. Imaging plane was parsed out of SeriesDescription (``'ax' in desc``), which
   matches "relax", "max", "sax"… The plane now comes from the image geometry
   (direction cosines), with the description used only as a fallback.

Selection also excludes localisers/scouts: a 3-slice survey could previously be
picked as the primary T2 sagittal volume for the whole analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

CONTRAST_T1 = "T1"
CONTRAST_T2 = "T2"
CONTRAST_PD = "PD"
CONTRAST_UNKNOWN = "unknown"

PLANE_SAGITTAL = "sagittal"
PLANE_CORONAL = "coronal"
PLANE_AXIAL = "axial"
PLANE_UNKNOWN = "unknown"

_LOCALIZER_TOKENS = ("localizer", "localiser", "scout", "survey", "smartbrain", "loc_", "3pl")
#: Fat-suppression techniques. "fs" needs word boundaries; "stir"/"tirm"/"spair"
#: are unambiguous. Dixon water-only images are fat-suppressed by construction.
_FATSAT_PATTERNS = (
    r"\bstir\b", r"\btirm\b", r"\bspair\b", r"\bspir\b", r"\bfatsat\b",
    r"\bfat[_\- ]?sat\b", r"(?<![a-z])fs(?![a-z])", r"\bsat\b", r"\bdixon\b",
    r"\bwater\b", r"\bfse?fs\b",
)
_IR_PATTERNS = (r"\bstir\b", r"\btirm\b", r"\bir\b", r"\bflair\b")


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def is_localizer(meta: dict) -> bool:
    """True for survey/localiser series, which are never analysis inputs."""
    text = _norm(meta.get("SeriesDescription", "")) + " " + _norm(meta.get("ProtocolName", ""))
    if _matches_any(text, [rf"\b{re.escape(t)}\b" for t in _LOCALIZER_TOKENS]):
        return True
    image_type = " ".join(str(x).lower() for x in (meta.get("ImageType") or []))
    return "localizer" in image_type or "survey" in image_type


def has_fat_saturation(meta: dict) -> bool:
    """Whether the series suppresses fat signal.

    Order of evidence: an explicit inversion time (STIR/TIRM) > ScanOptions
    fat-sat flag > vendor free text with word boundaries.
    """
    text = " ".join((
        _norm(meta.get("SeriesDescription")),
        _norm(meta.get("ProtocolName")),
        _norm(meta.get("SequenceName")),
        _norm(meta.get("ScanOptions")),
        _norm(" ".join(str(x) for x in (meta.get("ImageType") or []))),
    ))
    ti = meta.get("InversionTime")
    if ti not in (None, "", 0) and _matches_any(text, _IR_PATTERNS):
        return True
    scan_options = _norm(meta.get("ScanOptions"))
    if re.search(r"(?<![a-z])fs(?![a-z])", scan_options) or "sat" in scan_options.split():
        return True
    return _matches_any(text, _FATSAT_PATTERNS)


def classify_contrast(meta: dict) -> str:
    """T1 / T2 / PD / unknown from description first, TE+TR as fallback.

    TE/TR are in ms in dcm2niix sidecars for some vendors and in seconds for
    others; values are normalised before thresholding.
    """
    text = " ".join((
        _norm(meta.get("SeriesDescription")),
        _norm(meta.get("ProtocolName")),
        _norm(meta.get("SequenceName")),
    ))
    if re.search(r"\bt2\b|\bt2w\b|\btse\b.*\bt2\b", text):
        return CONTRAST_T2
    if re.search(r"\bt1\b|\bt1w\b", text):
        return CONTRAST_T1
    if re.search(r"\bstir\b|\btirm\b", text):
        return CONTRAST_T2  # STIR is a T2-weighted inversion-recovery acquisition
    if re.search(r"\bpd\b|\bproton\b", text):
        return CONTRAST_PD

    te, tr = _ms(meta.get("EchoTime"), "te"), _ms(meta.get("RepetitionTime"), "tr")
    if te is not None and tr is not None:
        if te > 60 and tr > 2000:
            return CONTRAST_T2
        if te < 30 and tr < 1000:
            return CONTRAST_T1
        if te < 30 and tr > 2000:
            return CONTRAST_PD
    return CONTRAST_UNKNOWN


#: Below these values a timing parameter must have been recorded in seconds.
#: dcm2niix sidecars use seconds, DICOM headers use milliseconds, and the two get
#: mixed in practice. The cut is per-parameter because the plausible ranges differ:
#: TE 12 is 12 ms (common), TR 3.5 is 3.5 s (also common).
_SECONDS_CUTOFF = {"te": 1.0, "tr": 20.0, "ti": 20.0}


def _ms(value: Any, kind: str = "te") -> float | None:
    """Normalise a TE/TR/TI value to milliseconds."""
    if value in (None, ""):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    cutoff = _SECONDS_CUTOFF.get(kind, 1.0)
    return v * 1000.0 if v < cutoff else v


def plane_from_direction(normal: Sequence[float]) -> str:
    """Imaging plane from the slice-normal vector in RAS coordinates.

    RAS axes: x = right, y = anterior, z = superior. A slice normal pointing
    mostly along x means slices stack left-to-right, i.e. sagittal images.
    """
    try:
        vals = [abs(float(v)) for v in list(normal)[:3]]
    except (TypeError, ValueError):
        return PLANE_UNKNOWN
    if len(vals) < 3 or max(vals) <= 0:
        return PLANE_UNKNOWN
    axis = vals.index(max(vals))
    return {0: PLANE_SAGITTAL, 1: PLANE_CORONAL, 2: PLANE_AXIAL}[axis]


def plane_from_description(meta: dict) -> str:
    text = " ".join((_norm(meta.get("SeriesDescription")), _norm(meta.get("ProtocolName"))))
    if re.search(r"\bsag\b|\bsagittal\b", text):
        return PLANE_SAGITTAL
    if re.search(r"\bcor\b|\bcoronal\b|\bcorronal\b", text):
        return PLANE_CORONAL
    if re.search(r"\bax\b|\baxial\b|\btra\b|\btransverse\b|\btrans\b", text):
        return PLANE_AXIAL
    return PLANE_UNKNOWN


@dataclass
class Series:
    """One NIfTI volume plus the facts needed to choose between volumes."""

    path: str
    name: str = ""
    contrast: str = CONTRAST_UNKNOWN
    fat_sat: bool = False
    plane: str = PLANE_UNKNOWN
    n_slices: int = 0
    shape: tuple[int, ...] = ()
    voxel_mm: tuple[float, ...] = ()
    description: str = ""
    echo_time_ms: float | None = None
    repetition_time_ms: float | None = None
    inversion_time_ms: float | None = None
    localizer: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def sequence_label(self) -> str:
        """Short human label: 'T2', 'T2 FS', 'STIR-like'."""
        base = self.contrast
        return f"{base} FS" if self.fat_sat else base

    @property
    def in_plane_mm(self) -> float:
        """Coarse in-plane resolution proxy used to rank candidate volumes."""
        if len(self.voxel_mm) < 3:
            return 99.0
        dims = sorted(float(v) for v in self.voxel_mm[:3])
        return (dims[0] + dims[1]) / 2.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "contrast": self.contrast,
            "fat_sat": self.fat_sat,
            "sequence_label": self.sequence_label,
            "plane": self.plane,
            "n_slices": self.n_slices,
            "shape": list(self.shape),
            "voxel_mm": [round(float(v), 3) for v in self.voxel_mm],
            "description": self.description,
            "echo_time_ms": self.echo_time_ms,
            "repetition_time_ms": self.repetition_time_ms,
            "inversion_time_ms": self.inversion_time_ms,
            "localizer": self.localizer,
            "notes": self.notes,
        }


def describe_series(meta: dict, *, path: str, name: str, shape, voxel_mm, normal=None) -> Series:
    """Build a Series record from a dcm2niix sidecar plus image geometry."""
    plane = plane_from_direction(normal) if normal is not None else PLANE_UNKNOWN
    notes: list[str] = []
    if plane == PLANE_UNKNOWN:
        plane = plane_from_description(meta)
        if plane != PLANE_UNKNOWN:
            notes.append("plane taken from series description (no usable affine)")
    n_slices = 0
    if shape is not None and len(shape) >= 3:
        # Slice count = extent along the through-plane axis.
        axis = {PLANE_SAGITTAL: 0, PLANE_CORONAL: 1, PLANE_AXIAL: 2}.get(plane)
        n_slices = int(shape[axis]) if axis is not None else int(min(shape[:3]))
    return Series(
        path=str(path),
        name=name,
        contrast=classify_contrast(meta),
        fat_sat=has_fat_saturation(meta),
        plane=plane,
        n_slices=n_slices,
        shape=tuple(int(s) for s in (shape or ())),
        voxel_mm=tuple(float(v) for v in (voxel_mm or ())),
        description=str(meta.get("SeriesDescription", "") or ""),
        echo_time_ms=_ms(meta.get("EchoTime"), "te"),
        repetition_time_ms=_ms(meta.get("RepetitionTime"), "tr"),
        inversion_time_ms=_ms(meta.get("InversionTime"), "ti"),
        localizer=is_localizer(meta),
        notes=notes,
    )


def pick(
    series: Sequence[Series],
    *,
    contrast: str | None = None,
    plane: str | None = None,
    fat_sat: bool | None = None,
    min_slices: int = 5,
) -> Series | None:
    """Best matching volume, or None.

    "Best" = not a localiser, enough slices, then most slices, then finest
    in-plane resolution. The old code took ``candidates[0]``, i.e. whatever
    dcm2niix happened to name first.
    """
    cands = [
        s for s in series
        if not s.localizer
        and s.n_slices >= min_slices
        and (contrast is None or s.contrast == contrast)
        and (plane is None or s.plane == plane)
        and (fat_sat is None or s.fat_sat == fat_sat)
    ]
    if not cands:
        return None
    return sorted(cands, key=lambda s: (-s.n_slices, s.in_plane_mm, s.name))[0]


def pick_fat_saturated(series: Sequence[Series], *, min_slices: int = 5,
                       prefer_planes: Sequence[str] = (PLANE_SAGITTAL, PLANE_AXIAL, PLANE_CORONAL)
                       ) -> Series | None:
    """Best fat-suppressed volume in the preferred plane order.

    Marrow and peri-facet oedema are only assessable on a fat-suppressed
    sequence, so the whole oedema branch of the pipeline hangs off this one pick.
    If it returns None, that branch must report "not assessable" — not a number.
    """
    for plane in prefer_planes:
        hit = pick(series, plane=plane, fat_sat=True, min_slices=min_slices)
        if hit is not None:
            return hit
    return pick(series, fat_sat=True, min_slices=min_slices)


def build_picks(series: Sequence[Series], *, min_slices: int = 5) -> dict[str, Any]:
    """The sequence choices the rest of the pipeline depends on."""
    t2_sag = pick(series, contrast=CONTRAST_T2, plane=PLANE_SAGITTAL, fat_sat=False, min_slices=min_slices) \
        or pick(series, contrast=CONTRAST_T2, plane=PLANE_SAGITTAL, min_slices=min_slices)
    t1_sag = pick(series, contrast=CONTRAST_T1, plane=PLANE_SAGITTAL, min_slices=min_slices)
    t2_ax = pick(series, contrast=CONTRAST_T2, plane=PLANE_AXIAL, min_slices=min_slices)
    fatsat = pick_fat_saturated(series, min_slices=min_slices)

    picks = {
        "T2_SAG": t2_sag.path if t2_sag else None,
        "T1_SAG": t1_sag.path if t1_sag else None,
        "T2_AX": t2_ax.path if t2_ax else None,
        "FATSAT_BEST": fatsat.path if fatsat else None,
        "FATSAT_PLANE": fatsat.plane if fatsat else None,
        "FATSAT_LABEL": fatsat.sequence_label if fatsat else None,
    }
    limitations = []
    if not t2_sag:
        limitations.append("no usable sagittal T2 — segmentation stages cannot run")
    if not fatsat:
        limitations.append(
            "no fat-suppressed sequence (STIR/TIRM/SPAIR/Dixon-water) in this study: "
            "bone-marrow and peri-facet oedema are NOT assessable"
        )
    elif fatsat.plane != PLANE_SAGITTAL:
        limitations.append(
            f"the only fat-suppressed sequence is {fatsat.plane}; resampling it onto a "
            "sagittal mask grid loses coverage, so oedema screening is limited"
        )
    if not t2_ax:
        limitations.append(
            "no axial T2 — facet joints are frontally oriented in the thoracic spine "
            "and are largely unassessable without an axial (or oblique) plane"
        )
    picks["limitations"] = limitations
    return picks
