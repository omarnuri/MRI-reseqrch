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

#: Vendor names for a positioning scan. "scano"/"scanogram"/"topogram" are the ones
#: this study actually uses ("Scano_SAG"), and none of them contain the word
#: "localizer" — a token list alone is never enough, hence the geometric rule below.
_LOCALIZER_TOKENS = ("localizer", "localiser", "scout", "survey", "smartbrain", "loc_",
                     "3pl", "scano", "scanogram", "topogram", "positioning")
#: A positioning scan is thick and short. Real diagnostic series in the spine are
#: <= 5 mm; a 10 mm slab with a handful of slices exists only to place the others.
_LOCALIZER_MIN_THICKNESS_MM = 7.0
_LOCALIZER_MAX_SLICES = 8
#: Fat-suppression techniques. "fs" needs word boundaries; "stir"/"tirm"/"spair"
#: are unambiguous. Dixon water-only images are fat-suppressed by construction.
_FATSAT_PATTERNS = (
    r"\bstir\b", r"\btirm\b", r"\bspair\b", r"\bspir\b", r"\bfatsat\b",
    r"\bfat[_\- ]?sat\b", r"(?<![a-z])fs(?![a-z])", r"\bsat\b", r"\bdixon\b",
    r"\bwater\b", r"\bfse?fs\b",
)
_IR_PATTERNS = (r"\bstir\b", r"\btirm\b", r"\bir\b", r"\bflair\b")
#: Above this echo time a T2 sequence is a myelogram: only free fluid keeps signal.
MYELOGRAPHY_TE_MS = 250.0


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def is_localizer(meta: dict, n_slices: int | None = None) -> bool:
    """True for survey/localiser series, which are never analysis inputs.

    Four independent signals, because any one of them misses real studies: the
    vendor's name for the series, the DICOM ImageType, the geometry (a thick,
    short slab is a positioning scan whatever it is called), and the timing.

    The timing rule exists because the 2026-08-21 study arrived with every
    SeriesDescription stripped by the exporter. Its AutoAlign scouts are 112
    slices of 1.7 mm — thin and long, so no geometric rule touches them — and with
    sidecar timings they classify as T1, which would have made a scout the T1
    sagittal volume of the analysis.
    """
    text = _norm(meta.get("SeriesDescription", "")) + " " + _norm(meta.get("ProtocolName", ""))
    if _matches_any(text, [rf"\b{re.escape(t)}\b" for t in _LOCALIZER_TOKENS]):
        return True
    image_type = " ".join(str(x).lower() for x in (meta.get("ImageType") or []))
    if "localizer" in image_type or "survey" in image_type or "projection image" in image_type:
        return True
    thickness = meta.get("SliceThickness")
    try:
        thickness = float(thickness) if thickness not in (None, "") else None
    except (TypeError, ValueError):
        thickness = None
    if (thickness is not None and thickness >= _LOCALIZER_MIN_THICKNESS_MM
            and n_slices is not None and n_slices <= _LOCALIZER_MAX_SLICES):
        return True
    # A single slice is never a diagnostic spine volume: it is a MIP/projection
    # slab (series 20 and 25 of that study are one 52 mm slab each).
    if n_slices == 1:
        return True
    # No description, no timing: a derived or composed picture rather than an
    # acquisition. Every real MR series records TR and TE.
    if not text.strip() and _raw_float(meta.get("RepetitionTime")) is None \
            and _raw_float(meta.get("EchoTime")) is None:
        return True
    return _is_survey_timing(meta)


#: A survey scan is an ultrafast gradient echo. Diagnostic spin-echo spine series
#: sit an order of magnitude above this: the fastest in the 2026-08-21 study is a
#: T1 TSE at TR 478 ms.
_SURVEY_MAX_TR_MS = 50.0
_SURVEY_MAX_TE_MS = 10.0


def _is_survey_timing(meta: dict) -> bool:
    """Ultrafast gradient-echo timing, i.e. a positioning scan.

    TR and TE are read together rather than through `_ms`, which decides the unit
    of each value on its own: TR 4.2 is 4.2 ms in a DICOM header and 4.2 s in a
    dcm2niix sidecar, and per-value guessing turns a 4.2 ms scout into a 4200 ms
    T2. Taking the pair and keeping the reading in which both numbers are
    physically plausible resolves it — TE 2380 ms does not exist, TE 2.38 ms does.

    Fat-suppression evidence vetoes the rule: a Dixon/VIBE volume has the same
    timing and is a diagnostic sequence (SPINEPS ships a model for it).
    """
    timing = _joint_timing_ms(meta)
    if timing is None or has_fat_saturation(meta):
        return False
    tr, te = timing
    return tr <= _SURVEY_MAX_TR_MS and te <= _SURVEY_MAX_TE_MS


def _joint_timing_ms(meta: dict) -> tuple[float, float] | None:
    """(TR, TE) in milliseconds, choosing the unit that makes both plausible."""
    tr, te = _raw_float(meta.get("RepetitionTime")), _raw_float(meta.get("EchoTime"))
    if tr is None or te is None:
        return None
    for scale in (1.0, 1000.0):  # header milliseconds first, then sidecar seconds
        tr_ms, te_ms = tr * scale, te * scale
        if 0.5 <= te_ms <= 500.0 and 1.0 <= tr_ms <= 20000.0:
            return tr_ms, te_ms
    return None


def _raw_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


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
    z_range_mm: tuple[float, float] | None = None
    """Superior-inferior extent of the volume in world (scanner) coordinates.

    Two acquisitions belong to the same station when these overlap. Without it a
    study that covers two levels of the spine is silently reduced to one."""
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
            "z_range_mm": list(self.z_range_mm) if self.z_range_mm else None,
            "notes": self.notes,
        }


def describe_series(meta: dict, *, path: str, name: str, shape, voxel_mm, normal=None,
                    z_range_mm=None) -> Series:
    """Build a Series record from a dcm2niix sidecar plus image geometry."""
    plane = plane_from_direction(normal) if normal is not None else PLANE_UNKNOWN
    notes: list[str] = []
    if plane == PLANE_UNKNOWN:
        plane = plane_from_description(meta)
        if plane != PLANE_UNKNOWN:
            notes.append("plane taken from series description (no usable affine)")
    n_slices = 0
    if shape is not None and len(shape) >= 3:
        # In a NIfTI the third axis IS the slice axis — the affine's third column is
        # the step between slices, which is where `normal` came from. Mapping the
        # plane to an axis index instead (sagittal->0 and so on) assumes canonical
        # RAS ordering, which raw dcm2niix output does not have: it reported 512
        # "slices" for a 512x512x17 sagittal stack, and that number then drove
        # series selection.
        n_slices = int(shape[2])
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
        localizer=is_localizer(meta, n_slices=n_slices),
        z_range_mm=(tuple(float(v) for v in z_range_mm) if z_range_mm else None),
        notes=notes,
    )


def pick(
    series: Sequence[Series],
    *,
    contrast: str | None = None,
    plane: str | None = None,
    fat_sat: bool | None = None,
    min_slices: int = 5,
    allow_survey: bool = False,
) -> Series | None:
    """Best matching volume, or None.

    "Best" = not a localiser, enough slices, then most slices, then finest
    in-plane resolution. The old code took ``candidates[0]``, i.e. whatever
    dcm2niix happened to name first.

    `allow_survey` exists for one case only: a level of the spine that the study
    covers with nothing but positioning scans, selected deliberately by station.
    """
    cands = [
        s for s in series
        if (allow_survey or not s.localizer)
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


@dataclass
class Station:
    """One craniocaudal block of a study: the volumes that image the same levels.

    A session can cover two levels of the spine — the 2026-08-21 study does — and
    every pick is a single volume per role. Without this grouping the analysis
    silently reduces to whichever block has one more slice.
    """

    id: int
    series: list[Series] = field(default_factory=list)
    z_range_mm: tuple[float, float] = (0.0, 0.0)
    unplaced: list[str] = field(default_factory=list)
    """Names of volumes with no usable geometry; listed, never silently dropped."""
    survey_only: bool = False
    """This block exists only on positioning scans.

    The 2026-08-21 study images the thoracic spine — the painful region — solely on
    the AutoAlign survey (3D gradient echo, TR 4.2 / TE 2.4, 1.7 mm). Shape is there;
    tissue contrast is not. Such a block is never chosen automatically and, when
    chosen explicitly, every signal-based stage refuses to run on it."""

    @property
    def label(self) -> str:
        return f"station {self.id}" + (" (survey only)" if self.survey_only else "")

    @property
    def extent_mm(self) -> float:
        return self.z_range_mm[1] - self.z_range_mm[0]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "z_range_mm": [round(v, 1) for v in self.z_range_mm],
            "extent_mm": round(self.extent_mm, 1),
            "survey_only": self.survey_only,
            "series": [s.name for s in self.series],
            "sequences": sorted({f"{s.sequence_label} {s.plane}" for s in self.series}),
            "unplaced": list(self.unplaced),
        }


def _overlap_groups(series: Sequence[Series]) -> list[tuple[list[Series], float, float]]:
    """Volumes whose superior-inferior extents overlap, merged transitively."""
    groups: list[tuple[list[Series], float, float]] = []
    for s in sorted(series, key=lambda s: -s.z_range_mm[1]):
        lo, hi = float(s.z_range_mm[0]), float(s.z_range_mm[1])
        for i, (members, g_lo, g_hi) in enumerate(groups):
            if lo <= g_hi and hi >= g_lo:
                groups[i] = (members + [s], min(g_lo, lo), max(g_hi, hi))
                break
        else:
            groups.append(([s], lo, hi))
    groups.sort(key=lambda g: -g[2])  # superior block first
    return groups


def _survey_groups(series: Sequence[Series]) -> list[tuple[list[Series], float, float]]:
    """Positioning scans grouped by acquisition block, not by overlap.

    Overlap is the wrong relation here. A session runs the survey once per table
    position, and a long locator that reaches across two of them would chain all of
    them into a single 1100 mm "station" — which then makes the pick between three
    different levels of the spine arbitrary. Blocks are therefore seeded by the
    widest volume, and a volume joins a block only if it mostly lies inside it.
    """
    groups: list[tuple[list[Series], float, float]] = []
    for s in sorted(series, key=lambda s: -(s.z_range_mm[1] - s.z_range_mm[0])):
        lo, hi = float(s.z_range_mm[0]), float(s.z_range_mm[1])
        span = max(hi - lo, 1e-6)
        for i, (members, g_lo, g_hi) in enumerate(groups):
            inside = min(hi, g_hi) - max(lo, g_lo)
            if inside / span >= 0.8:
                groups[i] = (members + [s], min(g_lo, lo), max(g_hi, hi))
                break
        else:
            groups.append(([s], lo, hi))
    groups.sort(key=lambda g: -g[2])
    return groups


def group_stations(series: Sequence[Series], *, min_slices: int = 5,
                   include_survey: bool = True) -> list[Station]:
    """Split a study into craniocaudal blocks, numbered from the head down.

    Two volumes belong together when their superior-inferior extents overlap; the
    relation is applied transitively, so an axial block that overlaps only part of
    a sagittal stack still joins it. Volumes without geometry cannot be placed, so
    they are attached to every station's `unplaced` list rather than dropped.

    Diagnostic and survey volumes are grouped separately and the diagnostic blocks
    are numbered first. Grouping them together would collapse the study into one
    station, because a whole-spine positioning scan overlaps everything — and it
    would also hide the fact that a level is covered *only* by a survey scan, which
    is exactly what the operator needs to know.
    """
    long_enough = [s for s in series if s.n_slices >= min_slices and s.z_range_mm]
    diagnostic = [s for s in long_enough if not s.localizer]
    survey = [s for s in long_enough if s.localizer]
    unplaced = [s.name for s in series
                if s.n_slices >= min_slices and not s.z_range_mm and not s.localizer]

    stations = [
        Station(id=i, series=members, z_range_mm=(lo, hi), unplaced=list(unplaced))
        for i, (members, lo, hi) in enumerate(_overlap_groups(diagnostic), start=1)
    ]
    if include_survey:
        stations += [
            Station(id=len(stations) + i, series=members, z_range_mm=(lo, hi),
                    survey_only=True)
            for i, (members, lo, hi) in enumerate(_survey_groups(survey), start=1)
        ]
    return stations


def build_picks(series: Sequence[Series], *, min_slices: int = 5,
                station: Station | None = None,
                stations: Sequence[Station] | None = None) -> dict[str, Any]:
    """The sequence choices the rest of the pipeline depends on.

    With `station` given, only that block's volumes are candidates and the blocks
    left unanalysed are named in the limitations, with the flag that selects them.
    """
    if station is not None:
        if stations is None:
            stations = group_stations(series, min_slices=min_slices)
        series = list(station.series)

    if station is not None and station.survey_only:
        # Contrast means nothing here — every volume is the same gradient echo —
        # so the choice is by plane and coverage, and the roles are filled only so
        # the segmentation stages have an input. Nothing signal-based may follow.
        t2_sag = pick(series, plane=PLANE_SAGITTAL, min_slices=min_slices, allow_survey=True)
        t1_sag = None
        t2_ax = pick(series, plane=PLANE_AXIAL, min_slices=min_slices, allow_survey=True)
        fatsat = None
    else:
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
        "FATSAT_TE_MS": fatsat.echo_time_ms if fatsat else None,
        "station": station.id if station else None,
        "survey_only": bool(station and station.survey_only),
    }
    limitations = []
    if picks["survey_only"]:
        limitations.append(
            "THIS BLOCK IS COVERED ONLY BY POSITIONING SCANS (fast 3D gradient echo). "
            "Shape can be measured — vertebral heights, wedge angles, curvature — and "
            "even that depends on segmentation models working off their training "
            "distribution, so the masks must be inspected. Everything signal-based "
            "(marrow, disc signal, facet comparison, texture) is NOT assessable and "
            "those stages refuse to run.")
    if fatsat is not None and (fatsat.echo_time_ms or 0) >= MYELOGRAPHY_TE_MS:
        # The gate this pipeline was written around is a STIR at TE ~ 60-100 ms.
        # A fat-suppressed 3D SPACE at TE 437 ms is an MR myelogram: bright where
        # there is free fluid, near-silent in marrow. It passes the fat-sat check
        # and must not be read as if it were a STIR.
        limitations.append(
            f"the fat-suppressed series is heavily T2-weighted (TE "
            f"{fatsat.echo_time_ms:.0f} ms, myelography-type): it shows free fluid "
            "well and bone-marrow oedema poorly — not a substitute for STIR/TIRM")
    if station is not None and stations and len(stations) > 1:
        others = [st for st in stations if st.id != station.id]
        listing = "; ".join(
            f"{st.label} ({st.z_range_mm[0]:.0f}…{st.z_range_mm[1]:.0f} mm, "
            f"{len(st.series)} series)" for st in others)
        limitations.append(
            f"this study covers {len(stations)} craniocaudal stations; only "
            f"{station.label} is analysed in this run. Not analysed: {listing}. "
            f"Run again with --station {others[0].id} for the next one.")
        uncovered = _diagnostic_gaps(stations)
        for lo, hi in uncovered:
            limitations.append(
                f"no diagnostic series covers {lo:.0f}…{hi:.0f} mm ({hi - lo:.0f} mm of "
                "spine) — between the diagnostic blocks. Any finding there would have to "
                "come from a positioning scan.")
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
    # dcm2niix splits one acquisition into several volumes when the slices are not
    # a single consistent stack (differing echoes, gaps, two slabs). The axial
    # series of this study comes out as 31 + 35 slices; picking one of them silently
    # analyses half the spine.
    for group, members in _split_series(series).items():
        contrast, plane = group
        counts = ", ".join(str(m.n_slices) for m in members)
        chosen = max(members, key=lambda m: m.n_slices)
        limitations.append(
            f"the {plane} {contrast} acquisition arrived as {len(members)} separate volumes "
            f"({counts} slices); only the largest ({chosen.n_slices} slices) is analysed, so "
            f"part of the covered anatomy is left out")

    picks["limitations"] = limitations
    return picks


def _diagnostic_gaps(stations: Sequence[Station]) -> list[tuple[float, float]]:
    """Craniocaudal ranges between diagnostic blocks, i.e. levels nobody imaged well.

    This is the number that matters when the symptom points at a level: the
    2026-08-21 study leaves ~220 mm between the cervical and lumbar blocks.
    """
    blocks = sorted((st.z_range_mm for st in stations if not st.survey_only),
                    key=lambda r: r[1], reverse=True)
    gaps = []
    for upper, lower in zip(blocks, blocks[1:]):
        if upper[0] > lower[1]:
            gaps.append((lower[1], upper[0]))
    return gaps


def _split_series(series: Sequence[Series], min_slices: int = 5) -> dict:
    """Groups of volumes that look like one acquisition split into several files."""
    groups: dict[tuple[str, str], list[Series]] = {}
    for s in series:
        if s.localizer or s.n_slices < min_slices:
            continue
        key = (s.sequence_label, s.plane)
        groups.setdefault(key, []).append(s)
    return {k: v for k, v in groups.items() if len(v) > 1}
