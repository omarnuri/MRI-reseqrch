"""Pure array maths. No file I/O, no tool calls — every function here is unit-tested.

All array inputs are expected in canonical RAS+ orientation
(axis 0 = L->R, axis 1 = P->A, axis 2 = I->S); see spinelab.utils.load_canonical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import AP_AXIS, LR_AXIS, SI_AXIS

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def angle_deg(v1, v2) -> float:
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-12:
        return 0.0
    cos = float(np.dot(v1, v2) / denom)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def column_si_extents(sag2d: np.ndarray) -> np.ndarray:
    """Superior-inferior extent (in voxels) of each anterior-posterior column."""
    out = np.zeros(sag2d.shape[0], dtype=int)
    for j in range(sag2d.shape[0]):
        si = np.flatnonzero(sag2d[j])
        if si.size >= 2:
            out[j] = int(si[-1] - si[0] + 1)
        elif si.size == 1:
            out[j] = 1
    return out


@dataclass
class BodyHeights:
    posterior_voxels: float
    anterior_voxels: float
    ap_span_voxels: int


def body_heights(body_mask: np.ndarray) -> BodyHeights | None:
    """Anterior and posterior height of a vertebral body, in voxels.

    Measured as the median SI extent over an edge-inset third at each end of the
    body's anterior-posterior span. The inset matters: the rounded corners of a
    vertebral body have a small SI extent, and including them is what produced
    posterior "heights" of 2-9 mm and absurd negative wedge angles in the
    original notebook (which additionally measured on the *whole* vertebra mask,
    so the posterior slab landed on the spinous process).

    Returns None when the body is too small to measure honestly.
    """
    if body_mask.ndim != 3 or not body_mask.any():
        return None
    sag = body_mask.any(axis=LR_AXIS)  # -> (AP, SI)
    ap_present = np.flatnonzero(sag.any(axis=1))
    if ap_present.size < 6:
        return None
    a0, a1 = int(ap_present[0]), int(ap_present[-1])
    span = a1 - a0 + 1
    heights = column_si_extents(sag)[a0:a1 + 1]
    inset = max(1, int(0.10 * span))
    third = max(2, int(0.30 * span))
    posterior = heights[inset:inset + third]
    anterior = heights[max(0, span - inset - third):max(0, span - inset)]
    posterior = posterior[posterior > 0]
    anterior = anterior[anterior > 0]
    if posterior.size == 0 or anterior.size == 0:
        return None
    return BodyHeights(float(np.median(posterior)), float(np.median(anterior)), span)


def wedge_angle_deg(posterior_mm: float, anterior_mm: float, ap_width_mm: float) -> float | None:
    """Anterior wedging angle. Positive = anterior shorter than posterior.

    That sign convention is the Scheuermann one: >= 5 deg over three or more
    contiguous vertebrae is the classic pattern.
    """
    if min(posterior_mm, anterior_mm, ap_width_mm) <= 0:
        return None
    return float(np.degrees(np.arctan2(posterior_mm - anterior_mm, ap_width_mm)))


def longest_run(values, predicate) -> int:
    """Longest run of consecutive items satisfying predicate."""
    best = run = 0
    for v in values:
        run = run + 1 if predicate(v) else 0
        best = max(best, run)
    return best


def curvature_metrics(centroids: dict[int, np.ndarray], labels) -> dict:
    """Angle between the upper and lower halves of a vertebral chain + scoliosis proxy.

    This is NOT a Cobb angle: a Cobb angle is measured between endplate
    tangents on a standing radiograph. It is a supine centroid-chain angle, which
    is why it is reported under its own name.
    """
    labels = sorted(l for l in labels if l in centroids)
    if len(labels) < 3:
        return {}
    first, last = labels[0], labels[-1]
    mid = labels[len(labels) // 2]
    top = centroids[mid] - centroids[first]
    bottom = centroids[last] - centroids[mid]
    coords = np.stack([centroids[l] for l in labels])
    p0, p1 = coords[0], coords[-1]
    axis = p1 - p0
    norm = np.linalg.norm(axis)
    devs = [0.0]
    if norm > 1e-9:
        axis = axis / norm
        devs = [float(np.linalg.norm(p - (p0 + np.dot(p - p0, axis) * axis))) for p in coords]
    return {
        "chain_angle_deg": angle_deg(top, bottom),
        "chain_angle_note": "supine centroid-chain angle, NOT a standing Cobb angle",
        "max_lateral_deviation_mm": float(max(devs)),
        "levels_used": [int(l) for l in labels],
    }


# --------------------------------------------------------------------------
# Intensity screening
# --------------------------------------------------------------------------


def modified_z(values: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin modified z-score (median/MAD based)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 0:
        # A zero MAD means a (near-)constant region: no outliers can be defined,
        # and dividing by an epsilon here is what turned single stray voxels into
        # z-scores of thousands in the old code (L1=5009 "outlier voxels").
        return np.zeros_like(values)
    return 0.6745 * (values - median) / mad


@dataclass
class RegionScreen:
    label: int
    name: str
    n_voxels: int
    median: float
    mad: float
    outlier_voxels: int
    outlier_fraction: float
    coverage: float
    excluded_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "label": int(self.label),
            "name": self.name,
            "n_voxels": int(self.n_voxels),
            "median_intensity": round(float(self.median), 3),
            "mad_intensity": round(float(self.mad), 3),
            "outlier_voxels": int(self.outlier_voxels),
            "outlier_fraction": round(float(self.outlier_fraction), 5),
            "in_fov_coverage": round(float(self.coverage), 3),
            "excluded_reason": self.excluded_reason,
        }


def screen_region(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    label: int,
    name: str,
    z_threshold: float = 3.5,
    min_voxels: int = 200,
    min_coverage: float = 0.6,
    reference_median: float | None = None,
) -> RegionScreen | None:
    """Robust bright-voxel screen inside one region.

    `coverage` is the fraction of mask voxels that carry signal. When a mask is
    resampled from another sequence's grid the out-of-field voxels become zero,
    and a region that is mostly out of field yields a meaningless distribution —
    those regions are reported as excluded rather than silently ranked.
    """
    mask = mask.astype(bool)
    n_mask = int(mask.sum())
    if n_mask < min_voxels:
        return None
    values = np.asarray(image)[mask].astype(float)
    inside = values[values > 0]
    coverage = float(inside.size) / float(n_mask)
    if coverage < min_coverage:
        return RegionScreen(label, name, n_mask, float("nan"), float("nan"), 0, 0.0,
                            coverage, "below_fov_coverage_threshold")
    median = float(np.median(inside))
    mad = float(np.median(np.abs(inside - median)))
    if reference_median is not None and median < reference_median:
        return RegionScreen(label, name, n_mask, median, mad, 0, 0.0, coverage,
                            "region_median_below_volume_reference")
    z = modified_z(inside)
    outliers = int((z > z_threshold).sum())
    return RegionScreen(label, name, n_mask, median, mad, outliers,
                        outliers / max(inside.size, 1), coverage)


def robust_threshold(reference_values: np.ndarray, k: float = 3.0) -> float | None:
    """median + k * sigma_MAD of a reference tissue.

    A threshold has to come from tissue that is *not* the tissue under test.
    Deriving it from the region being measured is self-referential and saturates:
    if one side of a paired region is uniformly bright, a high percentile of the
    pooled region lands inside that bright population and the measurement reports
    "no bright voxels on either side".
    """
    values = np.asarray(reference_values, dtype=float)
    values = values[values > 0]
    if values.size < 20:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = 1.4826 * mad  # MAD -> normal-consistent sigma
    if sigma <= 0:
        return None
    return median + k * sigma


def bright_fraction(image: np.ndarray, roi: np.ndarray, percentile: float = 95.0,
                    threshold: float | None = None) -> dict:
    """Fraction of ROI voxels above `threshold`, or above an in-ROI percentile.

    Computing the threshold on the whole volume (as the old cells did with
    ``np.percentile(img, 98)``) makes CSF, subcutaneous fat and vessels define
    the threshold, so the "oedema" map mostly highlights CSF. Prefer passing an
    explicit `threshold` from robust_threshold() on a reference tissue.
    """
    roi = roi.astype(bool)
    n_roi = int(roi.sum())
    if n_roi == 0:
        return {"roi_voxels": 0, "bright_voxels": 0, "bright_fraction": 0.0, "threshold": None}
    values = np.asarray(image)[roi].astype(float)
    inside = values[values > 0]
    if inside.size == 0:
        return {"roi_voxels": n_roi, "bright_voxels": 0, "bright_fraction": 0.0, "threshold": None}
    if threshold is None:
        threshold = float(np.percentile(inside, percentile))
    bright = int((values > threshold).sum())
    return {
        "roi_voxels": n_roi,
        "bright_voxels": bright,
        "bright_fraction": bright / n_roi,
        "threshold": float(threshold),
    }


def asymmetry_ratio(left: float, right: float) -> dict:
    """Symmetric comparison of two side measurements.

    Reports the ratio, the signed percentage difference relative to the mean, and
    which side is higher — with an explicit note when one side is zero, instead
    of dividing by 1e-9 and printing "inf x asymmetry".
    """
    left, right = float(left), float(right)
    total = left + right
    if total <= 0:
        return {"left": left, "right": right, "ratio": None, "diff_pct": None,
                "higher_side": None, "note": "both sides zero — nothing measured"}
    hi, lo = max(left, right), min(left, right)
    ratio = None if lo <= 0 else hi / lo
    return {
        "left": left,
        "right": right,
        "ratio": None if ratio is None else round(ratio, 3),
        "diff_pct": round(100.0 * (hi - lo) / (total / 2.0), 2),
        "higher_side": "right" if right > left else ("left" if left > right else "equal"),
        "note": None if ratio is not None else "one side is zero — ratio undefined",
    }


# --------------------------------------------------------------------------
# Side handling
# --------------------------------------------------------------------------


def facet_interface(semantic: np.ndarray, instance: np.ndarray, *, upper_label: int,
                    lower_label: int, inferior_process: int, superior_process: int,
                    dilate: int = 2) -> np.ndarray:
    """The region between the two bones that form one facet joint, on one side.

    A zygapophyseal joint at level N/N+1 is formed by the **inferior** articular
    process of the vertebra above and the **superior** articular process of the
    vertebra below. Dilating each and intersecting gives the interface between
    them — the joint space plus its immediate margins.

    That is the ROI worth measuring: the old code pooled every posterior-element
    label of every vertebra into one blob, which contains mostly bone and cannot
    show a joint effusion even in principle.
    """
    upper = (instance == upper_label) & (semantic == inferior_process)
    lower = (instance == lower_label) & (semantic == superior_process)
    if not upper.any() or not lower.any():
        return np.zeros_like(upper, dtype=bool)
    return binary_dilate(upper, dilate) & binary_dilate(lower, dilate)


def side_masks_from_labels(semantic: np.ndarray, left_labels, right_labels) -> tuple[np.ndarray, np.ndarray]:
    """Left/right masks taken from the segmentation's own side labels.

    SPINEPS already labels articular and costal processes per side
    (45/47 left, 46/48 right, 43 left, 44 right). Using those is exact. The old
    code instead split the array at ``shape[0] // 2``, which assumes the patient
    midline sits at the centre of the field of view and that axis 0 is L-R — on a
    non-canonicalised volume that can even mirror the sides, which for a
    right-sided pain case inverts the headline result.
    """
    left = np.isin(semantic, list(left_labels))
    right = np.isin(semantic, list(right_labels))
    return left, right


# --------------------------------------------------------------------------
# Morphology helpers (numpy only — scipy is deliberately not a dependency)
# --------------------------------------------------------------------------


def _shift(mask: np.ndarray, axis: int, offset: int, fill: bool = False) -> np.ndarray:
    """Shift a boolean array along one axis, filling the vacated edge.

    Written with slices rather than np.roll: roll wraps around, which for
    morphology means the top of the volume grows into the bottom.
    """
    out = np.full_like(mask, fill, dtype=bool)
    n = mask.shape[axis]
    if abs(offset) >= n:
        return out
    dst = [slice(None)] * mask.ndim
    src = [slice(None)] * mask.ndim
    if offset > 0:
        dst[axis] = slice(offset, n)
        src[axis] = slice(0, n - offset)
    elif offset < 0:
        dst[axis] = slice(0, n + offset)
        src[axis] = slice(-offset, n)
    else:
        return mask.astype(bool).copy()
    out[tuple(dst)] = mask[tuple(src)]
    return out


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """6-connected binary dilation.

    scipy.ndimage would do this, but scipy has no wheel that installs reliably on
    the operator's connection, and this is the only morphology the pipeline needs.
    """
    out = mask.astype(bool)
    for _ in range(max(0, int(iterations))):
        grown = out.copy()
        for axis in range(out.ndim):
            grown |= _shift(out, axis, 1, fill=False)
            grown |= _shift(out, axis, -1, fill=False)
        out = grown
    return out


def binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """6-connected binary erosion.

    Voxels outside the volume count as foreground, so eroding does not eat the
    volume's own faces — otherwise a body that touches the edge of the field of
    view would lose its rim to the border rather than to anatomy.
    """
    out = mask.astype(bool)
    for _ in range(max(0, int(iterations))):
        shrunk = out.copy()
        for axis in range(out.ndim):
            shrunk &= _shift(out, axis, 1, fill=True)
            shrunk &= _shift(out, axis, -1, fill=True)
        out = shrunk
    return out


#: Semantic labels that swap meaning when the volume is mirrored left<->right.
MIRROR_LABEL_PAIRS = ((43, 44), (45, 46), (47, 48), (63, 64))


def mirror_side_labels(semantic: np.ndarray) -> np.ndarray:
    """Swap left/right label ids (43<->44, 45<->46, 47<->48, 63<->64).

    Needed for mirror test-time augmentation: run the segmenter on an L-R
    flipped volume, flip the result back, and the anatomy lines up but every
    side-specific label now names the wrong side. Swapping them makes the two
    runs comparable — and their agreement is a direct check of whether the
    model's *side* assignment is stable, which is the one thing this case
    depends on.
    """
    out = np.asarray(semantic).copy()
    for a, b in MIRROR_LABEL_PAIRS:
        mask_a = semantic == a
        mask_b = semantic == b
        out[mask_a] = b
        out[mask_b] = a
    return out


def label_agreement(a: np.ndarray, b: np.ndarray, labels) -> dict:
    """Per-label Dice between two label volumes, plus the weakest label.

    Used for mirror-TTA self-consistency and for cross-model checks. Labels
    absent from both volumes are reported as None rather than 0, so "not present"
    is not confused with "disagrees".
    """
    out: dict[str, float | None] = {}
    for label in labels:
        mask_a, mask_b = a == label, b == label
        if not mask_a.any() and not mask_b.any():
            out[str(int(label))] = None
            continue
        out[str(int(label))] = round(dice(mask_a, mask_b), 4)
    scored = {k: v for k, v in out.items() if v is not None}
    return {
        "per_label_dice": out,
        "mean_dice": round(float(np.mean(list(scored.values()))), 4) if scored else None,
        "min_dice": round(float(min(scored.values())), 4) if scored else None,
        "weakest_label": (min(scored, key=scored.get) if scored else None),
    }


def midline_index(reference_mask: np.ndarray) -> float:
    """Patient midline along the L-R axis, from a midline structure's centroid.

    Fallback for tools that do not label sides. Still better than the array
    centre because it follows the anatomy.
    """
    mask = reference_mask.astype(bool)
    if not mask.any():
        return (reference_mask.shape[LR_AXIS] - 1) / 2.0
    coords = np.flatnonzero(mask.any(axis=(AP_AXIS, SI_AXIS)))
    weights = np.array([mask[i].sum() for i in coords], dtype=float)
    return float((coords * weights).sum() / weights.sum())


def split_by_midline(mask: np.ndarray, midline: float) -> tuple[np.ndarray, np.ndarray]:
    """Split a mask into (left, right) about a midline index on the L-R axis.

    In RAS+, increasing index along axis 0 goes towards the patient's RIGHT, so
    indices below the midline are the patient's LEFT.
    """
    left = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)
    cut = int(round(midline))
    left[:cut] = mask[:cut]
    right[cut:] = mask[cut:]
    return left, right


# --------------------------------------------------------------------------
# Canal
# --------------------------------------------------------------------------


def canal_area_profile(canal_mask: np.ndarray, voxel_area_mm2: float,
                       min_area_mm2: float = 20.0) -> dict:
    """Cross-sectional area of the canal per axial slice, plus a narrowing metric.

    Slices at the ends of the segmentation are dropped (a partially covered top
    or bottom slice has a small area for purely geometric reasons and used to be
    reported as the "narrowest point", over-calling stenosis).
    """
    mask = canal_mask.astype(bool)
    areas = np.array([float(mask[:, :, z].sum()) * float(voxel_area_mm2)
                      for z in range(mask.shape[SI_AXIS])])
    valid_idx = np.flatnonzero(areas > min_area_mm2)
    if valid_idx.size < 5:
        return {"n_slices_measured": int(valid_idx.size),
                "note": "too few valid slices to describe a profile"}
    # Trim one slice at each end of the contiguous measured block.
    lo, hi = int(valid_idx[0]) + 1, int(valid_idx[-1]) - 1
    core = areas[lo:hi + 1]
    core = core[core > min_area_mm2]
    if core.size < 3:
        return {"n_slices_measured": int(core.size),
                "note": "too few valid slices after edge trimming"}
    median = float(np.median(core))
    minimum = float(np.min(core))
    return {
        "n_slices_measured": int(core.size),
        "median_area_mm2": round(median, 2),
        "min_area_mm2": round(minimum, 2),
        "p10_area_mm2": round(float(np.percentile(core, 10)), 2),
        "max_narrowing_pct": round(100.0 * (1.0 - minimum / median), 2) if median > 0 else None,
        "slice_index_of_min": int(lo + int(np.argmin(areas[lo:hi + 1]))),
    }


# --------------------------------------------------------------------------
# Spinal Cord Toolbox interop
# --------------------------------------------------------------------------

#: Spinal Cord Toolbox numbers a disc by the vertebra *below* it, counting C1=1 …
#: C7=7, T1=8 … T12=19, L1=20 … L5=24, S1=25. So C3/C4 is 4 and C6/C7 is 7 — the
#: two anchors stated in the sct_detect_compression documentation, which is the
#: only place the convention is pinned down for the levels this project needs.
_SCT_VERTEBRA_INDEX = {
    **{f"C{i}": i for i in range(1, 8)},
    **{f"T{i}": 7 + i for i in range(1, 13)},
    **{f"L{i}": 19 + i for i in range(1, 6)},
    "S": 25, "S1": 25,
}

#: sct_detect_compression thresholds (Horáková 2022, logistic model on cervical
#: canal/cord morphometry; AUC 0.947 on an independent DCM cohort). Both bounds are
#: inclusive on the "possible" side, as documented.
COMPRESSION_P_LOW = 0.345
COMPRESSION_P_HIGH = 0.451


def sct_level_index(level_name) -> int | None:
    """'T7' -> 14, counting C1=1. None for anything unrecognised, never a guess.

    The one place a level name becomes a number, so that a reference cohort in one
    label space and this study in another are compared level for level rather than
    label for label.
    """
    if not level_name:
        return None
    return _SCT_VERTEBRA_INDEX.get(str(level_name).strip().upper())


def sct_level_name(index) -> str | None:
    """14 -> 'T7'. The inverse of sct_level_index, for reporting."""
    try:
        index = int(index)
    except (TypeError, ValueError):
        return None
    for name, value in _SCT_VERTEBRA_INDEX.items():
        if value == index and name != "S1":
            return name
    return None


def sct_disc_value(level_name) -> int | None:
    """SCT label value for a disc named like "C3-C4", or None if unrecognised.

    Returning None rather than a fallback is deliberate: a disc file with a wrong
    number produces measurements at the wrong level that look completely normal.
    """
    if not level_name:
        return None
    parts = str(level_name).upper().replace("/", "-").split("-")
    if len(parts) != 2:
        return None
    return _SCT_VERTEBRA_INDEX.get(parts[1].strip())


def disc_labels_for_sct(disc_volume: np.ndarray, level_map: dict,
                        keep_values=None) -> np.ndarray:
    """One single voxel per disc, at its posterior edge, numbered the SCT way.

    `sct_detect_compression -discfile` wants exactly one voxel per level, placed
    "at the posterior edge of the intervertebral disc". In canonical RAS the
    posterior edge is the lowest index on the anterior-posterior axis; among the
    voxels there, the one closest to the centre of the disc in the other two axes
    is chosen, so the point is reproducible rather than whichever voxel numpy
    happened to list first.

    `keep_values` optionally restricts the output to a set of SCT values — the
    compression model only covers 4…7 and a disc file carrying more levels than a
    tool supports is a source of silent mismatches.
    """
    data = np.asarray(disc_volume)
    out = np.zeros(data.shape, dtype=np.int16)
    wanted = None if keep_values is None else {int(v) for v in keep_values}

    for label in np.unique(data):
        if label == 0:
            continue
        value = sct_disc_value(level_map.get(int(label)))
        if value is None or (wanted is not None and value not in wanted):
            continue
        coords = np.array(np.nonzero(data == label))
        if coords.size == 0:
            continue
        posterior = coords[AP_AXIS].min()
        on_edge = coords[:, coords[AP_AXIS] == posterior]
        centre = np.median(coords, axis=1)
        distance = (np.abs(on_edge[LR_AXIS] - centre[LR_AXIS])
                    + np.abs(on_edge[SI_AXIS] - centre[SI_AXIS]))
        pick = on_edge[:, int(np.argmin(distance))]
        out[tuple(int(v) for v in pick)] = value
    return out


def classify_compression(probability) -> str | None:
    """"no" / "possible" / "yes" for a compression probability from SCT."""
    if probability is None:
        return None
    p = float(probability)
    if p < COMPRESSION_P_LOW:
        return "no"
    if p <= COMPRESSION_P_HIGH:
        return "possible"
    return "yes"


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * int((a & b).sum()) / denom)


# --------------------------------------------------------------------------
# Level morphometry — measured identically on any whole-vertebra label volume
# --------------------------------------------------------------------------
#
# These exist so that a reference cohort and this subject can be measured by the
# *same* code from the *same* kind of mask. `stages/geometry.py` measures heights
# and wedge angles on the vertebral **body** (the corpus subregion SPINEPS
# produces), and that is the right way to measure them — but a whole-spine label
# volume has one label per vertebra, posterior elements included, so a wedge angle
# from it lands partly on the spinous process. Comparing the two would be a
# silent mismatch of definitions. Nothing here needs a corpus label, and nothing
# here is a height or a wedge angle.


def label_table(volume: np.ndarray) -> dict[int, np.ndarray]:
    """Voxel coordinates of every non-zero label, as {label: (n, 3) array}.

    One pass over the volume, then all per-label work happens on the coordinate
    lists. The obvious alternative — `np.argwhere(volume == label)` in a loop —
    rescans the whole array once per label, which on a whole-spine volume with
    forty labels is forty passes over eight million voxels and took a hundred
    seconds per subject where this takes about one.
    """
    mask = volume != 0
    if not mask.any():
        return {}
    coords = np.argwhere(mask)
    values = volume[mask]
    order = np.argsort(values, kind="stable")
    values, coords = values[order], coords[order]
    labels, starts = np.unique(values, return_index=True)
    ends = list(starts[1:]) + [len(values)]
    return {int(label): coords[start:end]
            for label, start, end in zip(labels, starts, ends)}


def centroids_si(volume: np.ndarray) -> dict[int, float]:
    """Mean superior-inferior index of every non-zero label."""
    return {label: float(coords[:, SI_AXIS].mean())
            for label, coords in label_table(volume).items()}


def slice_counts(mask: np.ndarray) -> np.ndarray:
    """Voxels per superior-inferior slice. Computed once, sliced many times."""
    mask = mask.astype(bool)
    axes = tuple(axis for axis in range(mask.ndim) if axis != SI_AXIS)
    return mask.sum(axis=axes)


#: How close a label's centroid has to be to a disc anchor, as a fraction of the
#: local disc-to-disc spacing, before that label *is* that disc. Well inside the
#: gap: a matching disc lands within a millimetre or two, the nearest vertebra
#: roughly half a spacing away.
DISC_ANCHOR_TOLERANCE = 0.3


def levels_from_disc_anchors(label_si: dict[int, float], anchors: dict[int, float],
                             tolerance: float = DISC_ANCHOR_TOLERANCE,
                             ) -> tuple[dict[int, int], dict[int, int], str]:
    """Give every label in a whole-spine volume its anatomical level.

    `anchors` are disc positions already carrying the level numbering this project
    uses everywhere (a disc is named by the vertebra below it, C1=1 … T1=8 …
    L1=20) — in the reference cohort they are the manually placed disc points.
    Returns `(vertebra_map, disc_map, note)`, both mapping label value to level.

    Done this way, rather than by reading the label values, because the numbering
    inside a whole-spine label volume is dataset-specific and documented nowhere.
    An offset table copied from one dataset to another shifts every level by a
    constant, and every number downstream still looks perfectly reasonable.

    Sizes are not used either: C1 is about 5 cm3 and an L4/L5 disc about 12 cm3,
    so "vertebrae are the big ones" misfiles the entire cervical spine.
    """
    if len(anchors) < 2 or not label_si:
        return {}, {}, "fewer than two disc anchors, or no labels"
    keys = sorted(anchors)
    z = [anchors[k] for k in keys]
    if any(z[i] <= z[i + 1] for i in range(len(z) - 1)):
        return {}, {}, "disc anchors are not ordered head to foot"
    spacing = float(np.median([z[i] - z[i + 1] for i in range(len(z) - 1)]))
    if spacing <= 0:
        return {}, {}, "disc anchors have no spacing"

    # A label sitting on an anchor is that disc. Resolved before anything else,
    # because a disc's centroid lies exactly on the boundary between the two
    # vertebrae either side of it and would otherwise be counted as one of them.
    disc_map: dict[int, int] = {}
    for k in keys:
        nearest, distance = None, None
        for label, zz in label_si.items():
            d = abs(zz - anchors[k])
            if distance is None or d < distance:
                nearest, distance = label, d
        if nearest is not None and distance <= tolerance * spacing:
            disc_map[nearest] = k

    vertebra_map: dict[int, int] = {}
    for k, k_next in zip(keys, keys[1:]):
        if k_next != k + 1:
            continue                       # a gap in the anchors, not a level
        inside = [label for label, zz in label_si.items()
                  if anchors[k_next] < zz < anchors[k] and label not in disc_map]
        if len(inside) == 1 and 1 <= k <= 25:
            vertebra_map[inside[0]] = k
    if not vertebra_map:
        return {}, {}, ("no level has exactly one label between two consecutive disc "
                        "anchors")
    return vertebra_map, disc_map, (
        f"{len(vertebra_map)} vertebrae and {len(disc_map)} discs placed from "
        f"{len(anchors)} disc anchors, spacing {spacing:.1f} voxels")


def mean_thickness_mm(coords: np.ndarray, si_mm: float, ap_size: int) -> float | None:
    """Volume divided by axial footprint — a disc's mean height, in millimetres.

    Robust where a single mid-sagittal profile is not: a disc is wedge-shaped and
    its height depends on where the profile is taken, whereas volume/footprint
    uses every voxel and needs no plane to be chosen.

    Takes a coordinate list from `label_table` rather than a mask, so that a level
    is never re-extracted from the full volume.
    """
    coords = np.asarray(coords)
    if coords.size == 0:
        return None
    # One integer key per (left-right, anterior-posterior) column. The multiplier
    # has to be the size of the *second* axis, or two different columns collide and
    # the footprint comes out too small — which inflates the thickness.
    columns = coords[:, LR_AXIS].astype(np.int64) * int(ap_size) + coords[:, AP_AXIS]
    footprint = int(np.unique(columns).size)
    if footprint == 0:
        return None
    return float(len(coords)) / float(footprint) * float(si_mm)


def slab_area_mm2(counts: np.ndarray, z_low: int, z_high: int,
                  voxel_area_mm2: float) -> float | None:
    """Median cross-sectional area over an inclusive slice range of `slice_counts`."""
    counts = np.asarray(counts, dtype=float)
    z_low, z_high = int(max(z_low, 0)), int(min(z_high, counts.size - 1))
    if z_high < z_low:
        return None
    areas = counts[z_low:z_high + 1]
    areas = areas[areas > 0]
    if areas.size == 0:
        return None
    return round(float(np.median(areas)) * float(voxel_area_mm2), 3)


def segmental_angles_deg(centroids: dict[int, np.ndarray]) -> dict[int, float]:
    """Angle at each vertebra between the segment above it and the segment below.

    Keyed by the middle vertebra's label. Zero means the three centroids are
    collinear; the sign says which way the chain bends, positive for an apex
    pointing posteriorly (the sense of a kyphosis) and negative for one pointing
    anteriorly (a lordosis). Reported as a deviation from straight rather than as
    the 175-degree included angle, so that "more curved" reads as a bigger number.

    Measured in the sagittal (AP/SI) plane on centroids of whole vertebrae, so it
    is comparable between any two label volumes of the same kind — and it is not a
    Cobb angle, which is measured between endplate tangents on a standing
    radiograph, in a standing patient.
    """
    order = sorted(centroids, key=lambda label: -float(centroids[label][SI_AXIS]))
    out: dict[int, float] = {}
    for above, middle, below in zip(order, order[1:], order[2:]):
        top, mid, bot = (np.asarray(centroids[k], dtype=float) for k in (above, middle, below))
        upper = np.array([top[AP_AXIS] - mid[AP_AXIS], top[SI_AXIS] - mid[SI_AXIS]])
        lower = np.array([mid[AP_AXIS] - bot[AP_AXIS], mid[SI_AXIS] - bot[SI_AXIS]])
        # Cross product of (anterior, superior) vectors: negative when the middle
        # vertebra lies posterior to the chord between its neighbours.
        cross = float(lower[0] * upper[1] - lower[1] * upper[0])
        out[middle] = round(angle_deg(upper, lower) * (-1.0 if cross > 0 else 1.0), 3)
    return out


# --------------------------------------------------------------------------
# Reference distributions
# --------------------------------------------------------------------------

#: Below this many subjects a level's distribution is not summarised at all.
#: Percentiles of a handful of values are noise dressed as a reference, and the
#: whole point of a reference cohort is that it is not that.
MIN_REFERENCE_N = 10


def summarise_distribution(values, min_n: int = MIN_REFERENCE_N) -> dict | None:
    """Median, IQR and 5th/95th percentiles of a reference sample.

    Non-finite entries are dropped rather than propagated: a cohort volume where
    one level could not be measured must reduce n for that level and nothing else.
    Returns None when too few subjects remain — the caller reports the absence,
    which is why this never falls back to a smaller summary.
    """
    array = np.asarray([v for v in np.ravel(np.asarray(values, dtype=float))
                        if np.isfinite(v)], dtype=float)
    if array.size < max(int(min_n), 1):
        return None
    q5, q25, q50, q75, q95 = (float(x) for x in np.percentile(array, [5, 25, 50, 75, 95]))
    return {
        "n": int(array.size),
        "median": round(q50, 3),
        "iqr": [round(q25, 3), round(q75, 3)],
        "p5_p95": [round(q5, 3), round(q95, 3)],
        "min": round(float(array.min()), 3),
        "max": round(float(array.max()), 3),
    }


def percentile_of(value, distribution) -> float | None:
    """Where `value` falls inside a reference sample, as a percentage.

    Midrank convention: ties count as half, so a value equal to every member of a
    uniform sample lands at 50 and not at 0 or 100. Deliberately returns a
    *position*, never a verdict — no cut-off is applied here or by any caller,
    because this distribution is computed by this project rather than published
    with one (see docs/research/README.md).
    """
    if value is None or not np.isfinite(float(value)):
        return None
    sample = np.asarray([v for v in np.ravel(np.asarray(distribution, dtype=float))
                         if np.isfinite(v)], dtype=float)
    if sample.size == 0:
        return None
    below = float((sample < float(value)).sum())
    equal = float((sample == float(value)).sum())
    return round(100.0 * (below + 0.5 * equal) / sample.size, 1)
