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


def rim_to_core_ratio(image: np.ndarray, *, rim_iterations: int = 3,
                      core_iterations: int = 12, body_threshold_percentile: float = 99.0,
                      body_fraction: float = 0.10) -> dict | None:
    """Signal of the subcutaneous band relative to the body core.

    A fat-suppression check that needs no extra sequence and no segmentation:
    subcutaneous fat forms a band just inside the skin. On a sequence that
    suppresses fat that band is dark relative to the deep soft tissue; on a plain
    T2 it is much brighter. Running this on both series of the same study and
    comparing the two ratios says whether the fat suppression actually worked —
    which decides whether any oedema reading is meaningful at all.

    Returns None when the volume is too small or no body could be found.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 3 or image.size < 1000:
        return None
    positive = image[image > 0]
    if positive.size < 100:
        return None
    reference = float(np.percentile(positive, body_threshold_percentile))
    body = image > (body_fraction * reference)
    if body.sum() < 500:
        return None
    core = binary_erode(body, core_iterations)
    if core.sum() < 100:
        # Thin slab (a coronal STIR can be only a few slices thick): erode less.
        core = binary_erode(body, max(1, core_iterations // 4))
        if core.sum() < 100:
            return None
    rim = body & ~binary_erode(body, rim_iterations)
    if rim.sum() < 100:
        return None
    rim_p90 = float(np.percentile(image[rim], 90))
    core_median = float(np.median(image[core]))
    if core_median <= 0:
        return None
    return {
        "rim_voxels": int(rim.sum()),
        "core_voxels": int(core.sum()),
        "rim_p90": round(rim_p90, 3),
        "core_median": round(core_median, 3),
        "rim_to_core_ratio": round(rim_p90 / core_median, 3),
    }


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
# Agreement
# --------------------------------------------------------------------------


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * int((a & b).sum()) / denom)
