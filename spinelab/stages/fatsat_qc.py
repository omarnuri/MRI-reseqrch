"""Stage `fatsat_qc`: did the fat suppression actually work?

Everything the pipeline says about oedema rests on one assumption: that the
"fat-suppressed" series really suppresses fat. That is testable, and it must be
tested against **tissue**, not against geometry.

The first version of this stage compared a geometric shell just inside the body
surface against an eroded core. On the real study that produced rim = 2 964 573
voxels against core = 6 757 — the "rim" had swallowed the volume, because twelve
erosions annihilate a 23-slice coronal slab through-plane. Both series then
returned a ratio of 1.18, which reflected the shape of the mask and said nothing
about fat, yet it was reported as "NO evidence of fat suppression" and would have
voided the whole oedema branch.

So the test now uses TotalSegmentator's own tissue labels: subcutaneous fat against
skeletal muscle, in the same series. On a plain T2 fat is clearly brighter than
muscle; if suppression works, that ratio collapses. When those masks are not
available the stage reports "not assessed" — and the stages that depend on it treat
an unverified suppression as unverified, not as failed.
"""

from __future__ import annotations

import numpy as np

from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..runlog import event, get_logger
from ..utils import load_canonical, resample_mask_to, write_json

log = get_logger(__name__)

#: TotalSegmentator MR label names used as tissue references.
FAT_LABELS = ("subcutaneous_fat", "torso_fat")
MUSCLE_LABELS = ("skeletal_muscle", "autochthon_left", "autochthon_right")

#: Fat/muscle signal ratio below which fat looks suppressed.
SUPPRESSED_MAX_RATIO = 1.15
#: Ratio above which fat is clearly not suppressed (as on a plain T2).
UNSUPPRESSED_MIN_RATIO = 1.4


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    picks = ctx.stage_data("ingest").get("picks", {})
    fatsat = picks.get("FATSAT_BEST")
    if not fatsat:
        raise SkipStage("no fat-suppressed series to check")

    ts = ctx.stage_data("totalsegmentator")
    outputs = [str(p) for p in (ts.get("outputs") or [])]
    fat_masks = _find(outputs, FAT_LABELS)
    muscle_masks = _find(outputs, MUSCLE_LABELS)
    if not fat_masks or not muscle_masks:
        raise SkipStage(
            "fat suppression cannot be verified: TotalSegmentator tissue masks "
            f"(fat: {FAT_LABELS[0]}, muscle: {MUSCLE_LABELS[0]}) are not available. "
            "Downstream stages will report the suppression as unverified rather than "
            "assumed — a geometric proxy was tried and measured shape, not tissue.")

    control = picks.get("T2_SAG") or picks.get("T2_AX")
    measured = _fat_muscle_ratio(fatsat, fat_masks, muscle_masks)
    if measured is None:
        raise SkipStage("tissue masks do not overlap the fat-suppressed series' field of view")
    control_measured = _fat_muscle_ratio(control, fat_masks, muscle_masks) if control else None

    verdict, effective = _verdict(measured["fat_to_muscle_ratio"],
                                 (control_measured or {}).get("fat_to_muscle_ratio"))
    log.info("fat/muscle ratio: %.3f on %s, %s on the control series — %s",
             measured["fat_to_muscle_ratio"], picks.get("FATSAT_LABEL"),
             (control_measured or {}).get("fat_to_muscle_ratio"), verdict)
    event("fatsat_qc", ratio=measured["fat_to_muscle_ratio"],
          control_ratio=(control_measured or {}).get("fat_to_muscle_ratio"),
          effective=effective)

    payload = {
        "fatsat_image": fatsat,
        "fatsat_label": picks.get("FATSAT_LABEL"),
        "fatsat_plane": picks.get("FATSAT_PLANE"),
        "method": "median signal of subcutaneous fat against skeletal muscle, "
                  "TotalSegmentator masks, within one series",
        "fatsat_stats": measured,
        "control_image": control,
        "control_stats": control_measured,
        "suppressed_max_ratio": SUPPRESSED_MAX_RATIO,
        "unsuppressed_min_ratio": UNSUPPRESSED_MIN_RATIO,
        "suppression_effective": effective,
        "verdict": verdict,
        "interpretation_limits": [
            "Both tissues are read on the same image, so scanner scaling cancels; the "
            "comparison against the plain T2 additionally shows the direction of the "
            "change rather than an absolute level.",
            "The masks come from the sagittal T2 and are resampled onto each series, so "
            "only the overlapping field of view is measured; the voxel counts say how "
            "much that was.",
            "A negative result is the strong one: if fat does not darken relative to "
            "muscle, bright signal on this series cannot be read as fluid or oedema.",
        ],
    }
    write_json(cfg.intermediate_dir / "fatsat_qc.json", payload)
    return StageResult(
        name="fatsat_qc",
        status=Status.OK if effective else Status.PARTIAL,
        evidence=Evidence.MEASUREMENT,
        reason=None if effective else verdict,
        data=payload,
    )


def _find(paths: list[str], names: tuple[str, ...]) -> list[str]:
    out = []
    for name in names:
        out += [p for p in paths if name in p.replace("\\", "/").rsplit("/", 1)[-1].lower()]
    return out


def _fat_muscle_ratio(image_path: str | None, fat_masks, muscle_masks) -> dict | None:
    """Median fat signal over median muscle signal, on one image."""
    if not image_path:
        return None
    img = load_canonical(image_path)
    data = np.asarray(img.get_fdata(), dtype=float)

    fat = _union(fat_masks, img)
    muscle = _union(muscle_masks, img)
    fat_values = data[fat & (data > 0)]
    muscle_values = data[muscle & (data > 0)]
    if fat_values.size < 200 or muscle_values.size < 200:
        return None
    fat_median = float(np.median(fat_values))
    muscle_median = float(np.median(muscle_values))
    if muscle_median <= 0:
        return None
    return {
        "fat_voxels": int(fat_values.size),
        "muscle_voxels": int(muscle_values.size),
        "fat_median": round(fat_median, 3),
        "muscle_median": round(muscle_median, 3),
        "fat_to_muscle_ratio": round(fat_median / muscle_median, 3),
    }


def _union(mask_paths, reference) -> np.ndarray:
    total = None
    for path in mask_paths:
        mask = np.asarray(resample_mask_to(load_canonical(path), reference).get_fdata()) > 0.5
        total = mask if total is None else (total | mask)
    return total if total is not None else np.zeros(reference.shape, dtype=bool)


def _verdict(ratio: float, control_ratio: float | None) -> tuple[str, bool]:
    if control_ratio is None:
        if ratio <= SUPPRESSED_MAX_RATIO:
            return (f"fat is not brighter than muscle (ratio {ratio}) — consistent with "
                    f"working fat suppression, though no non-suppressed series was "
                    f"available for comparison"), True
        return (f"fat is {ratio}x muscle and there is no control series to compare "
                f"against — suppression is NOT confirmed"), False

    if ratio <= SUPPRESSED_MAX_RATIO and control_ratio > ratio:
        return (f"fat suppression works: fat/muscle is {ratio} on the suppressed series "
                f"against {control_ratio} on the plain T2"), True
    if ratio < control_ratio:
        return (f"partial suppression: fat/muscle drops from {control_ratio} to {ratio}, "
                f"but stays above {SUPPRESSED_MAX_RATIO}. Bright signal may still be fat — "
                f"treat oedema readings as unconfirmed"), False
    return (f"NO evidence of fat suppression: fat/muscle is {ratio} against "
            f"{control_ratio} on the plain T2. Bright signal on this series must not be "
            f"read as fluid or oedema"), False
