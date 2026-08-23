"""Stage `muscles`: paraspinal muscle volume and left/right asymmetry.

Kept: cross-sectional-area / volume asymmetry from the TotalSegmentator masks.
That is a real measurement.

Dropped: the "fatty fraction". The old cells thresholded normalised T2 at 0.6 and
then, when that returned zeros, switched to "above the 75th percentile of T1
inside the muscle" — a quantity that is 25% by construction for any muscle and
carries no information about fat. Those numbers then drove printed verdicts
("превосходное качество мышечной ткани, типичное для профессиональных
спортсменов"). Fat fraction needs a Dixon/2-point acquisition or MR
spectroscopy; without one this pipeline reports no fat measure at all, and says
why.

Also fixed: volumes are computed with the voxel size of the mask's own grid. The
old code multiplied TotalSegmentator voxel counts by the *sagittal T2* voxel
volume, which mixes two grids and produces a wrong number in cm^3.
"""

from __future__ import annotations


import numpy as np

from ..analysis import asymmetry_ratio
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import load_canonical, voxel_volume_mm3, write_json


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    ts = ctx.stage_data("totalsegmentator")
    pairs = ts.get("muscle_pairs") or {}
    if not pairs:
        raise SkipStage("no left/right paraspinal muscle masks from TotalSegmentator")

    results = {}
    for base, paths in sorted(pairs.items()):
        left_img = load_canonical(paths["left"])
        right_img = load_canonical(paths["right"])
        left = np.asarray(left_img.get_fdata()) > 0.5
        right = np.asarray(right_img.get_fdata()) > 0.5
        vox_mm3 = voxel_volume_mm3(left_img)
        n_left, n_right = int(left.sum()), int(right.sum())
        if n_left == 0 and n_right == 0:
            continue
        vol_left = n_left * vox_mm3 / 1000.0
        vol_right = n_right * vox_mm3 / 1000.0
        results[base] = {
            "voxels_left": n_left,
            "voxels_right": n_right,
            "volume_left_cm3": round(vol_left, 2),
            "volume_right_cm3": round(vol_right, 2),
            "voxel_volume_mm3": round(vox_mm3, 4),
            "comparison": asymmetry_ratio(vol_left, vol_right),
            "exceeds_threshold": bool(
                (asymmetry_ratio(vol_left, vol_right)["diff_pct"] or 0)
                > cfg.muscle_asymmetry_pct_threshold),
        }

    if not results:
        raise SkipStage("muscle masks are empty")

    total_cm3 = sum(r["volume_left_cm3"] + r["volume_right_cm3"] for r in results.values())
    payload = {
        "muscles": results,
        "total_paraspinal_volume_cm3": round(total_cm3, 2),
        "asymmetry_threshold_pct": cfg.muscle_asymmetry_pct_threshold,
        "fat_infiltration": {
            "measured": False,
            "reason": ("fat fraction requires a Dixon / 2-point or spectroscopic "
                       "acquisition; this study has none. An intensity threshold on T1 or "
                       "T2 is not a fat fraction and is deliberately not reported."),
        },
        "interpretation_limits": [
            "Volume within the segmented field of view only — a sagittal thoracic study "
            "does not cover the whole muscle, so absolute volume is not comparable to "
            "published norms; the left/right ratio inside one study is the usable part.",
            "Side asymmetry has many benign causes (handedness, sport, positioning) and by "
            "itself indicates nothing about the source of pain.",
            "The reporting threshold is this project's own choice, not a clinical cut-off: "
            "no published normative distribution of paraspinal asymmetry applies to this "
            "study's field of view.",
        ],
    }
    write_json(cfg.intermediate_dir / "muscles.json", payload)
    return StageResult(name="muscles", status=Status.OK, evidence=Evidence.MEASUREMENT,
                       data=payload)
