"""Stage `canal`: spinal canal cross-sectional area profile.

Replaces two contradictory cells that both wrote ``canal_stenosis.json``: one
canonicalised the volume and binarised at 0.5, the other did neither, so the
result depended on which cell was executed last. The area profile now also trims
the end slices, where a partially covered slice is narrow for purely geometric
reasons and used to be reported as "the narrowest point".
"""

from __future__ import annotations

import numpy as np

from ..analysis import canal_area_profile
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import LR_AXIS, AP_AXIS, load_canonical, write_json
from ..utils import find_outputs


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    tss = ctx.stage_data("totalspineseg")
    root = tss.get("output_dir")
    if not root:
        raise SkipStage("TotalSpineSeg did not run — no canal mask")

    candidates = [p for p in find_outputs(root, "*.nii.gz") if "canal" in str(p).lower()]
    if not candidates:
        raise SkipStage("no canal segmentation in the TotalSpineSeg output")

    img = load_canonical(candidates[0])
    data = np.asarray(img.get_fdata())
    mask = data > 0.5
    if not mask.any():
        raise SkipStage("canal mask is empty")

    zooms = img.header.get_zooms()
    voxel_area = float(zooms[LR_AXIS]) * float(zooms[AP_AXIS])
    profile = canal_area_profile(mask, voxel_area)

    narrowing = profile.get("max_narrowing_pct")
    flagged = bool(narrowing is not None and narrowing >= cfg.canal_stenosis_pct)
    payload = {
        "mask_file": str(candidates[0]),
        "voxel_area_mm2": round(voxel_area, 4),
        **profile,
        "narrowing_threshold_pct": cfg.canal_stenosis_pct,
        "exceeds_threshold": flagged,
        "interpretation_limits": [
            "Area is measured on the segmentation's own axial planes, relative to that "
            "study's own median. It is a shape descriptor, not a diagnosis of stenosis: "
            "clinical stenosis is defined on absolute area plus cord signal plus symptoms.",
            "A single narrow slice is often a segmentation artefact at a disc level; the "
            "10th-percentile area is the more stable number.",
        ],
    }
    write_json(cfg.intermediate_dir / "canal.json", payload)
    status = Status.OK if profile.get("median_area_mm2") else Status.PARTIAL
    return StageResult(name="canal", status=status, evidence=Evidence.MEASUREMENT,
                       reason=profile.get("note"), data=payload)
