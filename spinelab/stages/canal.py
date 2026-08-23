"""Stage `canal`: spinal canal cross-sectional area profile.

Replaces two contradictory cells that both wrote ``canal_stenosis.json``: one
canonicalised the volume and binarised at 0.5, the other did neither, so the
result depended on which cell was executed last. The area profile now also trims
the end slices, where a partially covered slice is narrow for purely geometric
reasons and used to be reported as "the narrowest point".
"""

from __future__ import annotations

import numpy as np

from .. import labels as L
from ..analysis import canal_area_profile
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import AP_AXIS, LR_AXIS, load_canonical, write_json


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    tss = ctx.stage_data("totalspineseg")
    # Read the canal out of the verified label volume, by label id. Matching "canal"
    # in the path picked `step1_canal`, which is a soft map — 8999 distinct values on
    # the real study — so this measured a thresholded probability, not a segmentation.
    label_volume = tss.get("label_volume")
    if not label_volume:
        raise SkipStage("TotalSpineSeg produced no verified label volume — "
                        f"{tss.get('label_volume_note') or 'stage did not run'}")

    img = load_canonical(label_volume)
    data = np.asarray(img.get_fdata()).astype(np.int32)
    mask = np.isin(data, L.TSS_CANAL_LABELS)
    if not mask.any():
        raise SkipStage(
            f"the label volume has no canal label ({', '.join(str(v) for v in L.TSS_CANAL_LABELS)})")

    zooms = img.header.get_zooms()
    voxel_area = float(zooms[LR_AXIS]) * float(zooms[AP_AXIS])
    profile = canal_area_profile(mask, voxel_area)

    narrowing = profile.get("max_narrowing_pct")
    flagged = bool(narrowing is not None and narrowing >= cfg.canal_stenosis_pct)
    payload = {
        "mask_file": str(label_volume),
        "canal_labels": list(L.TSS_CANAL_LABELS),
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
            "The narrowing threshold is this project's own, applied to the study's own "
            "median. Published cut-offs exist only for the cervical canal and are used "
            "separately in the `compression` stage, with the cohort they came from.",
        ],
    }
    write_json(cfg.intermediate_dir / "canal.json", payload)
    status = Status.OK if profile.get("median_area_mm2") else Status.PARTIAL
    return StageResult(name="canal", status=status, evidence=Evidence.MEASUREMENT,
                       reason=profile.get("note"), data=payload)
