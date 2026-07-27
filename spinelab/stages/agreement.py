"""Stage `agreement`: cross-tool sanity check on the spinal cord mask.

Two independently trained models segmenting the same structure should agree. When
they do not, every downstream number that uses either mask is suspect — which is
worth knowing before reading any of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import labels as L
from ..analysis import dice
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import load_canonical, resample_mask_to, write_json

GOOD_DICE = 0.7


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    spineps = ctx.stage_data("spineps")
    tss = ctx.stage_data("totalspineseg")
    semantic_masks = [Path(p) for p in spineps.get("semantic_masks", [])]
    if not semantic_masks:
        raise SkipStage("need both SPINEPS semantic and TotalSpineSeg outputs")

    sp_img = load_canonical(semantic_masks[0])
    sp_data = np.asarray(sp_img.get_fdata()).astype(np.int32)
    sp_cord = sp_data == L.SPINAL_CORD
    if int(sp_cord.sum()) < 50:
        raise SkipStage(f"SPINEPS semantic mask has no cord label ({L.SPINAL_CORD})")

    # The verified label volume, and the cord by its label id. This used to search for
    # "cord" in the path, which found `step1_cord` — a soft probability map that was
    # then thresholded at 0.5. Comparing a probability map against a segmentation
    # produced a cord Dice of 0.12 and read as "the two tools disagree", which is a
    # statement about the file choice and not about either model.
    used = tss.get("label_volume")
    if not used:
        raise SkipStage("TotalSpineSeg produced no verified label volume — "
                        f"{tss.get('label_volume_note') or 'stage did not run'}")

    img = resample_mask_to(load_canonical(used), sp_img)
    data = np.asarray(img.get_fdata()).astype(np.int32)
    tss_cord = data == L.TSS_SPINAL_CORD
    if int(tss_cord.sum()) < 50:
        raise SkipStage(f"the label volume has no cord label ({L.TSS_SPINAL_CORD}) "
                        f"after resampling onto the SPINEPS grid")

    score = dice(sp_cord, tss_cord)
    payload = {
        "spineps_mask": str(semantic_masks[0]),
        "totalspineseg_mask": str(used),
        "spineps_cord_voxels": int(sp_cord.sum()),
        "totalspineseg_cord_voxels": int(tss_cord.sum()),
        "dice": round(score, 3),
        "threshold": GOOD_DICE,
        "agreement_ok": bool(score >= GOOD_DICE),
        "interpretation_limits": [
            "Dice after nearest-neighbour resampling between two different grids is a "
            "coarse check: a 1 mm isotropic mask resampled onto an anisotropic sagittal "
            "grid loses detail, so values around 0.7-0.8 are normal even when both "
            "segmentations are correct.",
            "Low agreement does not say which tool is wrong, only that a human should look.",
        ],
    }
    write_json(cfg.intermediate_dir / "agreement.json", payload)
    return StageResult(
        name="agreement",
        status=Status.OK if score >= GOOD_DICE else Status.PARTIAL,
        evidence=Evidence.MEASUREMENT,
        reason=None if score >= GOOD_DICE else f"low cord agreement (Dice {score:.2f})",
        data=payload,
    )
