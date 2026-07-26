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
from ..utils import find_outputs, load_canonical, resample_mask_to, write_json

GOOD_DICE = 0.7


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    spineps = ctx.stage_data("spineps")
    tss = ctx.stage_data("totalspineseg")
    semantic_masks = [Path(p) for p in spineps.get("semantic_masks", [])]
    tss_root = tss.get("output_dir")
    if not semantic_masks or not tss_root:
        raise SkipStage("need both SPINEPS semantic and TotalSpineSeg outputs")

    sp_img = load_canonical(semantic_masks[0])
    sp_data = np.asarray(sp_img.get_fdata()).astype(np.int32)
    sp_cord = sp_data == L.SPINAL_CORD
    if int(sp_cord.sum()) < 50:
        raise SkipStage(f"SPINEPS semantic mask has no cord label ({L.SPINAL_CORD})")

    candidates = [p for p in find_outputs(tss_root, "*.nii.gz")
                  if "cord" in str(p).lower() or "step2" in str(p).lower()]
    if not candidates:
        raise SkipStage("no TotalSpineSeg volume containing a cord label")

    tss_cord = None
    used = None
    for path in candidates:
        img = resample_mask_to(load_canonical(path), sp_img)
        data = np.asarray(img.get_fdata()).astype(np.int32)
        mask = data == L.TSS_SPINAL_CORD if "step2" in str(path).lower() else data > 0.5
        if int(mask.sum()) >= 50:
            tss_cord, used = mask, path
            break
    if tss_cord is None:
        raise SkipStage("TotalSpineSeg cord mask is empty after resampling")

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
