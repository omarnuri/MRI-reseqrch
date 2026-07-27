"""Stage `crosscheck`: independent vertebra segmentation as a reliability check.

TotalSegmentator ships an MR-specific vertebra task (`vertebrae_mr`, dataset 756)
trained separately from SPINEPS. Running both and comparing per level tells us
which levels the two agree on — and therefore which numbers downstream are worth
reading. This is what the old MedSAM2 "cross-validation" was trying to be: that
one prompted a promptable segmenter with a centroid and computed Dice on a single
2D slice, which measures prompt quality, not mask correctness.

Only worth its GPU time when there is GPU time to spare — hence it sits behind
`Config.quality` and is off in the standard profile.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from .. import labels as L
from ..analysis import dice
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import (child_env, clean_reason, load_canonical, resample_mask_to, run_tool,
                     write_json)

TASK = "vertebrae_mr"
#: Below this, the two models disagree enough that per-level numbers for that
#: level should be read off the images rather than from the table.
RELIABLE_DICE = 0.7


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    if not cfg.quality:
        raise SkipStage("standard profile — set quality=True to spend GPU time on cross-checks")

    t2_sag = ctx.require_sequence("T2_SAG")
    spineps = ctx.stage_data("spineps")
    instance_masks = spineps.get("instance_masks") or []
    if not instance_masks:
        raise SkipStage("no SPINEPS instance mask to check against")

    try:
        import totalsegmentator  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SkipStage(f"TotalSegmentator not importable: {exc}") from exc

    out_dir = cfg.intermediate_dir / "crosscheck_vertebrae_mr"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = child_env()
    device = "cpu" if cfg.resolve_device() == "cpu" else "gpu"
    # With ml=True the `output` argument is a *file* path, not a directory. Passing the
    # directory made TotalSegmentator write `crosscheck_vertebrae_mr.nii` as a sibling
    # of it, so this stage searched inside the directory, found nothing, and reported
    # "produced no output" for a model that had just saved 4.4 MB of labels.
    mask_path = out_dir / "vertebrae_mr.nii.gz"
    script = (
        "from totalsegmentator.python_api import totalsegmentator\n"
        f"totalsegmentator(input={str(t2_sag)!r}, output={str(mask_path)!r}, "
        f"task={TASK!r}, ml=True, verbose=False, device={device!r})\n"
    )
    try:
        proc = run_tool([sys.executable, "-c", script],
                        log_path=out_dir / "tool.log", timeout=cfg.timeout_ts_s, env=env)
    except subprocess.TimeoutExpired:
        return StageResult(name="crosscheck", status=Status.FAILED, evidence=Evidence.MODEL,
                           reason=f"timeout after {cfg.timeout_ts_s}s")

    # `.nii` as well as `.nii.gz`: TotalSegmentator does not always compress.
    produced = sorted(out_dir.rglob("*.nii.gz")) + sorted(out_dir.rglob("*.nii"))
    if not produced:
        return StageResult(
            name="crosscheck", status=Status.SKIPPED, evidence=Evidence.MODEL,
            reason=f"{TASK} produced no output: {clean_reason(proc.stderr or proc.stdout)}",
        )

    ref_img = load_canonical(instance_masks[0])
    ref = np.asarray(ref_img.get_fdata()).astype(np.int32)
    other = np.asarray(resample_mask_to(load_canonical(produced[0]), ref_img)
                       .get_fdata()).astype(np.int32)

    # The two label spaces differ, so compare geometry per level: for each SPINEPS
    # vertebra, find the other model's label with the largest overlap and score
    # that pair. A one-to-one match with high Dice is agreement; a split or a
    # missing match is a level to look at by eye.
    rows = []
    for label_id in sorted(int(v) for v in np.unique(ref)):
        if not L.is_vertebra(label_id):
            continue
        mask = ref == label_id
        if int(mask.sum()) < 100:
            continue
        overlapping = other[mask]
        overlapping = overlapping[overlapping > 0]
        if overlapping.size == 0:
            rows.append({"level": L.vertebra_name(label_id), "matched": False,
                         "dice": 0.0, "reliable": False,
                         "note": "the second model segmented nothing here"})
            continue
        values, counts = np.unique(overlapping, return_counts=True)
        best = int(values[int(np.argmax(counts))])
        score = dice(mask, other == best)
        rows.append({
            "level": L.vertebra_name(label_id),
            "matched": True,
            "matched_label_other_model": best,
            "dice": round(score, 3),
            "reliable": bool(score >= RELIABLE_DICE),
        })

    unreliable = [r["level"] for r in rows if not r["reliable"]]
    scored = [r["dice"] for r in rows if r.get("matched")]
    payload = {
        "second_model": f"TotalSegmentator task {TASK}",
        "mask_file": str(produced[0]),
        "device": device,
        "levels": rows,
        "mean_dice": round(float(np.mean(scored)), 3) if scored else None,
        "dice_threshold": RELIABLE_DICE,
        "levels_needing_visual_check": unreliable,
        "interpretation_limits": [
            "Dice between two different label spaces after nearest-neighbour "
            "resampling is coarse; 0.7-0.85 is normal even when both are correct.",
            "Disagreement does not say which model is right — it says a human should "
            "look at that level before trusting its numbers.",
        ],
    }
    write_json(cfg.intermediate_dir / "crosscheck.json", payload)
    status = Status.OK if not unreliable else Status.PARTIAL
    return StageResult(
        name="crosscheck", status=status, evidence=Evidence.MODEL,
        reason=None if not unreliable else f"levels to check by eye: {', '.join(unreliable)}",
        data=payload,
    )
