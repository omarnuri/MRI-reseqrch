"""Stage `discs`: per-level disc signal ranking from TotalSpineSeg disc labels.

This is NOT Pfirrmann grading. Pfirrmann is a 5-class judgement of nucleus
signal, structure, disc height and endplate distinction, defined on lumbar discs;
the published graders (SpineNet and friends) are trained on lumbar studies and
anchor themselves on L5/S1, so on a thoracic-only study they either refuse to run
or produce an unanchored guess. What is defensible without those weights is a
*within-study* ranking of disc signal, which is what this stage produces — under
its own name, with the disc level spelled out ("T8-T9"), never a raw label id.
"""

from __future__ import annotations

import numpy as np

from .. import labels as L
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import find_outputs, load_canonical, resample_mask_to, write_json


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    t2_sag = ctx.require_sequence("T2_SAG")
    tss = ctx.stage_data("totalspineseg")
    root = tss.get("output_dir")
    if not root:
        raise SkipStage("TotalSpineSeg did not run — no disc labels")

    candidates = [p for p in find_outputs(root, "*.nii.gz") if "step2" in str(p).lower()]
    if not candidates:
        candidates = find_outputs(root, "*.nii.gz")
    if not candidates:
        raise SkipStage("no label volume in the TotalSpineSeg output")

    t2_img = load_canonical(t2_sag)
    t2 = np.asarray(t2_img.get_fdata(), dtype=float)
    label_img = resample_mask_to(load_canonical(candidates[0]), t2_img)
    label_data = np.asarray(label_img.get_fdata()).astype(np.int32)

    rows = []
    for label in sorted(int(v) for v in np.unique(label_data)):
        if label < L.TSS_DISC_LABEL_MIN:
            # < 63 is cord (1), CSF (2), vertebrae (11-47) or sacrum (50). Ranking
            # everything non-zero is how the CSF-filled canal — the brightest thing
            # on T2 — once came out as "the healthiest disc".
            continue
        mask = label_data == label
        n = int(mask.sum())
        if n < 50:
            continue
        values = t2[mask]
        rows.append({
            "label": label,
            "level": L.tss_disc_name(label),
            "voxel_count": n,
            "mean_signal": round(float(values.mean()), 3),
            "median_signal": round(float(np.median(values)), 3),
        })

    if not rows:
        raise SkipStage(f"no disc labels (>= {L.TSS_DISC_LABEL_MIN}) in the label volume")

    cohort = float(np.median([r["median_signal"] for r in rows]))
    rows.sort(key=lambda r: r["median_signal"])  # darkest (most dehydrated-looking) first
    for rank, row in enumerate(rows, start=1):
        row["darkness_rank"] = rank
        row["relative_to_cohort"] = round(row["median_signal"] / cohort, 3) if cohort > 0 else None

    payload = {
        "reference_image": t2_sag,
        "label_volume": str(candidates[0]),
        "cohort_median_signal": round(cohort, 3),
        "discs": rows,
        "method": "within-study T2 signal ranking of TotalSpineSeg disc labels",
        "interpretation_limits": [
            "Not a Pfirrmann grade. Ranking is relative to the other discs in this same "
            "study and cannot be compared across studies or scanners.",
            "Thoracic discs are normally thinner and darker than lumbar discs, so a low "
            "rank here is not by itself degeneration.",
            "Disc height and herniation are not measured by this stage.",
        ],
    }
    write_json(cfg.intermediate_dir / "discs.json", payload)
    return StageResult(name="discs", status=Status.OK, evidence=Evidence.MEASUREMENT,
                       data=payload)
