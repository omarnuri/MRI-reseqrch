"""Stage `normative`: this subject's levels placed in an open cohort's spread.

The first comparison in this project against anybody other than the subject
himself. What it is, and — more importantly — what it is not, is set out in
`spinelab/normative.py`; the short version is that the spread comes from sixty-odd
CC0 whole-spine volunteers (OpenNeuro ds005616), and that this project computed
that spread rather than reading it off a published table.

So the evidence level here is `MEASUREMENT`, not `CALIBRATED`. `CALIBRATED` in
this codebase means a cut-off someone published together with the cohort it was
fitted on and the outcome it predicts — today that is the cervical Spinal Cord
Toolbox measures and nothing else. A percentile is a position, not a verdict, and
this stage never says which side of anything a value falls on.

The one thing that makes the comparison worth making at all is that both sides are
measured by the same function, `normative.measure_label_volume`, from the same
kind of mask: one label per whole vertebra. That is also why no height and no
wedge angle appears here — those need the vertebral body separated from the
posterior elements, `stages/geometry.py` does that properly against this subject
himself, and mixing the two definitions would be exactly the sort of quiet
mismatch this pipeline exists to refuse.
"""

from __future__ import annotations

import numpy as np

from .. import labels as L
from .. import normative as N
from ..analysis import percentile_of, sct_level_index
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..runlog import event, get_logger
from ..utils import load_canonical, write_json

log = get_logger(__name__)

#: Above this voxel size the study is not the same kind of acquisition as the
#: 1 mm isotropic reference, and every comparison says so.
SAME_FAMILY_MAX_MM = 1.5


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    reference = N.load(cfg.cache_dir or cfg.work_dir)
    if reference is None:
        raise SkipStage(
            "no reference cohort in the cache — build it once with "
            f"`python -m spinelab normative --cache {cfg.cache_dir or cfg.work_dir}` "
            f"(about 17 MB of masks from {N.DATASET}, no images)")

    label_volume = ctx.stage_data("totalspineseg").get("label_volume")
    if not label_volume:
        raise SkipStage("TotalSpineSeg produced no verified label volume, so this "
                        "study has no per-level labels to place in the cohort")

    img = load_canonical(label_volume)
    data = np.asarray(img.get_fdata()).astype(np.int32)
    zooms = img.header.get_zooms()

    vertebra_map = N.tss_index_map(data)
    if not vertebra_map:
        raise SkipStage("no recognised vertebra label in the TotalSpineSeg volume")

    measured = N.measure_label_volume(
        data, vertebra_map, zooms,
        discs=data, disc_index_map=N.tss_disc_index_map(data),
        canal=np.isin(data, L.TSS_CANAL_LABELS),
        cord=(data == L.TSS_SPINAL_CORD))
    if not measured:
        raise SkipStage("no level could be measured on this study")

    match, match_note = _protocol_match(ctx, zooms, reference)
    levels_out: dict[str, dict] = {}
    compared = 0
    for name, entry in measured.items():
        reference_level = (reference.get("levels") or {}).get(name) or {}
        metrics: dict[str, dict] = {}
        for metric, value in entry.items():
            summary = reference_level.get(metric)
            record = {"value": value}
            if summary:
                record.update({
                    "cohort_n": summary["n"],
                    "cohort_median": summary["median"],
                    "cohort_p5_p95": summary["p5_p95"],
                    "percentile": percentile_of(value, summary.get("values") or []),
                })
                compared += 1
            else:
                record["cohort"] = "this level is not covered by the reference cohort"
            metrics[metric] = record
        levels_out[name] = {"metrics": metrics, "protocol_match": match}

    payload = {
        "source_label_volume": str(label_volume),
        "cohort": reference.get("cohort", {}),
        "protocol_match": match,
        "protocol_note": match_note,
        "levels": dict(sorted(levels_out.items(),
                              key=lambda kv: sct_level_index(kv[0]) or 99)),
        "comparisons_made": compared,
        "interpretation_limits": [
            "A percentile is a position inside a reference sample, not a diagnosis "
            "and not a threshold. Nothing here says a value is normal or abnormal, "
            "because no cut-off derived from this cohort has ever been validated.",
            match_note,
            *reference.get("what_this_is_not", []),
            "Both sides of every comparison are measured by the same function from "
            "one label per whole vertebra. Vertebral body heights and wedge angles "
            "are deliberately absent: they need the body separated from the "
            "posterior elements, and are reported by the geometry stage instead.",
        ],
    }
    write_json(cfg.intermediate_dir / "normative.json", payload)
    event("normative", levels=len(levels_out), comparisons=compared, match=match)

    if not compared:
        return StageResult(
            name="normative", status=Status.PARTIAL, evidence=Evidence.MEASUREMENT,
            reason="no level of this study overlaps the reference cohort",
            data=payload)
    return StageResult(name="normative", status=Status.OK,
                       evidence=Evidence.MEASUREMENT, data=payload)


def _protocol_match(ctx: Context, zooms, reference: dict) -> tuple[str, str]:
    """How far this acquisition is from the one the reference was built on."""
    cohort = reference.get("cohort", {})
    sequence = cohort.get("sequence", "the reference acquisition")
    picks = ctx.stage_data("ingest").get("picks", {})
    coarsest = max(float(z) for z in zooms[:3])
    if picks.get("survey_only"):
        return "survey_scan", (
            f"MEASURED ON A POSITIONING SCAN. The reference is {sequence}; this "
            f"block has nothing but a fast gradient echo at {coarsest:.1f} mm, and "
            "the segmentation that produced these labels is working outside its "
            "training distribution. Treat every position below as an indication of "
            "where to look on the images, not as a measurement.")
    if coarsest <= SAME_FAMILY_MAX_MM:
        return "same_family", (
            f"This study is {coarsest:.1f} mm at its coarsest against the reference's "
            f"{sequence} — the same kind of acquisition, which is the case this "
            "comparison is weakest at being wrong about.")
    return "coarser_slices", (
        f"This study is {coarsest:.1f} mm at its coarsest; the reference is "
        f"{sequence}. Thicker slices blur the top and bottom of every vertebra, so "
        "level boundaries and anything derived from them carry that difference.")
