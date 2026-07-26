"""Stage `fatsat_qc`: did the fat suppression actually work?

Everything the pipeline says about oedema rests on one assumption: that the
"fat-suppressed" series really suppresses fat. That assumption is testable
without any extra data, and it has never been tested in this project.

Method: subcutaneous fat forms a band just inside the skin. Compare the 90th
percentile of that band against the median of the deep body core, on the
fat-suppressed series **and** on the plain T2 of the same study. On T2 the band is
much brighter than the core (fat is bright); if suppression worked, that ratio
collapses on the fat-suppressed series. The comparison is within one study, so
scanner scaling cancels out.

If the ratio does not drop, bright signal on that series is fat, not fluid, and
the oedema branch of the pipeline is void — which is worth knowing before reading
any of its numbers.
"""

from __future__ import annotations

import numpy as np

from ..analysis import rim_to_core_ratio
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import load_canonical, write_json

#: Ratio on the fat-suppressed series below which suppression looks effective.
GOOD_SUPPRESSION_RATIO = 1.3
#: How much lower the ratio must be than on the non-suppressed series.
MIN_RELATIVE_DROP = 0.75


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    picks = ctx.stage_data("ingest").get("picks", {})
    fatsat = picks.get("FATSAT_BEST")
    if not fatsat:
        raise SkipStage("no fat-suppressed series to check")
    control = picks.get("T2_SAG") or picks.get("T2_AX")

    fat_stats = rim_to_core_ratio(np.asarray(load_canonical(fatsat).get_fdata(), dtype=float))
    if fat_stats is None:
        raise SkipStage("could not delineate a body rim and core on the fat-suppressed series")

    control_stats = None
    if control:
        control_stats = rim_to_core_ratio(
            np.asarray(load_canonical(control).get_fdata(), dtype=float))

    ratio = fat_stats["rim_to_core_ratio"]
    verdict, effective = _verdict(ratio, control_stats)

    payload = {
        "fatsat_image": fatsat,
        "fatsat_label": picks.get("FATSAT_LABEL"),
        "fatsat_plane": picks.get("FATSAT_PLANE"),
        "fatsat_stats": fat_stats,
        "control_image": control,
        "control_stats": control_stats,
        "ratio_threshold": GOOD_SUPPRESSION_RATIO,
        "suppression_effective": effective,
        "verdict": verdict,
        "interpretation_limits": [
            "This is a geometric heuristic, not a phantom measurement: the band just "
            "inside the skin is mostly subcutaneous fat, but it also contains skin and, "
            "in a thin slab, partial-volume air.",
            "A negative result is the strong one: if the band does not darken relative to "
            "the body core, bright signal on this series cannot be read as fluid or oedema.",
        ],
    }
    write_json(cfg.intermediate_dir / "fatsat_qc.json", payload)
    return StageResult(
        name="fatsat_qc",
        status=Status.OK if effective else Status.PARTIAL,
        evidence=Evidence.HEURISTIC,
        reason=None if effective else verdict,
        data=payload,
    )


def _verdict(ratio: float, control: dict | None) -> tuple[str, bool]:
    if control is None:
        if ratio <= GOOD_SUPPRESSION_RATIO:
            return (f"subcutaneous band is not bright relative to the body core "
                    f"(ratio {ratio}); consistent with working fat suppression, though "
                    f"no non-suppressed series was available for comparison"), True
        return (f"subcutaneous band is {ratio}x the body core and there is no control "
                f"series to compare against — fat suppression is NOT confirmed"), False

    control_ratio = control["rim_to_core_ratio"]
    drop = ratio / control_ratio if control_ratio > 0 else 1.0
    if ratio <= GOOD_SUPPRESSION_RATIO and drop <= MIN_RELATIVE_DROP:
        return (f"fat suppression looks effective: subcutaneous band / core is {ratio} on "
                f"the suppressed series against {control_ratio} on the plain T2"), True
    if drop <= MIN_RELATIVE_DROP:
        return (f"partial suppression: the band drops from {control_ratio} to {ratio}, but "
                f"stays above {GOOD_SUPPRESSION_RATIO}. Bright signal may still be fat — "
                f"treat oedema readings as unconfirmed"), False
    return (f"NO evidence of fat suppression: band / core is {ratio} against "
            f"{control_ratio} on the plain T2. Bright signal on this series should not be "
            f"read as fluid or oedema"), False
