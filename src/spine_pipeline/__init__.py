"""Pure, CPU-only logic extracted from notebooks/spine_analysis_pipeline.ipynb.

This package mirrors the dependency-light computational helpers used by the
notebook so they can be unit-tested without a GPU or the heavy segmentation
tools (SPINEPS, TotalSpineSeg, TotalSegmentator, torch). The notebook keeps its
own inline copies so it stays self-contained in Colab; keep the two in sync when
changing logic here.
"""

from .serialization import to_jsonable
from .sequences import classify, pick_sequence, pick_any_orientation, pick_stir_or_fatsat
from .geometry import (
    LABEL_NAMES,
    angle_deg,
    parse_centroids,
    centroid_global_metrics,
    slab_si_extent_vox,
    col_si_extents,
    body_heights_vox,
    vertebra_metrics,
)
from .anomaly import zscore_anomaly
from .discs import rank_discs
from .muscles import normalize_t2, muscle_side_metrics, fatty_fraction_from_t1
from .costovertebral import bright_mask, side_bright_fractions
from .agreement import dice
from .outputs import detect_seg_outputs, clean_reason, load_json
from .aggregate import merge_per_vertebra, build_findings
from .perf import measure_tool, aggregate_performance, write_performance_log

__all__ = [
    "to_jsonable",
    "classify",
    "pick_sequence",
    "pick_any_orientation",
    "pick_stir_or_fatsat",
    "LABEL_NAMES",
    "angle_deg",
    "parse_centroids",
    "centroid_global_metrics",
    "slab_si_extent_vox",
    "col_si_extents",
    "body_heights_vox",
    "vertebra_metrics",
    "zscore_anomaly",
    "rank_discs",
    "normalize_t2",
    "muscle_side_metrics",
    "fatty_fraction_from_t1",
    "bright_mask",
    "side_bright_fractions",
    "dice",
    "detect_seg_outputs",
    "clean_reason",
    "load_json",
    "merge_per_vertebra",
    "build_findings",
    "measure_tool",
    "aggregate_performance",
    "write_performance_log",
]
