"""Evidence levels — the one rule that keeps this pipeline honest.

The previous version of this project put "острейший правосторонний фасеточный
синдром" in a markdown cell on the strength of (a) a bright-voxel count on a
non-fat-saturated sequence and (b) an autoencoder with *random, untrained*
weights. Both numbers were real; neither supported the claim.

So every value the pipeline emits is tagged with how it was obtained, and the
report renders the tag next to the number. A heuristic can still be useful — it
just may never be phrased as a finding.
"""

from __future__ import annotations

from enum import Enum


class Evidence(str, Enum):
    #: A value with a **published cut-off validated on an external clinical cohort**.
    #: The only level allowed to say which side of a threshold a number falls on,
    #: because it is the only one where the threshold came from somebody else's
    #: patients rather than from this study's own distribution. So far: the cervical
    #: canal measures in Spinal Cord Toolbox (compression probability, aSCOR).
    #: The cohort and the citation travel with every such number into the report.
    CALIBRATED = "calibrated"
    #: Output of a peer-reviewed, pretrained segmentation model run with its real
    #: published weights (SPINEPS, TotalSpineSeg, TotalSegmentator).
    MODEL = "model"
    #: A deterministic geometric measurement computed on such a mask. Trustworthy
    #: *as a number*; its clinical meaning still needs a radiologist.
    MEASUREMENT = "measurement"
    #: A transparent intensity heuristic (percentiles, robust z-scores). Sensitive
    #: to sequence type, coil shading and normalisation. Screening only.
    HEURISTIC = "heuristic"
    #: Wired up but not usable for inference (no weights, gated, wrong anatomy).
    #: Never contributes to any conclusion.
    NOT_DIAGNOSTIC = "not_diagnostic"


LABELS_RU = {
    Evidence.CALIBRATED: "ПОРОГ ИЗ ВНЕШНЕЙ КОГОРТЫ",
    Evidence.MODEL: "ВАЛИДИРОВАННАЯ МОДЕЛЬ",
    Evidence.MEASUREMENT: "ИЗМЕРЕНИЕ ПО МАСКЕ",
    Evidence.HEURISTIC: "ЭВРИСТИКА — СКРИНИНГ, НЕ ДИАГНОЗ",
    Evidence.NOT_DIAGNOSTIC: "НЕ ПРИГОДНО ДЛЯ ВЫВОДОВ",
}

COLORS = {
    Evidence.CALIBRATED: "#0ea5e9",
    Evidence.MODEL: "#10b981",
    Evidence.MEASUREMENT: "#f59e0b",
    Evidence.HEURISTIC: "#ef4444",
    Evidence.NOT_DIAGNOSTIC: "#6b7280",
}


class Status(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"
    CACHED = "cached"


def badge_html(level: Evidence) -> str:
    color = COLORS[level]
    text = LABELS_RU[level]
    return (
        f"<span style='background:{color};color:#fff;font-size:11px;"
        f"padding:2px 8px;border-radius:10px;font-weight:600;white-space:nowrap'>"
        f"{text}</span>"
    )
