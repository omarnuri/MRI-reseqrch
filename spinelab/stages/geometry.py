"""Stage `geometry`: per-vertebra body heights, wedge angles, chain curvature."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import labels as L
from ..analysis import body_heights, curvature_metrics, longest_run, wedge_angle_deg
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import AP_AXIS, SI_AXIS, load_canonical, resample_mask_to, write_json


def parse_centroids(payload) -> dict[int, np.ndarray]:
    """Read SPINEPS centroid JSON in either the list or dict layout."""
    out: dict[int, np.ndarray] = {}
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "label" in item:
                try:
                    label = int(item["label"])
                except (TypeError, ValueError):
                    continue
                out[label] = np.array([item.get("X", 0), item.get("Y", 0), item.get("Z", 0)],
                                      dtype=float)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            try:
                label = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                out[label] = np.array([value.get("X", 0), value.get("Y", 0), value.get("Z", 0)],
                                      dtype=float)
            elif isinstance(value, (list, tuple)) and len(value) >= 3:
                out[label] = np.array(value[:3], dtype=float)
    return out


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    spineps = ctx.stage_data("spineps")
    instance_masks = [Path(p) for p in spineps.get("instance_masks", [])]
    semantic_masks = [Path(p) for p in spineps.get("semantic_masks", [])]
    if not instance_masks:
        raise SkipStage("no SPINEPS vertebra instance mask")

    inst_img = load_canonical(instance_masks[0])
    inst = np.asarray(inst_img.get_fdata()).astype(np.int32)
    zooms = inst_img.header.get_zooms()
    ap_mm, si_mm = float(zooms[AP_AXIS]), float(zooms[SI_AXIS])

    corpus_mask = None
    corpus_label = None
    notes: list[str] = []
    if semantic_masks:
        sem_img = load_canonical(semantic_masks[0])
        sem_img = resample_mask_to(sem_img, inst_img)
        sem = np.asarray(sem_img.get_fdata()).astype(np.int32)
        corpus_label = L.resolve_corpus_label(sem)
        if corpus_label is not None:
            corpus_mask = sem == corpus_label
            notes.append(f"heights measured on the vertebral body (semantic label {corpus_label})")
        else:
            notes.append("no corpus label found in the semantic mask — heights use the full "
                         "vertebra mask and wedge angles are NOT reliable")
    else:
        notes.append("no semantic mask — heights use the full vertebra mask "
                     "(includes the spinous process) and wedge angles are NOT reliable")

    per_vertebra = []
    for label_id in sorted(int(v) for v in np.unique(inst)):
        if not L.is_vertebra(label_id):
            continue
        full = inst == label_id
        voxels = int(full.sum())
        if voxels < 100:
            continue
        body = full & corpus_mask if corpus_mask is not None else full
        body_based = corpus_mask is not None and int(body.sum()) >= 100
        if not body_based:
            body = full

        record = {
            "label_id": label_id,
            "name": L.vertebra_name(label_id),
            "voxel_count": voxels,
            "volume_mm3": round(voxels * ap_mm * si_mm * float(zooms[0]), 1),
            "body_based": body_based,
            "truncated": _touches_boundary(body),
        }
        heights = body_heights(body)
        if heights is not None:
            post_mm = heights.posterior_voxels * si_mm
            ant_mm = heights.anterior_voxels * si_mm
            ap_mm_total = heights.ap_span_voxels * ap_mm
            wedge = wedge_angle_deg(post_mm, ant_mm, ap_mm_total)
            record.update({
                "ap_width_mm": round(ap_mm_total, 2),
                "posterior_height_mm": round(post_mm, 2),
                "anterior_height_mm": round(ant_mm, 2),
                "ant_post_height_ratio": round(ant_mm / post_mm, 3) if post_mm > 0 else None,
                "wedge_angle_deg": None if wedge is None else round(wedge, 2),
            })
        per_vertebra.append(record)

    if not per_vertebra:
        raise SkipStage("instance mask contains no vertebra large enough to measure")

    thoracic = [r for r in per_vertebra if r["label_id"] in L.THORACIC_LABELS]

    # A vertebra clipped by the edge of the field of view has no measurable height.
    # On the real study C7 sat at the top edge and came out with a posterior height of
    # 1.99 mm against 13.28 mm anteriorly — a wedge of -19.5 degrees. It happened to
    # be negative and so did not become the maximum, but the same clipping at the
    # bottom of the stack would have produced a large positive wedge and headlined the
    # report as a deformity. The level stays in the table, flagged, and is kept out of
    # every number derived from heights.
    intact = [r for r in per_vertebra if not r.get("truncated")]
    truncated_levels = [r["name"] for r in per_vertebra if r.get("truncated")]
    if truncated_levels:
        notes.append(
            f"excluded from wedge statistics — clipped by the edge of the field of "
            f"view, so their heights are not measurable: {', '.join(truncated_levels)}")
    wedges = [r.get("wedge_angle_deg") for r in intact]
    run_len = longest_run(wedges, lambda w: w is not None and w >= cfg.wedge_scheuermann_deg)

    # Centroids are computed from the canonical instance mask, not read from the
    # SPINEPS centroid JSON. That file stores coordinates in the *original* image's
    # own axis order together with a direction code; consuming its X/Y/Z as if they
    # were canonical millimetres silently mixes coordinate conventions, and the
    # curvature numbers that come out of it look plausible either way. The JSON is
    # loaded only as a cross-check on how many levels were found.
    centroids = _centroids_from_mask(inst, zooms)
    notes.append("centroids computed from the canonical instance mask "
                 "(the SPINEPS centroid file uses the original image's axis order)")
    for path in spineps.get("centroid_files", []):
        try:
            with open(path, encoding="utf-8") as fh:
                declared = parse_centroids(json.load(fh))
        except Exception:
            continue
        if declared:
            missing = sorted(set(centroids) - set(declared))
            if missing:
                notes.append("SPINEPS centroid file lists fewer levels than the mask: "
                             f"missing {[L.vertebra_name(m) for m in missing]}")
            break

    curvature = curvature_metrics(centroids, L.THORACIC_LABELS)
    scheuermann = run_len >= cfg.wedge_scheuermann_run

    payload = {
        "per_vertebra": per_vertebra,
        "levels_measured": [r["name"] for r in per_vertebra],
        "thoracic_levels_measured": [r["name"] for r in thoracic],
        "corpus_label_used": corpus_label,
        "max_wedge_angle_deg": max((w for w in wedges if w is not None), default=None),
        "wedge_threshold_deg": cfg.wedge_scheuermann_deg,
        "longest_run_at_or_above_threshold": run_len,
        "scheuermann_pattern": scheuermann,
        "scheuermann_rule": (f">= {cfg.wedge_scheuermann_deg} deg wedging on "
                             f"{cfg.wedge_scheuermann_run}+ contiguous vertebrae"),
        "curvature": curvature,
        "notes": notes,
    }
    write_json(cfg.intermediate_dir / "geometry.json", payload)
    return StageResult(name="geometry", status=Status.OK, evidence=Evidence.MEASUREMENT,
                       data=payload)


def _touches_boundary(mask: np.ndarray) -> bool:
    """Is this body clipped by the edge of the volume?

    Only the anterior-posterior and superior-inferior faces are checked (canonical
    axes 1 and 2). Those are the two directions the height and AP-width measurements
    span. The left-right faces are not: a sagittal stack is only a few centimetres
    wide in the slice direction, so a vertebral body legitimately reaches both edges
    there, and flagging that would exclude every level.
    """
    if mask.ndim != 3 or not mask.any():
        return False
    for axis in (1, 2):
        first = mask.take(0, axis=axis)
        last = mask.take(mask.shape[axis] - 1, axis=axis)
        if first.any() or last.any():
            return True
    return False


def _centroids_from_mask(inst: np.ndarray, zooms) -> dict[int, np.ndarray]:
    """Centre of mass per vertebra label, in millimetres."""
    out: dict[int, np.ndarray] = {}
    scale = np.array([float(z) for z in zooms[:3]], dtype=float)
    for label_id in np.unique(inst):
        label_id = int(label_id)
        if not L.is_vertebra(label_id):
            continue
        coords = np.argwhere(inst == label_id)
        if coords.size == 0:
            continue
        out[label_id] = coords.mean(axis=0) * scale
    return out
