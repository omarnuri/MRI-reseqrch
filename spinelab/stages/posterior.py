"""Stage `posterior`: left/right comparison of facet and costal-process regions.

This is the stage the whole case hangs on ("bending left -> right-sided pain"),
and it is where the old pipeline went wrong in the most consequential way. Old
Cell 10:

* built the region from semantic labels 41-48 *pooled*, then split it into left
  and right at ``shape[0] // 2`` — the centre of the image array, not the
  patient's midline, on a volume whose axis 0 was not guaranteed to be left-right;
* thresholded at 85% of the 99th percentile of the whole region, mixing both
  sides into one threshold;
* was run on a coronal STIR resampled onto a sagittal mask grid, so the two sides
  had different in-field coverage;
* and the resulting "9x brighter on the right" was reported as an acute
  right-sided facet syndrome.

Here the sides come from SPINEPS's own side labels (superior/inferior articular
process left = 45/47, right = 46/48; costal process left = 43, right = 44), so no
midline has to be guessed. Coverage per side is measured and the comparison is
refused when the two sides are not comparably in-field. Results are per level as
well as pooled, because a single global number cannot be acted on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import labels as L
from ..analysis import asymmetry_ratio, bright_fraction, robust_threshold
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import load_canonical, resample_mask_to, write_json

#: Minimum in-field coverage of a side's region before the sides may be compared.
MIN_SIDE_COVERAGE = 0.7
#: Maximum tolerated coverage imbalance between the two sides.
MAX_COVERAGE_RATIO = 1.25

GROUPS = {
    "facet": (L.FACET_LEFT, L.FACET_RIGHT),
    "costal_process": (L.COSTAL_LEFT, L.COSTAL_RIGHT),
}


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    picks = ctx.stage_data("ingest").get("picks", {})
    spineps = ctx.stage_data("spineps")
    semantic_masks = [Path(p) for p in spineps.get("semantic_masks", [])]
    instance_masks = [Path(p) for p in spineps.get("instance_masks", [])]
    if not semantic_masks:
        raise SkipStage("no SPINEPS semantic mask — posterior elements are not segmented")

    fatsat = picks.get("FATSAT_BEST")
    reference = fatsat or picks.get("T2_SAG")
    if not reference:
        raise SkipStage("no image to measure signal on")

    fat_suppressed = bool(fatsat)
    evidence = Evidence.HEURISTIC if fat_suppressed else Evidence.NOT_DIAGNOSTIC

    img = load_canonical(reference)
    image = np.asarray(img.get_fdata(), dtype=float)
    sem = np.asarray(resample_mask_to(load_canonical(semantic_masks[0]), img)
                     .get_fdata()).astype(np.int32)
    inst = None
    if instance_masks:
        inst = np.asarray(resample_mask_to(load_canonical(instance_masks[0]), img)
                          .get_fdata()).astype(np.int32)

    present = set(int(v) for v in np.unique(sem))
    groups_out: dict[str, dict] = {}
    warnings: list[str] = []

    # The bright-signal threshold comes from vertebral marrow — a tissue that is
    # not under test — so it cannot be dragged along by the very asymmetry being
    # measured. On a fat-suppressed sequence normal marrow is dark, and peri-facet
    # fluid or oedema is what rises above it.
    corpus_label = L.resolve_corpus_label(sem)
    reference_name = None
    threshold = None
    if corpus_label is not None:
        threshold = robust_threshold(image[sem == corpus_label], k=cfg.posterior_reference_k)
        reference_name = f"vertebral marrow (semantic label {corpus_label})"
    if threshold is None:
        posterior_all = np.isin(sem, list(L.POSTERIOR_MIDLINE))
        if posterior_all.any():
            threshold = robust_threshold(image[posterior_all], k=cfg.posterior_reference_k)
            reference_name = "midline posterior elements (arch/spinous process)"
    fallback_percentile = threshold is None
    if fallback_percentile:
        reference_name = (f"{cfg.posterior_bright_percentile:.0f}th percentile inside the "
                          "region itself (no reference tissue available — weaker)")

    for group, (left_labels, right_labels) in GROUPS.items():
        if not (set(left_labels) & present) and not (set(right_labels) & present):
            groups_out[group] = {"status": "absent",
                                 "reason": f"labels {left_labels + right_labels} not present "
                                           "in the semantic mask"}
            continue
        left = np.isin(sem, list(left_labels))
        right = np.isin(sem, list(right_labels))
        cov_l = _coverage(image, left)
        cov_r = _coverage(image, right)
        comparable, why = _comparable(cov_l, cov_r)

        # One threshold for both sides so they are measured against the same
        # yardstick; from reference tissue where available.
        pooled = left | right
        group_threshold = threshold
        if group_threshold is None:
            group_threshold = bright_fraction(image, pooled,
                                             cfg.posterior_bright_percentile)["threshold"]
        side_stats = {}
        for side_name, mask, cov in (("left", left, cov_l), ("right", right, cov_r)):
            values = image[mask.astype(bool)]
            inside = values[values > 0]
            n_bright = int((values > group_threshold).sum()) if group_threshold else 0
            side_stats[side_name] = {
                "region_voxels": int(mask.sum()),
                "in_fov_coverage": round(cov, 3),
                "median_intensity": round(float(np.median(inside)), 3) if inside.size else None,
                "bright_voxels": n_bright,
                "bright_fraction": round(n_bright / max(int(mask.sum()), 1), 5),
            }

        entry = {
            "status": "measured" if comparable else "not_comparable",
            "threshold_reference": reference_name,
            "threshold_k_sigma": cfg.posterior_reference_k if not fallback_percentile else None,
            "threshold_value": None if group_threshold is None else round(float(group_threshold), 3),
            "sides": side_stats,
            "bright_fraction_comparison": asymmetry_ratio(
                side_stats["left"]["bright_fraction"], side_stats["right"]["bright_fraction"]),
            "median_intensity_comparison": asymmetry_ratio(
                side_stats["left"]["median_intensity"] or 0.0,
                side_stats["right"]["median_intensity"] or 0.0),
            "volume_comparison": asymmetry_ratio(
                side_stats["left"]["region_voxels"], side_stats["right"]["region_voxels"]),
        }
        if not comparable:
            entry["reason"] = why
            warnings.append(f"{group}: {why}")
        if inst is not None:
            entry["per_level"] = _per_level(image, sem, inst, left_labels, right_labels,
                                            group_threshold)
        groups_out[group] = entry

    limits = [
        "Left/right sides come from the segmentation's own side labels, not from a "
        "midline guess.",
        f"Bright-signal threshold derived from {reference_name}.",
        "Signal intensity in MR has no absolute units: only the two sides of the same "
        "image are compared, never one study against another.",
    ]
    if not fat_suppressed:
        limits.insert(0,
            "NO fat-suppressed sequence was available, so this measurement was made on "
            "plain T2. Bright signal there is dominated by fat and cannot be read as "
            "oedema or effusion. Reported for completeness only — it supports no conclusion.")
    else:
        limits.insert(0,
            f"Measured on the {picks.get('FATSAT_LABEL')} series in the "
            f"{picks.get('FATSAT_PLANE')} plane. Thoracic facet joints are oriented close to "
            "the coronal plane, so a sagittal-only acquisition under-samples them; an axial "
            "or oblique fat-suppressed series is what a targeted question needs.")

    payload = {
        "reference_image": reference,
        "fat_suppressed": fat_suppressed,
        "groups": groups_out,
        "warnings": warnings,
        "interpretation_limits": limits,
    }
    write_json(cfg.intermediate_dir / "posterior.json", payload)

    measured = [g for g in groups_out.values() if g.get("status") == "measured"]
    if not measured:
        return StageResult(name="posterior", status=Status.PARTIAL, evidence=evidence,
                           reason="no side comparison was admissible; see warnings",
                           data=payload)
    return StageResult(name="posterior", status=Status.OK, evidence=evidence, data=payload)


def _coverage(image: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    n = int(mask.sum())
    if n == 0:
        return 0.0
    return float((image[mask] > 0).sum()) / float(n)


def _comparable(cov_left: float, cov_right: float) -> tuple[bool, str | None]:
    if min(cov_left, cov_right) <= 0:
        return False, "one side has no in-field voxels on this sequence"
    if min(cov_left, cov_right) < MIN_SIDE_COVERAGE:
        return False, (f"in-field coverage too low (L={cov_left:.2f}, R={cov_right:.2f}); "
                       "the sequence does not cover both sides of the region")
    ratio = max(cov_left, cov_right) / min(cov_left, cov_right)
    if ratio > MAX_COVERAGE_RATIO:
        return False, (f"coverage differs between sides (L={cov_left:.2f}, R={cov_right:.2f}); "
                       "a left/right difference here would be a field-of-view artefact")
    return True, None


def _per_level(image, sem, inst, left_labels, right_labels, threshold) -> list[dict]:
    """Side comparison per vertebral level, so a finding has an address."""
    out = []
    if threshold is None:
        return out
    for label_id in sorted(int(v) for v in np.unique(inst)):
        if not L.is_vertebra(label_id):
            continue
        level = inst == label_id
        left = level & np.isin(sem, list(left_labels))
        right = level & np.isin(sem, list(right_labels))
        if int(left.sum()) < 20 or int(right.sum()) < 20:
            continue
        lb = int((image[left] > threshold).sum())
        rb = int((image[right] > threshold).sum())
        out.append({
            "level": L.vertebra_name(label_id),
            "left_region_voxels": int(left.sum()),
            "right_region_voxels": int(right.sum()),
            "left_bright_voxels": lb,
            "right_bright_voxels": rb,
            "comparison": asymmetry_ratio(lb / max(int(left.sum()), 1),
                                          rb / max(int(right.sum()), 1)),
        })
    return out
