"""Stage `facets_axial`: per-joint, per-side measurement on the axial T2.

This study contains a 66-slice axial T2 at 0.49 mm in-plane — the highest
resolution series in it, and the plane in which thoracic facet joints are
actually resolvable. The previous pipeline picked that series and then never read
it: every statement it made about facet joints came from sagittal and coronal
data instead.

What is measured here, per joint and per side:

* the interface between the two bones that form the joint (inferior articular
  process above, superior articular process below) — the joint space plus its
  margins, not a blob of pooled posterior-element labels;
* its volume in mm^3, and the left/right difference;
* the fraction of that interface which is bright on T2, with the threshold taken
  from vertebral marrow. Fluid in a joint is bright on T2, so this is an
  effusion-oriented measure — but it is an intensity heuristic, and a facet joint
  6 voxels across is at the edge of what 0.49 x 0.49 x 4 mm can resolve.

What is NOT measured: joint space width in millimetres. With 4 mm slices and a
mask whose left-right precision comes from a 3.5 mm sagittal acquisition, a width
in millimetres would be a number with no support behind it.
"""

from __future__ import annotations

import numpy as np

from .. import labels as L
from ..analysis import asymmetry_ratio, facet_interface, robust_threshold
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import load_canonical, resample_mask_to, voxel_volume_mm3, write_json

#: Minimum interface size worth reporting, in voxels.
MIN_INTERFACE_VOXELS = 15


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    picks = ctx.stage_data("ingest").get("picks", {})
    axial = picks.get("T2_AX")
    if not axial:
        raise SkipStage("no axial series in this study")

    spineps = ctx.stage_data("spineps")
    instance_masks = spineps.get("instance_masks") or []
    semantic_masks = spineps.get("semantic_masks") or []
    if not instance_masks or not semantic_masks:
        raise SkipStage("need both SPINEPS instance and semantic masks")

    registered = ctx.masks_in_space("axial")
    if registered:
        instance_masks = [registered["instance_mask"]]
        semantic_masks = [registered["semantic_mask"]]

    img = load_canonical(axial)
    image = np.asarray(img.get_fdata(), dtype=float)
    inst = np.asarray(resample_mask_to(load_canonical(instance_masks[0]), img)
                      .get_fdata()).astype(np.int32)
    sem = np.asarray(resample_mask_to(load_canonical(semantic_masks[0]), img)
                     .get_fdata()).astype(np.int32)
    voxel_mm3 = voxel_volume_mm3(img)

    corpus_label = L.resolve_corpus_label(sem)
    threshold = None
    reference_name = None
    if corpus_label is not None:
        threshold = robust_threshold(image[sem == corpus_label], k=cfg.posterior_reference_k)
        reference_name = f"vertebral marrow on this axial series (label {corpus_label})"
    if threshold is None:
        raise SkipStage("no marrow reference region on the axial series — a bright-signal "
                        "threshold would have to come from the measured region itself")

    present_levels = sorted(int(v) for v in np.unique(inst) if L.is_vertebra(int(v)))
    joints = []
    for upper, lower in zip(present_levels, present_levels[1:]):
        if lower != upper + 1:
            continue  # not adjacent: a gap in the mask, not a joint
        entry = {
            "joint": f"{L.vertebra_name(upper)}-{L.vertebra_name(lower)}",
            "sides": {},
        }
        for side, inferior, superior in (
            ("left", L.INFERIOR_ARTICULAR_LEFT, L.SUPERIOR_ARTICULAR_LEFT),
            ("right", L.INFERIOR_ARTICULAR_RIGHT, L.SUPERIOR_ARTICULAR_RIGHT),
        ):
            roi = facet_interface(sem, inst, upper_label=upper, lower_label=lower,
                                  inferior_process=inferior, superior_process=superior,
                                  dilate=cfg.facet_dilate_voxels)
            n = int(roi.sum())
            if n < MIN_INTERFACE_VOXELS:
                entry["sides"][side] = {"interface_voxels": n, "measurable": False,
                                        "reason": "interface too small or processes not "
                                                  "segmented at this level"}
                continue
            values = image[roi]
            in_field = values[values > 0]
            bright = int((values > threshold).sum())
            entry["sides"][side] = {
                "interface_voxels": n,
                "interface_volume_mm3": round(n * voxel_mm3, 2),
                "in_fov_coverage": round(float(in_field.size) / n, 3),
                "median_signal": round(float(np.median(in_field)), 2) if in_field.size else None,
                "bright_voxels": bright,
                "bright_fraction": round(bright / n, 4),
                "measurable": True,
            }
        left, right = entry["sides"].get("left", {}), entry["sides"].get("right", {})
        if left.get("measurable") and right.get("measurable"):
            entry["comparison"] = {
                "bright_fraction": asymmetry_ratio(left["bright_fraction"],
                                                   right["bright_fraction"]),
                "interface_volume": asymmetry_ratio(left["interface_volume_mm3"],
                                                     right["interface_volume_mm3"]),
                "median_signal": asymmetry_ratio(left["median_signal"] or 0.0,
                                                  right["median_signal"] or 0.0),
            }
            entry["comparable"] = _coverage_ok(left, right)
            entry["above_noise_floor"] = _above_noise_floor(left, right)
            if not entry["above_noise_floor"]:
                entry["comparison"]["bright_fraction"]["note"] = (
                    f"too few bright voxels to compare "
                    f"({left['bright_voxels']} left, {right['bright_voxels']} right; "
                    f"at least {MIN_BRIGHT_VOXELS} on one side is required)")
        else:
            entry["comparable"] = False
            entry["above_noise_floor"] = False
        joints.append(entry)

    measurable = [j for j in joints if j.get("comparable")]
    if not measurable:
        return StageResult(
            name="facets_axial", status=Status.PARTIAL, evidence=Evidence.HEURISTIC,
            reason="no facet joint had both sides measurable on the axial series",
            data={"axial_image": axial, "joints": joints,
                  "threshold_value": round(float(threshold), 3),
                  "threshold_reference": reference_name},
        )

    # Rank only joints whose bright-voxel counts are large enough to be compared.
    # Without this the headline was three joints at "200% difference" — the value
    # `asymmetry_ratio` returns when one side is zero — produced by counts like 0
    # against 3 out of ~500 voxels. That is the noise a 3-sigma threshold makes, and
    # it was outranking every real difference.
    rankable = [j for j in measurable if j.get("above_noise_floor")]
    ranked = sorted(
        rankable,
        key=lambda j: -(j["comparison"]["bright_fraction"].get("diff_pct") or 0.0),
    )
    payload = {
        "axial_image": axial,
        "voxel_mm3": round(voxel_mm3, 4),
        "masks_motion_corrected": bool(registered and registered.get("applied")),
        "threshold_value": round(float(threshold), 3),
        "threshold_reference": reference_name,
        "threshold_k_sigma": cfg.posterior_reference_k,
        "joints": joints,
        "n_comparable": len(measurable),
        "n_above_noise_floor": len(rankable),
        "min_bright_voxels": MIN_BRIGHT_VOXELS,
        "largest_side_difference": [
            {"joint": j["joint"],
             "higher_side": j["comparison"]["bright_fraction"]["higher_side"],
             "diff_pct": j["comparison"]["bright_fraction"]["diff_pct"],
             "bright_voxels": {"left": j["sides"]["left"]["bright_voxels"],
                               "right": j["sides"]["right"]["bright_voxels"]}}
            for j in ranked[:3]
        ],
        "side_difference_verdict": (
            f"no facet joint has enough bright voxels to compare sides "
            f"(fewer than {MIN_BRIGHT_VOXELS} on both sides everywhere) — on this "
            f"series that is the expected result, not a negative finding"
            if not rankable else
            f"{len(rankable)} of {len(measurable)} joints have comparable bright-voxel "
            f"counts"),
        "interpretation_limits": [
            "Bright signal on T2 inside a facet joint interface is consistent with fluid, "
            "but degenerative change, partial-volume averaging with CSF or vessels, and "
            "segmentation error all produce the same number. This orders joints for a "
            "human to look at; it does not diagnose an effusion.",
            "Slice thickness is 4 mm and the mask's left-right precision comes from a "
            "3.5 mm sagittal acquisition, so a joint 2-4 mm across is at the resolution "
            "limit. Small left/right differences are not interpretable.",
            "Peri-facet and marrow oedema are NOT assessable here: this series has no fat "
            "suppression. Only the fat-suppressed series can speak to oedema, and only "
            "where it covers the joints.",
            "No joint-space width in millimetres is reported: the data cannot support one.",
        ],
    }
    write_json(cfg.intermediate_dir / "facets_axial.json", payload)
    return StageResult(name="facets_axial", status=Status.OK, evidence=Evidence.HEURISTIC,
                       data=payload)


#: Bright voxels needed on at least one side before the two sides may be compared.
#: The threshold is median + 3*sigma_MAD of reference marrow, so on a region of a few
#: hundred voxels a handful of voxels above it is what chance produces — for a normal
#: distribution about 0.13%, i.e. under one voxel in 500. Counts of 0 against 3 are
#: therefore indistinguishable from noise, and comparing them yielded "200%".
MIN_BRIGHT_VOXELS = 10


def _above_noise_floor(left: dict, right: dict) -> bool:
    return max(left.get("bright_voxels") or 0, right.get("bright_voxels") or 0) >= MIN_BRIGHT_VOXELS


def _coverage_ok(left: dict, right: dict, min_coverage: float = 0.8,
                 max_ratio: float = 1.25) -> bool:
    cl, cr = left.get("in_fov_coverage") or 0.0, right.get("in_fov_coverage") or 0.0
    if min(cl, cr) < min_coverage:
        return False
    return max(cl, cr) / max(min(cl, cr), 1e-9) <= max_ratio
