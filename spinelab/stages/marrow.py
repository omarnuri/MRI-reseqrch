"""Stage `marrow`: bright-signal screen inside vertebral bodies.

This stage answers exactly one question: *on the fat-suppressed sequence of this
study, is any vertebral body brighter than the others in a way worth a second
look by a radiologist?* It does not detect bone-marrow oedema, and it says so.

Three substantive changes from the old Cell 8/8b:

1. **No fat-suppressed sequence, no result.** T2 without fat saturation cannot
   separate oedema from normal fatty marrow, so if the study has no STIR/TIRM/
   SPAIR/Dixon-water series the stage is skipped with that reason instead of
   emitting numbers that were then read as oedema.
2. **The mask is resampled onto the image, not the image onto the mask.** The old
   direction interpolated intensities (order=3) and zero-padded everything
   outside the source field of view; vertebrae landing in that padding got
   median=0, MAD=0 and therefore thousands of fake "outlier voxels"
   (L1=5009, T12=2441 in the old outputs).
3. **The untrained autoencoder is gone.** MONAI's ``AutoEncoder`` with random
   weights has learned nothing about normal anatomy, so its reconstruction error
   tracks region size and contrast. Its ranking was reported as "нейросеть нашла
   1659 аномальных вокселей в T11"; that number was noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import labels as L
from ..analysis import screen_region
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import load_canonical, resample_mask_to, write_json


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    picks = ctx.stage_data("ingest").get("picks", {})
    fatsat = picks.get("FATSAT_BEST")
    if not fatsat:
        raise SkipStage(
            "no fat-suppressed sequence in this study — bone-marrow signal is NOT "
            "assessable (a bright-voxel count on plain T2 cannot distinguish oedema "
            "from normal fatty marrow)"
        )

    spineps = ctx.stage_data("spineps")
    instance_masks = [Path(p) for p in spineps.get("instance_masks", [])]
    semantic_masks = [Path(p) for p in spineps.get("semantic_masks", [])]
    if not instance_masks:
        raise SkipStage("no SPINEPS vertebra instance mask")

    # If the fat suppression could not be confirmed, bright signal here may simply
    # be fat. The measurement still runs — the numbers are what they are — but it
    # must not be read as anything oedema-related, and that has to travel with it.
    qc = ctx.stage_data("fatsat_qc")
    suppression_ok = qc.get("suppression_effective") if qc else None
    suppression_note: list[str] = []
    if suppression_ok is False:
        suppression_note.append(
            "FAT SUPPRESSION NOT CONFIRMED on this series (" + str(qc.get("verdict")) +
            "). Bright signal below may be fat rather than fluid; do not read these "
            "numbers as oedema.")
    elif suppression_ok is None:
        suppression_note.append(
            "Fat suppression was not checked (the fatsat_qc stage did not run), so it is "
            "assumed rather than verified.")

    img = load_canonical(fatsat)
    image = np.asarray(img.get_fdata(), dtype=float)

    # Prefer masks that `register` already moved into this series' space with a
    # motion-corrected transform; fall back to header-based resampling.
    registered = ctx.masks_in_space("fatsat")
    if registered:
        instance_masks = [registered["instance_mask"]]
        semantic_masks = [registered["semantic_mask"]] if registered.get("semantic_mask") else []

    inst_img = resample_mask_to(load_canonical(instance_masks[0]), img)
    inst = np.asarray(inst_img.get_fdata()).astype(np.int32)

    body_mask = None
    corpus_label = None
    if semantic_masks:
        sem = np.asarray(resample_mask_to(load_canonical(semantic_masks[0]), img)
                         .get_fdata()).astype(np.int32)
        corpus_label = L.resolve_corpus_label(sem)
        if corpus_label is not None:
            body_mask = sem == corpus_label

    nonzero = image[image > 0]
    reference_median = float(np.percentile(nonzero, 5)) if nonzero.size else None

    screens = []
    for label_id in sorted(int(v) for v in np.unique(inst)):
        if not L.is_vertebra(label_id):
            continue
        mask = inst == label_id
        if body_mask is not None:
            restricted = mask & body_mask
            if int(restricted.sum()) >= 200:
                mask = restricted
        result = screen_region(
            image, mask,
            label=label_id,
            name=L.vertebra_name(label_id),
            z_threshold=cfg.marrow_robust_z,
            min_coverage=0.6,
            reference_median=reference_median,
        )
        if result is not None:
            screens.append(result)

    measured = [s for s in screens if s.excluded_reason is None]
    excluded = [s for s in screens if s.excluded_reason is not None]
    if not measured:
        return StageResult(
            name="marrow", status=Status.SKIPPED, evidence=Evidence.HEURISTIC,
            reason=("no vertebra had sufficient coverage on the fat-suppressed series "
                    f"({picks.get('FATSAT_PLANE')} plane): nothing measurable"),
            data={"excluded": [s.to_dict() for s in excluded],
                  "sequence": fatsat, "plane": picks.get("FATSAT_PLANE")},
        )

    # Relative brightness: each body's median against the median of all measured
    # bodies. This is closer to how marrow signal is actually read (one level
    # against its neighbours) than an absolute intensity, which is meaningless in
    # arbitrary MR units.
    medians = np.array([s.median for s in measured], dtype=float)
    cohort_median = float(np.median(medians))
    ranked = []
    for s in sorted(measured, key=lambda r: -r.outlier_fraction):
        row = s.to_dict()
        row["relative_median_vs_cohort"] = (
            round(float(s.median / cohort_median), 3) if cohort_median > 0 else None
        )
        ranked.append(row)

    payload = {
        "sequence_used": fatsat,
        "sequence_label": picks.get("FATSAT_LABEL"),
        "plane": picks.get("FATSAT_PLANE"),
        "corpus_label_used": corpus_label,
        "restricted_to_vertebral_body": body_mask is not None,
        "masks_motion_corrected": bool(registered and registered.get("applied")),
        "cohort_median_intensity": round(cohort_median, 3),
        "z_threshold": cfg.marrow_robust_z,
        "n_measured": len(measured),
        "regions": ranked,
        "excluded": [s.to_dict() for s in excluded],
        "top_candidates": [r["name"] for r in ranked[:3]
                           if r["outlier_voxels"] >= cfg.marrow_min_outlier_voxels],
        "fat_suppression_verified": suppression_ok,
        "interpretation_limits": suppression_note + [
            "A modified z-score inside one vertebra measures internal signal "
            "heterogeneity, not oedema. Degenerative endplate changes, haemangiomas, "
            "vessels and coil shading all raise it.",
            "Counts are comparable between vertebrae only where in-field coverage is "
            "similar; see in_fov_coverage per region.",
            "A radiologist has to look at the images. This is a triage order, nothing more.",
        ],
    }
    write_json(cfg.intermediate_dir / "marrow.json", payload)
    status = Status.OK if len(measured) >= 3 else Status.PARTIAL
    return StageResult(
        name="marrow", status=status, evidence=Evidence.HEURISTIC,
        reason=None if status is Status.OK else f"only {len(measured)} vertebrae measurable",
        data=payload,
    )
