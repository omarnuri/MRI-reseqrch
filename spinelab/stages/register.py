"""Stage `register`: bring the SPINEPS masks into every other series' space.

The masks are computed on the sagittal T2, but the two series that carry the
information this case needs are elsewhere: the fat-suppressed signal is on a
coronal STIR, and the resolution that can resolve a facet joint is on an axial T2.
Both are targets here.

Two things have to be right before a left/right comparison of a 2-4 mm joint means
anything:

1. **Alignment.** Header geometry puts all series in the same patient frame, but
   inter-series motion of a few millimetres is normal and, at facet scale, moves
   the region of interest off the joint. A rigid refinement fixes that; the
   correction is reported so we can see whether it mattered.
2. **Coverage.** Registration cannot add data. A coronal slab is thin
   anterior-posteriorly, so the posterior elements may lie outside it; consumers
   measure per-side coverage and refuse the comparison when the sides differ.
"""

from __future__ import annotations

from pathlib import Path

from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..registration import apply_to_label_volume, rigid_register
from ..runlog import event, get_logger
from ..utils import write_json

log = get_logger(__name__)

#: target name -> sequence pick key
TARGETS = {"fatsat": "FATSAT_BEST", "axial": "T2_AX"}


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    picks = ctx.stage_data("ingest").get("picks", {})
    t2_sag = picks.get("T2_SAG")
    if not t2_sag:
        raise SkipStage("no sagittal T2 — masks live in its space")

    spineps = ctx.stage_data("spineps")
    instance_masks = spineps.get("instance_masks") or []
    semantic_masks = spineps.get("semantic_masks") or []
    if not instance_masks:
        raise SkipStage("no SPINEPS masks to move")

    wanted = {name: picks.get(key) for name, key in TARGETS.items() if picks.get(key)}
    wanted = {name: path for name, path in wanted.items()
              if Path(path).resolve() != Path(t2_sag).resolve()}
    if not wanted:
        raise SkipStage("no other series to register onto (masks already in the only space)")

    try:
        import SimpleITK  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SkipStage(f"SimpleITK not installed ({exc}); header-based alignment "
                        "will be used instead") from exc

    out_dir = cfg.intermediate_dir / "registered"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[str, dict] = {}
    for name, fixed_image in sorted(wanted.items()):
        log.info("registering masks onto %s target: %s", name, Path(fixed_image).name)
        transform, result = rigid_register(fixed_image, t2_sag)
        log.info("  applied=%s translation=%.2f mm rotation=%.2f deg metric %s -> %s%s",
                 result.applied, result.translation_magnitude_mm, result.rotation_deg,
                 result.metric_before, result.metric_after,
                 f" ({result.reason})" if result.reason else "")
        event("registration", target=name, **result.to_dict())
        if not result.applied:
            event("problem", stage="register", detail=f"{name}: {result.reason}")
        entry = {
            "image": fixed_image,
            "plane": picks.get("FATSAT_PLANE") if name == "fatsat" else "axial",
            "applied": result.applied,
            "registration": result.to_dict(),
        }
        for key, paths in (("instance_mask", instance_masks), ("semantic_mask", semantic_masks)):
            if not paths:
                continue
            dst = out_dir / f"{key}_in_{name}_space.nii.gz"
            entry[key] = apply_to_label_volume(str(Path(paths[0])), fixed_image, transform,
                                              str(dst))
        targets[name] = entry

    payload = {
        "moving_image": t2_sag,
        "targets": targets,
        "interpretation_limits": [
            "Alignment only corrects for patient motion between series. It cannot add "
            "coverage: structures outside a series' field of view stay unmeasurable.",
            "Masks were moved with nearest-neighbour interpolation; the target images are "
            "never resampled, so their intensities are untouched.",
        ],
    }
    rejected = [name for name, t in targets.items() if not t["applied"]]
    if rejected:
        payload["interpretation_limits"].append(
            f"Rigid refinement was NOT applied for: {', '.join(rejected)}. Those masks sit "
            "where the DICOM geometry says they do.")

    write_json(cfg.intermediate_dir / "registration.json", payload)
    reasons = "; ".join(f"{n}: {targets[n]['registration'].get('reason')}" for n in rejected)
    return StageResult(
        name="register",
        status=Status.OK if not rejected else Status.PARTIAL,
        evidence=Evidence.MEASUREMENT,
        reason=reasons or None,
        data=payload,
        artifacts=[t[k] for t in targets.values()
                   for k in ("instance_mask", "semantic_mask") if t.get(k)],
    )
