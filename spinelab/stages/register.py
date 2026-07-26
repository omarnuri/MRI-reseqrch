"""Stage `register`: bring the SPINEPS masks into the fat-suppressed series' space.

This is the stage that decides whether the facet question can be answered at all
with this study. The masks are computed on the sagittal T2; the only
fat-suppressed series here is coronal. Two things have to be right before a
left/right comparison of a 2-4 mm joint means anything:

1. **Alignment.** Header geometry puts both series in the same patient frame, but
   inter-series motion of a few millimetres is normal and, at facet scale, moves
   the region of interest off the joint. A rigid refinement fixes that; the
   correction is reported so we can see whether it mattered.
2. **Coverage.** A coronal slab covers left and right well but is thin in the
   anterior-posterior direction, so the posterior elements may simply lie outside
   it. Registration cannot fix missing data. This stage measures the coverage of
   each side's region *after* alignment and reports it, so `posterior` can refuse
   the comparison when the two sides are not equally in-field.
"""

from __future__ import annotations

from pathlib import Path

from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..registration import apply_to_label_volume, rigid_register
from ..utils import write_json


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    picks = ctx.stage_data("ingest").get("picks", {})
    fatsat = picks.get("FATSAT_BEST")
    t2_sag = picks.get("T2_SAG")
    if not fatsat:
        raise SkipStage("no fat-suppressed series — nothing to register onto")
    if not t2_sag:
        raise SkipStage("no sagittal T2 — masks live in its space")
    if Path(fatsat).resolve() == Path(t2_sag).resolve():
        raise SkipStage("the fat-suppressed series is the mask space already")

    spineps = ctx.stage_data("spineps")
    instance_masks = spineps.get("instance_masks") or []
    semantic_masks = spineps.get("semantic_masks") or []
    if not instance_masks:
        raise SkipStage("no SPINEPS masks to move")

    try:
        import SimpleITK  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SkipStage(f"SimpleITK not installed ({exc}); "
                        "header-based alignment will be used instead") from exc

    out_dir = cfg.intermediate_dir / "registered"
    out_dir.mkdir(parents=True, exist_ok=True)

    transform, result = rigid_register(fatsat, t2_sag)
    payload = {
        "fixed_image": fatsat,
        "fixed_plane": picks.get("FATSAT_PLANE"),
        "moving_image": t2_sag,
        "registration": result.to_dict(),
    }

    written = {}
    for key, paths in (("instance_mask", instance_masks), ("semantic_mask", semantic_masks)):
        if not paths:
            continue
        src = Path(paths[0])
        dst = out_dir / f"{key}_in_fatsat_space.nii.gz"
        written[key] = apply_to_label_volume(str(src), fatsat, transform, str(dst))

    payload.update(written)
    payload["space"] = "fatsat"
    payload["applied"] = result.applied
    payload["interpretation_limits"] = [
        "Alignment only corrects for patient motion between series. It cannot add "
        "coverage: structures outside the fat-suppressed slab stay unmeasurable.",
        "Masks were moved with nearest-neighbour interpolation; the fat-suppressed "
        "image itself is never resampled, so its intensities are untouched.",
    ]
    if not result.applied:
        payload["interpretation_limits"].append(
            f"Rigid refinement was NOT applied ({result.reason}); the masks sit where the "
            "DICOM geometry says they do.")

    write_json(cfg.intermediate_dir / "registration.json", payload)
    status = Status.OK if written else Status.PARTIAL
    reason = result.reason if not result.applied else None
    return StageResult(name="register", status=status, evidence=Evidence.MEASUREMENT,
                       reason=reason, data=payload,
                       artifacts=list(written.values()))
