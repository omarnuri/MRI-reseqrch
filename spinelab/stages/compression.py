"""Stage `compression`: the cervical canal, against published cut-offs.

Everything else in this pipeline compares the subject to himself — one side against
the other, one level against its neighbours — because there is no normative cohort.
This stage is the exception, and the only place `Evidence.CALIBRATED` is allowed:

* `sct_detect_compression` — probability of cord compression per level, from a
  logistic model fitted on cervical morphometry (Horáková 2022), AUC 0.947 on an
  independent degenerative-cervical-myelopathy cohort. Cut-offs 0.345 / 0.451.
* `sct_compute_ascor` — cord-to-canal area ratio, cut-offs 36.9 % and 49.3 %
  (Hohenhaus 2024, n=202, 3D T2w cervical — the same kind of volume as this study's
  3D SPACE).
* `sct_compute_compression` — morphometry normalised to the PAM50 template.

Coverage is the hard limit and is stated with every number: the compression model
only covers C3/C4 … C6/C7. There is no thoracic or lumbar equivalent, so on those
stations this stage refuses rather than producing an unanchored number.
"""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

from .. import labels as L
from ..analysis import (
    COMPRESSION_P_HIGH,
    COMPRESSION_P_LOW,
    classify_compression,
    disc_labels_for_sct,
)
from ..envsetup import sct_binary
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..runlog import event, get_logger, log_command
from ..utils import (
    child_env,
    clean_reason,
    load_canonical,
    parse_cli_flags,
    resample_mask_to,
    run_tool,
    write_json,
)

log = get_logger(__name__)

#: SCT disc values the compression model handles: C3/C4=4 … C6/C7=7.
COMPRESSION_LEVELS = (4, 5, 6, 7)
#: Written into the disc file. One level of context either side of what the model
#: uses, because SCT needs neighbours to place a level unambiguously.
DISC_FILE_LEVELS = (3, 4, 5, 6, 7, 8)

CITATIONS = {
    "compression_probability": ("Horáková 2022, logistic model on cervical canal and "
                               "cord morphometry; AUC 0.947 on an independent DCM cohort"),
    "ascor": ("Hohenhaus 2024, n=202, 3D T2w cervical, levels C2/3–C6/7; "
              "cut-offs 36.9 % and 49.3 %"),
    "pam50": "PAM50 normative morphometry, Spine Generic healthy volunteers",
}


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    masks = (ctx.stage_data("seg_sct").get("masks") or {})
    if not masks:
        raise SkipStage("seg_sct produced no cord or canal masks")

    role, cord, canal = _pick_pair(masks)
    if cord is None:
        raise SkipStage("no volume has both a cord and a canal mask from SCT")

    disc_file, levels_found = _write_disc_file(ctx, cord)
    if disc_file is None:
        raise SkipStage(levels_found)          # message, not a level list
    covered = sorted(set(levels_found) & set(COMPRESSION_LEVELS))
    if not covered:
        raise SkipStage(
            "this station covers no cervical level between C3/C4 and C6/C7 — the "
            "compression model and aSCOR thresholds are cervical only, and no "
            "thoracic or lumbar equivalent exists (see docs/stack-review-2026-08.md)")

    out_dir = cfg.intermediate_dir / "compression"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "sequence_role": role,
        "cord_mask": str(cord), "canal_mask": str(canal) if canal else None,
        "disc_file": str(disc_file),
        "levels_in_disc_file": sorted(levels_found),
        "levels_covered_by_the_model": covered,
        "thresholds": {"compression_p_low": COMPRESSION_P_LOW,
                       "compression_p_high": COMPRESSION_P_HIGH},
        "citations": CITATIONS,
        "interpretation_limits": [
            "Cut-offs come from external cohorts, not from this study — that is what "
            "makes these the only anchored numbers in the report. They still describe "
            "morphometry, not symptoms.",
            "Coverage is C3/C4 to C6/C7 only. Nothing here says anything about the "
            "thoracic spine, where this subject's pain is.",
        ],
    }

    detect = _run_detect(ctx, cord, disc_file, out_dir)
    payload["compression"] = detect
    payload["ascor"] = _run_ascor(ctx, cord, canal, disc_file, out_dir)
    payload["pam50_morphometry"] = _run_morphometry(ctx, cord, disc_file, out_dir)

    write_json(cfg.intermediate_dir / "compression.json", payload)
    event("compression", role=role, levels=covered,
          detected=bool(detect.get("levels")))

    produced = [k for k in ("compression", "ascor", "pam50_morphometry")
                if payload[k].get("ok")]
    if not produced:
        reasons = "; ".join(payload[k].get("reason", "") for k in
                            ("compression", "ascor", "pam50_morphometry"))
        return StageResult(name="compression", status=Status.FAILED,
                           evidence=Evidence.CALIBRATED,
                           reason=f"no SCT measurement ran: {reasons}"[:400],
                           data=payload)
    return StageResult(
        name="compression",
        status=Status.OK if len(produced) == 3 else Status.PARTIAL,
        evidence=Evidence.CALIBRATED,
        reason=None if len(produced) == 3 else f"produced: {', '.join(produced)}",
        data=payload,
        artifacts=[str(disc_file)],
    )


def _pick_pair(masks: dict) -> tuple[str | None, Path | None, Path | None]:
    """The volume with both masks, preferring the one the thresholds were built on.

    The aSCOR cut-offs come from 3D T2w acquisitions and the compression model from
    axial morphometry, so those two are tried before the sagittal stack.
    """
    for role in ("T2_AX", "FATSAT_BEST", "T2_SAG"):
        entry = masks.get(role) or {}
        if entry.get("cord"):
            return role, Path(entry["cord"]), Path(entry["canal"]) if entry.get("canal") else None
    return None, None, None


def _write_disc_file(ctx: Context, reference: Path):
    """Single-voxel disc labels on the cord mask's own grid, numbered the SCT way."""
    import numpy as np

    label_volume = ctx.stage_data("totalspineseg").get("label_volume")
    if not label_volume:
        return None, ("no TotalSpineSeg label volume, so disc levels cannot be placed "
                      "— sct_detect_compression needs one voxel per disc")
    import nibabel as nib

    ref_img = load_canonical(reference)
    discs = resample_mask_to(load_canonical(label_volume), ref_img)
    data = np.asarray(discs.get_fdata()).astype(np.int32)
    labels = disc_labels_for_sct(data, L.TSS_DISCS, keep_values=DISC_FILE_LEVELS)
    found = sorted({int(v) for v in np.unique(labels) if v})
    if not found:
        return None, ("no cervical disc labels (C2/C3–C7/T1) in the segmentation of "
                      "this station")
    path = ctx.config.intermediate_dir / "compression" / "discs.nii.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(labels, ref_img.affine, ref_img.header), str(path))
    log.info("disc file: %s with levels %s", path.name, found)
    return path, found


def _sct(ctx: Context, name: str):
    cfg = ctx.config
    return sct_binary(name, cfg.work_dir, cfg.cache_dir)


def _flags(binary: str) -> set[str]:
    """What this build of the tool actually accepts."""
    try:
        proc = subprocess.run([binary, "-h"], capture_output=True, text=True,
                              timeout=120, env=child_env())
    except Exception as exc:  # noqa: BLE001
        log.warning("%s -h failed: %s", binary, exc)
        return set()
    return parse_cli_flags((proc.stdout or "") + (proc.stderr or ""))


def _call(ctx: Context, binary: str, argv: list[str], log_path: Path) -> tuple[bool, str]:
    timeout = ctx.config.timeout_sct_s
    t0 = time.time()
    try:
        proc = run_tool(argv, log_path=log_path, timeout=timeout, env=child_env())
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    log_command("compression", argv, proc, seconds=time.time() - t0)
    if proc.returncode != 0:
        return False, clean_reason(proc.stdout or proc.stderr)[:200]
    return True, "ok"


def _run_detect(ctx: Context, cord: Path, disc_file: Path, out_dir: Path) -> dict:
    binary = _sct(ctx, "sct_detect_compression")
    if binary is None:
        return {"ok": False, "reason": "sct_detect_compression not installed"}
    flags = _flags(binary)
    for required in ("-s", "-discfile"):
        if flags and required not in flags:
            return {"ok": False,
                    "reason": f"this build of sct_detect_compression has no {required} "
                              f"(flags seen: {', '.join(sorted(flags))[:120]})"}
    output = out_dir / "compression.csv"
    ok, reason = _call(ctx, binary,
                       [binary, "-s", str(cord), "-discfile", str(disc_file),
                        "-o", str(output)], out_dir / "detect.log")
    if not ok:
        return {"ok": False, "reason": reason}
    rows = _read_csv(output)
    return {"ok": True, "csv": str(output), "levels": _levels_from_rows(rows),
            "rows": rows, "citation": CITATIONS["compression_probability"]}


def _run_ascor(ctx: Context, cord: Path, canal: Path | None, disc_file: Path,
               out_dir: Path) -> dict:
    binary = _sct(ctx, "sct_compute_ascor")
    if binary is None:
        return {"ok": False, "reason": "sct_compute_ascor not installed"}
    if canal is None:
        return {"ok": False, "reason": "no canal mask, and aSCOR is cord area / canal area"}
    flags = _flags(binary)
    # The published documentation page for this command 404s, so the arguments are
    # taken from the installed binary. Anything else would be a guess printed as a
    # calibrated number.
    argv = [binary]
    for flag, value in (("-i", cord), ("-s", cord), ("-canal", canal),
                        ("-discfile", disc_file), ("-o", out_dir / "ascor.csv")):
        if flag in flags:
            argv += [flag, str(value)]
    if len(argv) < 5:
        return {"ok": False,
                "reason": f"could not build a command from the flags this build accepts: "
                          f"{', '.join(sorted(flags))[:200]}"}
    ok, reason = _call(ctx, binary, argv, out_dir / "ascor.log")
    if not ok:
        return {"ok": False, "reason": reason}
    rows = _read_csv(out_dir / "ascor.csv")
    return {"ok": True, "csv": str(out_dir / "ascor.csv"), "rows": rows,
            "thresholds_pct": {"stenosis": 36.9, "upper": 49.3},
            "citation": CITATIONS["ascor"]}


def _run_morphometry(ctx: Context, cord: Path, disc_file: Path, out_dir: Path) -> dict:
    binary = _sct(ctx, "sct_compute_compression")
    if binary is None:
        return {"ok": False, "reason": "sct_compute_compression not installed"}
    flags = _flags(binary)
    argv = [binary]
    for flag, value in (("-i", cord), ("-s", cord), ("-discfile", disc_file),
                        ("-o", out_dir / "morphometry.csv")):
        if flag in flags:
            argv += [flag, str(value)]
    if len(argv) < 5:
        return {"ok": False, "reason": "arguments not recognised on this build"}
    ok, reason = _call(ctx, binary, argv, out_dir / "morphometry.log")
    if not ok:
        return {"ok": False, "reason": reason}
    return {"ok": True, "csv": str(out_dir / "morphometry.csv"),
            "rows": _read_csv(out_dir / "morphometry.csv"),
            "citation": CITATIONS["pam50"]}


def _read_csv(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    except Exception as exc:  # noqa: BLE001 — an unreadable result is reported, not raised
        log.warning("cannot read %s: %s", path, exc)
        return []


def _levels_from_rows(rows: list[dict]) -> list[dict]:
    """Level, probability and category, without assuming SCT's column names.

    The column layout is read from the file rather than hard-coded: it has changed
    across SCT versions, and a KeyError here would throw away a measurement that
    had already been computed. Unrecognised columns are kept in `rows` verbatim.
    """
    out = []
    for row in rows:
        probability, level = None, None
        for key, value in row.items():
            name = (key or "").strip().lower()
            if probability is None and "prob" in name:
                probability = _as_float(value)
            if level is None and ("level" in name or "disc" in name):
                level = value
        if probability is None and level is None:
            continue
        out.append({"level": level, "probability": probability,
                    "category": classify_compression(probability)})
    return out


def _as_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
