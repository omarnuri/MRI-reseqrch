"""Stage `seg_sct`: spinal cord and canal masks from Spinal Cord Toolbox.

Why a fourth segmenter, when SPINEPS and TotalSpineSeg already produce cord and
canal masks: these two models are what the *calibrated* cervical measurements were
built on. `sct_detect_compression` and `sct_compute_ascor` carry thresholds fitted
to cord and canal areas measured this way, so feeding them a mask from a different
segmenter would compare our numbers against someone else's cut-off — the exact
mistake the evidence levels in this project exist to prevent.

The contrast-agnostic cord model is also the only tool in the stack whose published
validation covers the thoracic spine and several vendors, on pathological cohorts
rather than healthy volunteers.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ..envsetup import sct_binary
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..runlog import event, get_logger, log_command
from ..utils import child_env, clean_reason, run_tool

log = get_logger(__name__)

#: Which SCT model produces which mask. The names are `sct_deepseg` task names.
MODELS = {"cord": "spinalcord", "canal": "sc_canal_t2"}

#: Sequence roles worth segmenting, best first. The sagittal T2 is the anatomical
#: reference every other stage already uses; the axial T2 is what the compression
#: model's own training data looked like; the fat-suppressed 3D volume is included
#: because the aSCOR thresholds were derived on 3D T2w cervical acquisitions.
TARGETS = ("T2_SAG", "T2_AX", "FATSAT_BEST")


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    executable = sct_binary("sct_deepseg", cfg.work_dir, cfg.cache_dir)
    if executable is None:
        raise SkipStage(
            "Spinal Cord Toolbox is not installed — run `python -m spinelab setup "
            "--sct` (a ~3 GB install; the cervical canal measurements need it)")

    out_dir = cfg.intermediate_dir / "sct"
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes = {role: ctx.sequence(role) for role in TARGETS}
    volumes = {role: path for role, path in volumes.items() if path}
    if not volumes:
        raise SkipStage("no T2 volume to segment")

    results: dict[str, dict] = {}
    failures: list[str] = []
    for role, path in volumes.items():
        results[role] = {}
        for kind, model in MODELS.items():
            produced, note = _segment(executable, model, path, out_dir / role / kind, cfg)
            results[role][kind] = str(produced) if produced else None
            if produced is None:
                failures.append(f"{role}/{kind}: {note}")
                log.warning("SCT %s on %s failed: %s", model, role, note)
            else:
                log.info("SCT %s on %s -> %s", model, role, produced.name)
    event("sct_masks", produced={r: {k: bool(v) for k, v in m.items()}
                                 for r, m in results.items()})

    any_mask = any(v for masks in results.values() for v in masks.values())
    if not any_mask:
        return StageResult(name="seg_sct", status=Status.FAILED, evidence=Evidence.MODEL,
                           reason="; ".join(failures)[:400] or "no masks produced")
    return StageResult(
        name="seg_sct",
        status=Status.PARTIAL if failures else Status.OK,
        evidence=Evidence.MODEL,
        reason="; ".join(failures)[:400] or None,
        data={"masks": results, "models": MODELS, "output_dir": str(out_dir),
              "sct_binary": executable},
        artifacts=[v for masks in results.values() for v in masks.values() if v],
    )


def _segment(executable: str, model: str, image: str, out_dir: Path, cfg
             ) -> tuple[Path | None, str]:
    """Run one deepseg model, tolerating both argument spellings SCT has used.

    `sct_deepseg <task> -i … -o …` is the 7.x form; `sct_deepseg -task <task> …`
    is the older one. Trying both and reporting which worked costs one failed
    process and removes a whole class of version-pinning guesswork.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{model}.nii.gz"
    tool_log = out_dir / "sct.log"
    forms = (
        [executable, model, "-i", str(image), "-o", str(output)],
        [executable, "-task", model, "-i", str(image), "-o", str(output)],
    )
    last = "not attempted"
    for cmd in forms:
        t0 = time.time()
        try:
            proc = run_tool(cmd, log_path=tool_log, timeout=cfg.timeout_sct_s,
                            env=child_env())
        except subprocess.TimeoutExpired:
            return None, f"timeout after {cfg.timeout_sct_s}s"
        log_command("seg_sct", cmd, proc, seconds=time.time() - t0)
        if output.exists():
            return output, "ok"
        # SCT sometimes appends its own suffix; accept whatever single volume
        # appeared rather than insisting on the name we asked for.
        produced = sorted(out_dir.glob("*.nii.gz"))
        if produced:
            return produced[0], "ok (renamed by the tool)"
        last = clean_reason(proc.stdout or proc.stderr)
    return None, last[:200]
