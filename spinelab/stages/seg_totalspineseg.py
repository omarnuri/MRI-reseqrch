"""Stage `totalspineseg`: cord, canal and intervertebral discs.

Fixes carried over from the old Cell 4: the input copy lives outside the output
directory (so a successful run is no longer reported as "failed" because the
only ``*.nii.gz`` found was the input), and the model data directory is the
Drive-backed cache via ``TOTALSPINESEG_DATA``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .. import labels as L
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..runlog import event, get_logger, log_command
from ..utils import child_env, clean_reason, find_outputs, run_tool, tool_path

log = get_logger(__name__)


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    t2_sag = ctx.require_sequence("T2_SAG")

    executable = tool_path("totalspineseg")
    if executable is None:
        raise SkipStage("totalspineseg CLI not found next to this interpreter or on "
                        "PATH (pip install totalspineseg)")

    out_dir = cfg.intermediate_dir / "totalspineseg"
    out_dir.mkdir(parents=True, exist_ok=True)
    # A sibling of the output directory, not inside it: totalspineseg writes its
    # results into the output tree, and an input copy living there would be found
    # by the output search. `/tmp/spinelab_tss_input` was hardcoded here, which on
    # Windows lands in C:\tmp and is shared between every run on the machine.
    in_dir = cfg.intermediate_dir / "totalspineseg_input"
    if in_dir.exists():
        shutil.rmtree(in_dir, ignore_errors=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(t2_sag, in_dir / Path(t2_sag).name)

    env = child_env()
    env.setdefault("TOTALSPINESEG_DATA", str(cfg.weights_dir / "totalspineseg"))
    Path(env["TOTALSPINESEG_DATA"]).mkdir(parents=True, exist_ok=True)

    # `--device` defaults to cuda-if-available, which is right on Colab but hides
    # the choice; passing it makes the run reproducible and lets --device cpu work.
    device = cfg.resolve_device()
    cmd = [executable, str(in_dir), str(out_dir), "--iso", "--device", device]
    log.info("data dir: %s, device: %s", env["TOTALSPINESEG_DATA"], device)
    import time as _time
    _t0 = _time.time()
    tool_log = out_dir.parent / "totalspineseg_tool.log"
    log.info("streaming totalspineseg output to %s", tool_log)
    try:
        proc = run_tool(cmd, log_path=tool_log, timeout=cfg.timeout_tss_s, env=env)
        log_command("totalspineseg", cmd, proc, seconds=_time.time() - _t0)
    except subprocess.TimeoutExpired:
        log.error("totalspineseg timed out after %ss", cfg.timeout_tss_s)
        event("problem", stage="totalspineseg", detail=f"timeout {cfg.timeout_tss_s}s")
        return StageResult(name="totalspineseg", status=Status.FAILED,
                           evidence=Evidence.MODEL,
                           reason=f"timeout after {cfg.timeout_tss_s}s")

    outputs = find_outputs(out_dir, "*.nii.gz")
    log.info("outputs: %d volumes (%d from step2)", len(outputs),
             sum(1 for p in outputs if "step2" in str(p).lower()))
    event("tss_outputs", n=len(outputs), files=[str(p) for p in outputs[:8]])
    if not outputs:
        return StageResult(
            name="totalspineseg", status=Status.FAILED, evidence=Evidence.MODEL,
            reason=f"rc={proc.returncode}: {clean_reason(proc.stderr or proc.stdout)}",
            data={"stdout_tail": (proc.stdout or "")[-1500:],
                  "stderr_tail": (proc.stderr or "")[-1500:]},
        )

    label_volume, label_note = _pick_label_volume(outputs)
    # Files on disk are not the same thing as a finished run. This reported OK on a
    # run that exited rc=1 partway through, and the missing labels then showed up two
    # stages later as "no canal segmentation" and "no volume containing a cord label"
    # — symptoms of a failure that had already been recorded and ignored here.
    partial = proc.returncode != 0 or label_volume is None
    reason = None
    if proc.returncode != 0:
        reason = (f"totalspineseg exited rc={proc.returncode} after writing "
                  f"{len(outputs)} volume(s) — the output is incomplete: "
                  f"{clean_reason(proc.stdout)}")
    elif label_volume is None:
        reason = f"no usable label volume in the output — {label_note}"
    log.info("label volume: %s (%s)", label_volume, label_note)
    return StageResult(
        name="totalspineseg",
        status=Status.PARTIAL if partial else Status.OK,
        evidence=Evidence.MODEL, reason=reason,
        data={
            "outputs": [str(p) for p in outputs],
            # The one file every consumer should read. Naming it here is the point:
            # three stages each guessed at this from the file names and each guessed
            # differently, and two of them landed on a soft probability map.
            "label_volume": str(label_volume) if label_volume else None,
            "label_volume_note": label_note,
            "output_dir": str(out_dir),
            "device": device,
        },
        artifacts=[str(label_volume)] if label_volume else [str(p) for p in outputs[:10]],
    )


#: Where the discrete label volumes live, best first. `step2_output` is the final
#: result; `step1_output` is the coarse first pass and only a fallback.
#:
#: Everything else in the tree looks like a label volume and is not one:
#: `step2_input` is a binary mask (2 distinct values), and `step1_canal` and
#: `step1_cord` are soft maps — on the real study they held 8999 and 6104 distinct
#: values. Consumers that matched on "step2" or "cord" in the path picked those, which
#: is why the disc stage found no disc labels and the cord agreement came out at Dice
#: 0.12 against a probability map.
LABEL_DIRS = ("step2_output", "step1_output")
#: A discrete label volume for this task has tens of values, not thousands and not two.
MIN_LABELS = 5
MAX_LABELS = 200


def _pick_label_volume(outputs) -> tuple[Path | None, str]:
    """The discrete label volume, chosen by directory and verified by its contents."""
    import numpy as np

    for directory in LABEL_DIRS:
        for path in [p for p in outputs if p.parent.name.lower() == directory]:
            try:
                import nibabel as nib

                values = np.unique(np.asarray(nib.load(str(path)).dataobj))
            except Exception as exc:  # noqa: BLE001
                log.warning("cannot read %s: %s", path, exc)
                continue
            n = int(values.size)
            if MIN_LABELS <= n <= MAX_LABELS and float(values.max()) >= L.TSS_DISC_LABEL_MIN:
                return path, f"{directory}, {n} distinct labels"
            log.info("%s has %d distinct values, max %s — not a label volume",
                     path.name, n, values.max() if n else None)
    return None, (f"none of {', '.join(LABEL_DIRS)} contained a discrete label volume "
                  f"({MIN_LABELS}-{MAX_LABELS} values reaching "
                  f"{L.TSS_DISC_LABEL_MIN})")
