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

    step2 = [p for p in outputs if "step2" in str(p).lower()]
    # Files on disk are not the same thing as a finished run. This reported OK on a
    # run that exited rc=1 partway through, and the missing labels then showed up two
    # stages later as "no canal segmentation" and "no volume containing a cord label"
    # — symptoms of a failure that had already been recorded and ignored here.
    partial = proc.returncode != 0 or not step2
    reason = None
    if proc.returncode != 0:
        reason = (f"totalspineseg exited rc={proc.returncode} after writing "
                  f"{len(outputs)} volume(s) — the output is incomplete: "
                  f"{clean_reason(proc.stdout)}")
    elif not step2:
        reason = ("no step2 output — only the coarse first-pass labels are available, "
                  "so cord, canal and disc labels may be missing")
    return StageResult(
        name="totalspineseg",
        status=Status.PARTIAL if partial else Status.OK,
        evidence=Evidence.MODEL, reason=reason,
        data={
            "outputs": [str(p) for p in outputs],
            "step2_outputs": [str(p) for p in step2],
            "output_dir": str(out_dir),
            "device": device,
        },
        artifacts=[str(p) for p in (step2 or outputs)[:10]],
    )
