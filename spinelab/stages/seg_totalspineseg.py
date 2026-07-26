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
from ..utils import clean_reason, find_outputs

log = get_logger(__name__)


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    t2_sag = ctx.require_sequence("T2_SAG")

    if shutil.which("totalspineseg") is None:
        raise SkipStage("totalspineseg CLI not on PATH (pip install totalspineseg)")

    out_dir = cfg.intermediate_dir / "totalspineseg"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path("/tmp/spinelab_tss_input")
    if in_dir.exists():
        shutil.rmtree(in_dir, ignore_errors=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(t2_sag, in_dir / Path(t2_sag).name)

    env = dict(os.environ)
    env.setdefault("TOTALSPINESEG_DATA", str(cfg.weights_dir / "totalspineseg"))
    Path(env["TOTALSPINESEG_DATA"]).mkdir(parents=True, exist_ok=True)

    cmd = ["totalspineseg", str(in_dir), str(out_dir), "--iso"]
    log.info("data dir: %s", env["TOTALSPINESEG_DATA"])
    import time as _time
    _t0 = _time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=cfg.timeout_tss_s, env=env)
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
    return StageResult(
        name="totalspineseg", status=Status.OK, evidence=Evidence.MODEL,
        data={
            "outputs": [str(p) for p in outputs],
            "step2_outputs": [str(p) for p in step2],
            "output_dir": str(out_dir),
        },
        artifacts=[str(p) for p in (step2 or outputs)[:10]],
    )
