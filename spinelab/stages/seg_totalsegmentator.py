"""Stage `totalsegmentator`: paraspinal muscles and ribs (TotalSegmentator MRI).

Removed from the old Cell 5:

* the ``pip install --upgrade dynamic-network-architectures`` mid-pipeline plus
  ``importlib.reload(site)`` and hand-deletion of ``sys.modules`` entries. Mutating
  the environment halfway through a run means the stages before and after it ran
  against different library versions;
* the loop that reassigned ``nnUNet_results`` on every already-imported module.
  Setting the environment variables before the tool's process starts does the
  same thing without reaching into other packages' internals;
* the manual ``wget`` of the weight zips. TotalSegmentator's own
  ``download_pretrained_weights`` resolves the same URLs
  (``v2.5.0-weights/Dataset85{0,1}_TotalSegMRI_...``) and validates the archive.

Inference runs in a subprocess: the nnU-Net predictor leaks GPU memory into the
parent kernel otherwise, which is what made later cells hit OOM.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..runlog import event, get_logger, log_command
from ..utils import child_env, clean_reason, run_tool

log = get_logger(__name__)

#: TotalSegmentator MR tasks used by `task="total_mr"` (part1 organs, part2 muscles).
MR_TASK_IDS = (850, 851)

MUSCLE_BASES = ("autochthon", "erector_spinae", "multifidus", "iliocostalis",
                "longissimus", "spinalis", "psoas_major", "quadratus_lumborum")


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    t2_sag = ctx.require_sequence("T2_SAG")

    try:
        import totalsegmentator  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SkipStage(f"TotalSegmentator not importable: {exc}") from exc

    out_dir = cfg.intermediate_dir / "totalsegmentator"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = child_env()
    home = Path(env.get("TOTALSEG_HOME_DIR", cfg.weights_dir / "totalsegmentator"))
    results = home / "nnunet" / "results"
    results.mkdir(parents=True, exist_ok=True)
    env["TOTALSEG_HOME_DIR"] = str(home)
    env["nnUNet_results"] = str(results)

    log.info("weights dir: %s", results)
    weights = _ensure_weights(results, env)
    log.info("weights per task: %s", weights)
    event("ts_weights", weights=weights, dir=str(results))
    missing = [tid for tid, ok in weights.items() if not ok]
    if missing:
        return StageResult(
            name="totalsegmentator", status=Status.FAILED, evidence=Evidence.MODEL,
            reason=f"weights for MR task(s) {missing} unavailable — muscle analysis cannot run",
            data={"weights": weights, "weights_dir": str(results)},
        )

    # `device` is passed explicitly: the default is "gpu", and on a host without
    # CUDA that fails inside nnU-Net rather than falling back.
    device = "cpu" if cfg.resolve_device() == "cpu" else "gpu"
    log.info("device: %s", device)
    script = (
        "from totalsegmentator.python_api import totalsegmentator\n"
        f"totalsegmentator(input={str(t2_sag)!r}, output={str(out_dir)!r}, "
        f"task='total_mr', ml=False, verbose=False, device={device!r})\n"
    )
    import time as _time
    _t0 = _time.time()
    tool_log = out_dir.parent / "totalsegmentator_tool.log"
    log.info("streaming totalsegmentator output to %s", tool_log)
    try:
        proc = run_tool([sys.executable, "-c", script], log_path=tool_log,
                        timeout=cfg.timeout_ts_s, env=env)
        log_command("totalsegmentator", ["python", "-c", "totalsegmentator(task=total_mr)"],
                    proc, seconds=_time.time() - _t0)
    except subprocess.TimeoutExpired:
        log.error("totalsegmentator timed out after %ss", cfg.timeout_ts_s)
        event("problem", stage="totalsegmentator", detail=f"timeout {cfg.timeout_ts_s}s")
        return StageResult(name="totalsegmentator", status=Status.FAILED,
                           evidence=Evidence.MODEL,
                           reason=f"timeout after {cfg.timeout_ts_s}s")

    outputs = sorted(out_dir.rglob("*.nii.gz"))
    if proc.returncode != 0 and not outputs:
        return StageResult(
            name="totalsegmentator", status=Status.FAILED, evidence=Evidence.MODEL,
            reason=clean_reason(proc.stderr or proc.stdout),
            data={"stderr_tail": (proc.stderr or "")[-1500:]},
        )

    log.info("outputs: %d masks in %s", len(outputs), out_dir)
    muscle_pairs = {}
    for base in MUSCLE_BASES:
        left = next((p for p in outputs if f"{base}_left" in p.name.lower()), None)
        right = next((p for p in outputs if f"{base}_right" in p.name.lower()), None)
        if left and right:
            muscle_pairs[base] = {"left": str(left), "right": str(right)}

    log.info("muscle pairs found: %s", sorted(muscle_pairs))
    event("ts_outputs", n_masks=len(outputs), muscle_pairs=sorted(muscle_pairs))
    status = Status.OK if muscle_pairs else Status.PARTIAL
    return StageResult(
        name="totalsegmentator", status=status, evidence=Evidence.MODEL,
        reason=None if muscle_pairs else "no left/right muscle pairs in the output set",
        data={
            "outputs": [str(p) for p in outputs],
            "muscle_pairs": muscle_pairs,
            "reference_image": str(t2_sag),
            "weights_dir": str(results),
            "device": device,
        },
        artifacts=[v["left"] for v in muscle_pairs.values()][:5],
    )


def _ensure_weights(results_dir: Path, env: dict) -> dict[int, bool]:
    """Download the MR weights through TotalSegmentator's own resolver.

    Completeness is checked the way nnU-Net loads a model: a checkpoint plus a
    plans/dataset json. A half-downloaded directory is deleted rather than
    reused, because nnU-Net's failure mode on a truncated archive is an opaque
    stack trace hundreds of lines into inference.
    """
    out = {}
    for task_id in MR_TASK_IDS:
        target = _dataset_dir(results_dir, task_id)
        if target and _complete(target):
            out[task_id] = True
            continue
        if target and target.exists():
            import shutil

            shutil.rmtree(target, ignore_errors=True)
        script = (
            "from totalsegmentator.libs import download_pretrained_weights\n"
            f"download_pretrained_weights({task_id})\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, env=env, timeout=3600)
        target = _dataset_dir(results_dir, task_id)
        out[task_id] = bool(target and _complete(target)) and proc.returncode == 0
    return out


def _dataset_dir(results_dir: Path, task_id: int) -> Path | None:
    for path in sorted(results_dir.glob(f"Dataset{task_id}_*")):
        if path.is_dir():
            return path
    return None


def _complete(path: Path) -> bool:
    has_ckpt = any(path.rglob("checkpoint_*.pth"))
    has_plans = any(path.rglob("plans.json")) or any(path.rglob("dataset.json"))
    return has_ckpt and has_plans


def label_index(results_dir: Path, task_id: int) -> dict:
    """Class-index -> structure-name map, read from the model's dataset.json."""
    target = _dataset_dir(results_dir, task_id)
    if not target:
        return {}
    for candidate in target.rglob("dataset.json"):
        try:
            with open(candidate, encoding="utf-8") as fh:
                return json.load(fh).get("labels", {})
        except Exception:
            continue
    return {}
