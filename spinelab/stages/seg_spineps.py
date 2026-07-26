"""Stage `spineps`: vertebra instances + semantic subregions (SPINEPS >= 2.0).

What changed relative to the old Cell 3:

* No hand-rolled download into ``/usr/local/lib/python3.12/dist-packages/
  spineps/models``. The URLs the old cell used are the same ones SPINEPS resolves
  itself (``spineps/utils/auto_download.py``), so the download was redundant —
  but the hard-coded interpreter path breaks on any other Python build, the
  re-flattening of the extracted folders can desync it from what
  ``check_available_models`` expects, and nothing was cached between sessions.
* Weights go to ``SPINEPS_SEGMENTOR_MODELS`` (a Drive-backed cache), so a rerun
  after a Colab disconnect does not re-download several GB.
* The labeling model (``t2w_labeling``, i.e. VERIDAH) is enabled. The old run
  never used it, yet the whole case is stated in terms of a specific level
  ("T11"): on a thoracic-only field of view, vertebra enumeration is precisely
  what goes wrong, and this is the model that guards against it.
* The input copy lives in its own directory, so the input file can never be
  mistaken for a model output.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import clean_reason, find_outputs


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    t2_sag = ctx.require_sequence("T2_SAG")

    if shutil.which("spineps") is None:
        raise SkipStage("spineps CLI not on PATH (pip install SPINEPS>=2.0.0)")

    out_dir = cfg.intermediate_dir / "spineps"
    # `spineps sample` writes its derivatives next to the input file, so the input
    # gets its own subdirectory inside the stage output tree.
    in_dir = out_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    work_input = in_dir / Path(t2_sag).name
    if not work_input.exists():
        shutil.copy(t2_sag, work_input)

    models_dir = os.environ.get("SPINEPS_SEGMENTOR_MODELS", str(cfg.weights_dir / "spineps"))
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "spineps", "sample",
        "-ignore_bids_filter",
        "-ignore_inference_compatibility",
        "-i", str(work_input),
        "-der_name", "derivatives_seg",
        "-model_semantic", "t2w",
        "-model_instance", "instance",
        "-model_labeling", "t2w_labeling",
    ]
    env = dict(os.environ, SPINEPS_SEGMENTOR_MODELS=models_dir)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=cfg.timeout_spineps_s, env=env)
    except subprocess.TimeoutExpired:
        return StageResult(name="spineps", status=Status.FAILED, evidence=Evidence.MODEL,
                           reason=f"timeout after {cfg.timeout_spineps_s}s")

    # Outputs land under <in_dir>/derivatives_seg/... — search the whole stage tree
    # and drop the input copy explicitly (name matching alone previously let the
    # input file count as an output).
    instance_masks = [p for p in find_outputs(out_dir, "*vert_msk*.nii.gz", exclude_dirs=())
                      if p != work_input]
    semantic_masks = [p for p in find_outputs(out_dir, "*spine_msk*.nii.gz", exclude_dirs=())
                      if p != work_input]
    centroids = find_outputs(out_dir, "*ctd*.json", exclude_dirs=()) \
        + find_outputs(out_dir, "*snp*.json", exclude_dirs=())

    if not instance_masks and not semantic_masks:
        reason = clean_reason(proc.stderr or proc.stdout, "no masks written")
        return StageResult(name="spineps", status=Status.FAILED, evidence=Evidence.MODEL,
                           reason=f"rc={proc.returncode}: {reason}",
                           data={"stdout_tail": (proc.stdout or "")[-1500:],
                                 "stderr_tail": (proc.stderr or "")[-1500:]})

    status = Status.OK if (instance_masks and semantic_masks) else Status.PARTIAL
    reason = None
    if status is Status.PARTIAL:
        missing = "semantic (posterior elements)" if not semantic_masks else "instance"
        reason = f"{missing} mask missing — dependent stages will report reduced scope"

    return StageResult(
        name="spineps", status=status, evidence=Evidence.MODEL, reason=reason,
        data={
            "instance_masks": [str(p) for p in instance_masks],
            "semantic_masks": [str(p) for p in semantic_masks],
            "centroid_files": [str(p) for p in centroids],
            "models_dir": models_dir,
            "version": _spineps_version(),
        },
        artifacts=[str(p) for p in instance_masks + semantic_masks],
    )


def _spineps_version() -> str | None:
    try:
        out = subprocess.run([sys.executable, "-c",
                              "import spineps,sys;sys.stdout.write(getattr(spineps,'__version__','?'))"],
                             capture_output=True, text=True, timeout=120)
        return (out.stdout or "").strip() or None
    except Exception:
        return None
