"""Run logging: a human log, a machine-readable event stream, and a digest.

The point is diagnosis at a distance. A run happens on someone else's Colab
session; when something goes wrong the only thing that travels back is text. So
every step writes two records:

* ``results/run.log`` — timestamped lines, readable top to bottom;
* ``results/run.jsonl`` — one JSON object per event, so a run can be queried
  instead of read (which stage, which command, what return code, which file was
  missing).

Both are written as the run proceeds, not at the end: a session that dies halfway
still leaves everything up to that point. `digest()` turns them into a paste-sized
summary — that is the thing to send when a run misbehaves.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LOG_FILENAME = "run.log"
EVENTS_FILENAME = "run.jsonl"

_STATE: dict[str, Any] = {"events_path": None, "handlers": [], "results_dir": None}

LOGGER_NAME = "spinelab"


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger for a module: `get_logger(__name__)`."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    short = name.split(".")[-1]
    return logging.getLogger(f"{LOGGER_NAME}.{short}")


def setup(results_dir: str | Path, *, verbose: bool = True, console: bool = False) -> Path:
    """Attach file handlers for this results directory. Safe to call repeatedly."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / LOG_FILENAME

    if _STATE["results_dir"] == str(results_dir) and _STATE["handlers"]:
        return log_path

    logger = logging.getLogger(LOGGER_NAME)
    for handler in _STATE["handlers"]:
        logger.removeHandler(handler)
        handler.close()
    _STATE["handlers"] = []

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(file_handler)
    _STATE["handlers"].append(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setLevel(logging.INFO)
        stream.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream)
        _STATE["handlers"].append(stream)

    _STATE["events_path"] = results_dir / EVENTS_FILENAME
    _STATE["results_dir"] = str(results_dir)
    logger.info("--- logging started (%s) ---", time.strftime("%Y-%m-%d %H:%M:%S"))
    return log_path


def event(kind: str, **fields: Any) -> None:
    """Append one structured event. Never raises — logging must not break a run."""
    path = _STATE.get("events_path")
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind}
    record.update(_jsonable(fields))
    get_logger().debug("event %s %s", kind, json.dumps(_short(record), ensure_ascii=False))
    if path is None:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — a failed write must not abort the pipeline
        pass


def _jsonable(obj: Any) -> Any:
    from .utils import to_jsonable

    return to_jsonable(obj)


def _short(record: dict, limit: int = 400) -> dict:
    out = {}
    for key, value in record.items():
        text = value if isinstance(value, (int, float, bool)) or value is None else str(value)
        if isinstance(text, str) and len(text) > limit:
            text = text[:limit] + "…"
        out[key] = text
    return out


def log_command(stage: str, cmd: list[str], proc, *, seconds: float | None = None) -> None:
    """Record an external tool invocation with its return code and output tails.

    Output is recorded on success as well as on failure. A tool that "worked" but
    printed a warning about missing weights is exactly the case that needs to be
    visible afterwards.
    """
    log = get_logger(stage)
    rc = getattr(proc, "returncode", None)
    stdout = (getattr(proc, "stdout", "") or "")[-4000:]
    stderr = (getattr(proc, "stderr", "") or "")[-4000:]
    level = logging.INFO if rc == 0 else logging.WARNING
    log.log(level, "command rc=%s (%.1fs): %s", rc, seconds or -1.0, " ".join(cmd))
    if stdout:
        log.debug("stdout tail:\n%s", stdout)
    if stderr:
        log.log(level, "stderr tail:\n%s", stderr)
    event("command", stage=stage, cmd=cmd, returncode=rc, seconds=seconds,
          stdout_tail=stdout[-1500:], stderr_tail=stderr[-1500:])


def log_environment(extra: dict | None = None) -> dict:
    """Snapshot of everything that explains a run's behaviour."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    for module in ("numpy", "nibabel", "pydicom", "SimpleITK", "torch", "nnunetv2",
                   "spineps", "totalspineseg", "totalsegmentator", "pandas"):
        info[module] = _version(module)
    info["gpu"] = _gpu_info()
    info["dcm2niix"] = _which("dcm2niix")
    info["spineps_cli"] = _which("spineps")
    info["env"] = {k: os.environ.get(k) for k in
                   ("SPINEPS_SEGMENTOR_MODELS", "TOTALSEG_HOME_DIR", "TOTALSPINESEG_DATA",
                    "nnUNet_results", "HF_HOME", "TORCH_HOME")}
    info["source_commit"] = _git_sha()
    if extra:
        info.update(extra)

    log = get_logger()
    log.info("environment: python %s, torch %s, gpu %s", info["python"], info["torch"],
             info["gpu"].get("name") if isinstance(info["gpu"], dict) else info["gpu"])
    for module in ("spineps", "totalspineseg", "totalsegmentator", "nnunetv2"):
        log.info("  %-18s %s", module, info[module])
    event("environment", **info)
    return info


def _version(module: str) -> str | None:
    try:
        import importlib

        return getattr(importlib.import_module(module), "__version__", "present")
    except Exception as exc:  # noqa: BLE001
        return f"NOT IMPORTABLE: {str(exc)[:80]}"


def _gpu_info() -> dict | str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA device visible"
        props = torch.cuda.get_device_properties(0)
        return {"name": props.name, "vram_gb": round(props.total_memory / 1e9, 1),
                "cuda": torch.version.cuda}
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {str(exc)[:80]}"


def _which(binary: str) -> str | None:
    from .utils import tool_path

    return tool_path(binary)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=Path(__file__).resolve().parent.parent,
                             capture_output=True, text=True, timeout=15)
        return (out.stdout or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Digest
# --------------------------------------------------------------------------


def read_events(results_dir: str | Path) -> list[dict]:
    path = Path(results_dir) / EVENTS_FILENAME
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def results_path(work_dir: str | Path, station: int = 0) -> Path:
    """Where a run's results live. Mirrors Config.results_dir.

    Kept here as well because `diagnose` has no Config to ask: a study covering two
    craniocaudal stations is analysed one at a time, and pointing the diagnostics at
    `results/` after a `--station 2` run reports on the previous station instead of
    saying that this one has no results.
    """
    base = Path(work_dir) / "results"
    return base / f"station-{int(station)}" if station else base


def digest(work_dir: str | Path, *, log_tail: int = 40, station: int = 0) -> str:
    """Paste-sized account of a run: environment, stages, problems, tail of the log."""
    results = results_path(work_dir, station)
    lines: list[str] = []

    summary = _read_json(results / "summary.json")
    events = read_events(results)
    env = next((e for e in reversed(events) if e.get("kind") == "environment"), {})

    lines.append("=== spinelab run digest ===")
    lines.append(f"results dir : {results}")
    if summary:
        lines.append(f"version     : {summary.get('spinelab_version')}  "
                     f"subject: {summary.get('subject_id')}")
        if summary.get("aborted"):
            lines.append(f"ABORTED     : {summary['aborted']}")
    if env:
        gpu = env.get("gpu")
        gpu_text = gpu.get("name") if isinstance(gpu, dict) else gpu
        lines.append(f"source      : {env.get('source_commit')}   python {env.get('python')}")
        lines.append(f"gpu         : {gpu_text}"
                     + (f" ({gpu['vram_gb']} GB)" if isinstance(gpu, dict) else ""))
        lines.append("tools       : " + ", ".join(
            f"{m}={env.get(m)}" for m in ("spineps", "totalspineseg", "totalsegmentator",
                                          "nnunetv2", "torch", "SimpleITK")))

    if summary and summary.get("stages"):
        lines.append("")
        lines.append(f"{'stage':18s} {'status':9s} {'time':>8s}  reason")
        lines.append("-" * 78)
        for name, st in summary["stages"].items():
            lines.append(f"{name:18s} {st.get('status',''):9s} "
                         f"{st.get('duration_s', 0):7.1f}s  {(st.get('reason') or '')[:44]}")

    commands = [e for e in events if e.get("kind") == "command"]
    if commands:
        lines.append("")
        lines.append("external commands:")
        for c in commands:
            lines.append(f"  [{c.get('stage')}] rc={c.get('returncode')} "
                         f"{(c.get('seconds') or 0):.0f}s  {' '.join(c.get('cmd') or [])[:88]}")
            tail = (c.get("stderr_tail") or "").strip().splitlines()
            if c.get("returncode") not in (0, None) and tail:
                lines.append(f"       stderr: {tail[-1][:110]}")

    problems = [e for e in events if e.get("kind") in ("problem", "stage_failed")]
    if problems:
        lines.append("")
        lines.append("problems recorded:")
        for p in problems:
            lines.append(f"  [{p.get('stage')}] {str(p.get('detail') or p.get('reason'))[:110]}")

    log_path = results / LOG_FILENAME
    if log_path.exists() and log_tail:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-log_tail:]
        lines.append("")
        lines.append(f"last {len(tail)} log lines:")
        lines.extend("  " + t for t in tail)

    return "\n".join(lines)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
