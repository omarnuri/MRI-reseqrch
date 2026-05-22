"""Per-tool performance measurement (VRAM peak, wall-clock) for the pipeline.

Used to log objective resource footprints into results/performance_log.json so
the team can compare actual cost-per-output across tools (and detect
regressions when a tool starts using 2x its previous VRAM).

The module is intentionally torch-tolerant: when torch is missing or
torch.cuda is not available (CPU-only test environment), measurements
collapse to wall-clock only with vram fields set to None.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any


def _torch_cuda():
    """Return torch.cuda if available, else None."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch.cuda
    except Exception:
        pass
    return None


@contextlib.contextmanager
def measure_tool(name: str, perf_log: list[dict[str, Any]]):
    """Context manager that records wall-clock and CUDA peak VRAM for a tool.

    Resets peak memory stats on entry so each tool's record reflects its OWN
    peak rather than the running max across the whole session.

    Args:
        name: tool identifier (e.g. 'SPINEPS')
        perf_log: list mutated in-place with the resulting record

    Records appended (one per `with measure_tool(...)` invocation):
        {
            "tool": str,
            "wall_clock_sec": float,
            "vram_peak_mb": float or None,
            "vram_at_completion_mb": float or None,
            "error": str (only if the wrapped block raised),
        }
    Exceptions in the wrapped block are re-raised after recording.
    """
    cuda = _torch_cuda()
    if cuda is not None:
        try:
            cuda.empty_cache()
            cuda.reset_peak_memory_stats()
        except Exception:
            pass
    t0 = time.time()
    err: Exception | None = None
    try:
        yield
    except Exception as e:
        err = e
        raise
    finally:
        elapsed = time.time() - t0
        if cuda is not None:
            try:
                peak_mb = float(cuda.max_memory_allocated()) / 1e6
                cur_mb = float(cuda.memory_allocated()) / 1e6
            except Exception:
                peak_mb = None
                cur_mb = None
        else:
            peak_mb = None
            cur_mb = None
        record: dict[str, Any] = {
            "tool": name,
            "wall_clock_sec": round(elapsed, 2),
            "vram_peak_mb": (round(peak_mb, 0) if peak_mb is not None else None),
            "vram_at_completion_mb": (round(cur_mb, 0) if cur_mb is not None else None),
        }
        if err is not None:
            record["error"] = type(err).__name__ + ": " + (str(err)[:200])
        perf_log.append(record)


def aggregate_performance(perf_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pipeline-level summary from per-tool records.

    Returns a dict suitable for dumping into performance_log.json alongside the
    per-tool list. Robust to missing VRAM fields.
    """
    total_time = sum(r.get("wall_clock_sec", 0.0) for r in perf_log)
    peaks = [r.get("vram_peak_mb") for r in perf_log if r.get("vram_peak_mb") is not None]
    return {
        "n_tools": len(perf_log),
        "pipeline_total_wall_clock_sec": round(total_time, 2),
        "pipeline_peak_vram_mb": (round(max(peaks), 0) if peaks else None),
        "n_tools_with_errors": sum(1 for r in perf_log if "error" in r),
    }


def write_performance_log(perf_log: list[dict[str, Any]], path: str | Path,
                          session_metadata: dict[str, Any] | None = None) -> None:
    """Dump perf_log + summary to a JSON file.

    `session_metadata` is merged into the top-level dict (e.g. GPU name, run id).
    Does NOT include the full perf_log if it's empty — writes an empty dict for
    `per_tool` so consumers can still load the file.
    """
    summary = aggregate_performance(perf_log)
    payload: dict[str, Any] = {
        **(session_metadata or {}),
        "summary": summary,
        "per_tool": list(perf_log),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
