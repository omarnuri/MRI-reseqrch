"""CPU-only tests for spine_pipeline.perf — no GPU required."""

import json
import time
from pathlib import Path

import pytest

from spine_pipeline.perf import (
    aggregate_performance,
    measure_tool,
    write_performance_log,
)


def test_measure_tool_records_wall_clock():
    log: list = []
    with measure_tool("dummy", log):
        time.sleep(0.05)
    assert len(log) == 1
    assert log[0]["tool"] == "dummy"
    assert log[0]["wall_clock_sec"] >= 0.04
    # On CPU-only environment, VRAM fields should be None
    assert log[0]["vram_peak_mb"] is None
    assert log[0]["vram_at_completion_mb"] is None


def test_measure_tool_records_exception_and_reraises():
    log: list = []
    with pytest.raises(ValueError):
        with measure_tool("explodes", log):
            raise ValueError("kaboom")
    assert len(log) == 1
    assert log[0]["tool"] == "explodes"
    assert "ValueError" in log[0]["error"]


def test_measure_tool_multiple_records():
    log: list = []
    for i in range(3):
        with measure_tool(f"tool_{i}", log):
            time.sleep(0.01)
    assert [r["tool"] for r in log] == ["tool_0", "tool_1", "tool_2"]


def test_aggregate_performance_empty():
    out = aggregate_performance([])
    assert out["n_tools"] == 0
    assert out["pipeline_total_wall_clock_sec"] == 0
    assert out["pipeline_peak_vram_mb"] is None
    assert out["n_tools_with_errors"] == 0


def test_aggregate_performance_with_data():
    log = [
        {"tool": "a", "wall_clock_sec": 1.5, "vram_peak_mb": 8000},
        {"tool": "b", "wall_clock_sec": 2.0, "vram_peak_mb": 12000},
        {"tool": "c", "wall_clock_sec": 0.5, "vram_peak_mb": None},
    ]
    out = aggregate_performance(log)
    assert out["n_tools"] == 3
    assert out["pipeline_total_wall_clock_sec"] == 4.0
    assert out["pipeline_peak_vram_mb"] == 12000
    assert out["n_tools_with_errors"] == 0


def test_aggregate_counts_errors():
    log = [
        {"tool": "a", "wall_clock_sec": 1.0},
        {"tool": "b", "wall_clock_sec": 0.5, "error": "RuntimeError: oom"},
    ]
    assert aggregate_performance(log)["n_tools_with_errors"] == 1


def test_write_performance_log_creates_file(tmp_path: Path):
    log = [{"tool": "a", "wall_clock_sec": 1.0, "vram_peak_mb": 5000,
            "vram_at_completion_mb": 500}]
    out = tmp_path / "perf.json"
    write_performance_log(log, out, session_metadata={"session_id": "test-1"})
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["session_id"] == "test-1"
    assert data["summary"]["n_tools"] == 1
    assert data["per_tool"][0]["tool"] == "a"


def test_write_performance_log_creates_parent_dir(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "perf.json"
    write_performance_log([], nested)
    assert nested.exists()


def test_write_performance_log_empty_log(tmp_path: Path):
    out = tmp_path / "perf.json"
    write_performance_log([], out, session_metadata={"gpu": "none"})
    data = json.loads(out.read_text())
    assert data["per_tool"] == []
    assert data["summary"]["n_tools"] == 0
    assert data["gpu"] == "none"
