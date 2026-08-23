"""The Colab launcher, checked without a Colab.

The notebook is the only part of this project a user actually touches, and it is
also the part no test suite normally reaches: a broken cell is discovered on a
GPU runtime, twenty minutes and one slow upload into a session. The checks here
are cheap and cover the two failures that would waste that session — a cell that
does not parse, and a cell that reads a path the pipeline no longer writes to.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from spinelab.config import DEFAULT_STAGES

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pipeline.ipynb"


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return payload["cells"]


@pytest.fixture(scope="module")
def code(cells) -> dict[int, str]:
    return {i: "".join(c["source"]) for i, c in enumerate(cells) if c["cell_type"] == "code"}


class TestLauncher:
    def test_every_code_cell_parses(self, code):
        for index, source in code.items():
            try:
                ast.parse(source)
            except SyntaxError as exc:
                pytest.fail(f"cell {index} does not parse: {exc}")

    def test_no_outputs_are_committed(self, cells):
        # A notebook with embedded outputs was 1.7 MB in this repository and had
        # to be excluded from the sparse checkout. Outputs can also carry data.
        for index, cell in enumerate(cells):
            assert not cell.get("outputs"), f"cell {index} has committed outputs"

    def test_the_station_reaches_the_run_and_the_diagnostics(self, code):
        run_cell = next(s for s in code.values() if "run_pipeline(config)" in s)
        assert "STATION = 0  # @param" in run_cell
        assert "station=STATION" in run_cell
        diagnose = [s for s in code.values() if '"diagnose"' in s]
        assert diagnose, "no cell runs `spinelab diagnose`"
        for source in diagnose:
            assert '"--station"' in source

    def test_results_are_read_from_where_the_run_wrote_them(self, code):
        # With a station the results live in results/station-N. Every consumer
        # cell has to follow that, or it silently reports the previous station.
        run_cell = next(s for s in code.values() if "run_pipeline(config)" in s)
        assert "RESULTS_DIR = str(config.results_dir)" in run_cell
        for index, source in code.items():
            if "run_pipeline(config)" in source:
                continue
            assert '"/content/spine_work/results"' not in source or \
                   'globals().get("RESULTS_DIR"' in source, \
                   f"cell {index} hard-codes the results path"

    def test_the_optional_downloads_are_switches_with_a_stated_size(self, code):
        setup = next(s for s in code.values() if '"setup"' in s)
        assert "INSTALL_SCT = True  # @param" in setup
        assert "BUILD_NORMATIVE = True  # @param" in setup
        assert '"--sct"' in setup and '"normative", "--cache"' in setup
        assert "ГБ" in setup and "МБ" in setup

    def test_the_stage_shortcuts_name_stages_that_exist(self, code):
        run_cell = next(s for s in code.values() if "run_pipeline(config)" in s)
        line = next(line for line in run_cell.splitlines() if line.startswith("STAGES ="))
        choices = json.loads(line.split("# @param", 1)[1].strip())
        for choice in choices:
            if choice == "all":
                continue
            for stage in choice.split(","):
                assert stage.strip() in DEFAULT_STAGES, f"{stage} is not a stage"
