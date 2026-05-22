"""Static checks on the notebook: valid JSON and every code cell compiles.

Catches accidental syntax breakage from editing the .ipynb, without needing the
heavy GPU dependencies (compile() does not execute imports).
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "spine_analysis_pipeline.ipynb"


def _strip_ipython_magics(src):
    out = []
    for ln in src.split("\n"):
        stripped = ln.lstrip()
        if (stripped.startswith("!") or stripped.startswith("%")
                or stripped.startswith("get_ipython")):
            indent = ln[: len(ln) - len(stripped)]
            out.append(indent + "pass  # ipython magic")
        else:
            out.append(ln)
    return "\n".join(out)


def test_notebook_is_valid_nbformat():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and len(nb["cells"]) > 0


def test_all_code_cells_compile():
    nb = json.loads(NB.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        cleaned = _strip_ipython_magics("".join(cell["source"]))
        try:
            compile(cleaned, f"<cell {i}>", "exec")
        except SyntaxError as e:  # pragma: no cover - failure path
            raise AssertionError(f"Cell {i} has a syntax error: {e}")
