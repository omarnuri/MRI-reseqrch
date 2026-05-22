"""Segmentation-output detection + log parsing helpers (notebook Cells 4/13).

`detect_seg_outputs` fixes the TotalSpineSeg false-"failed": the previous
filename patterns didn't match the tool's real output names, so a successful run
was reported as a failure. Since the input copy lives outside the output dir,
any produced `*.nii.gz` under the output dir is a genuine result.
"""

import json
from pathlib import Path


def detect_seg_outputs(out_dir, exclude_dir=None, pattern="*.nii.gz"):
    """All `pattern` files under `out_dir`, excluding any under `exclude_dir`."""
    out_dir = Path(out_dir)
    exclude = Path(exclude_dir) if exclude_dir is not None else None
    found = []
    for p in out_dir.rglob(pattern):
        if exclude is not None and (exclude == p or exclude in p.parents):
            continue
        found.append(p)
    return list(dict.fromkeys(found))  # dedupe, preserve order


def clean_reason(text, fallback="no output produced"):
    """Last meaningful log line, skipping tqdm/progress noise."""
    if not text:
        return fallback
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    lines = [ln for ln in lines
             if "%|" not in ln and "it/s" not in ln and "B/s" not in ln]
    return lines[-1][:300] if lines else fallback


def load_json(p, default=None):
    """Load JSON, returning `default` on any error."""
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default
