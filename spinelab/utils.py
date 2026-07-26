"""Small shared helpers: JSON safety, canonical image loading, resampling, logs.

Deliberately free of any heavy import at module level so `spinelab.utils` can be
imported (and unit-tested) on a machine that has nothing but the stdlib.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, Paths and sets to JSON types.

    `header.get_zooms()` returns float32, which json.dump refuses; in the old
    notebook that raised mid-run and killed the geometry stage. Numbers must stay
    numbers (not str), so this converts rather than stringifies.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    # numpy without importing numpy: duck-type on .item()/.tolist()
    if hasattr(obj, "item") and hasattr(obj, "dtype") and getattr(obj, "shape", None) == ():
        return obj.item()
    if hasattr(obj, "tolist") and hasattr(obj, "dtype"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2, ensure_ascii=False)
    return path


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


# --------------------------------------------------------------------------
# Log helpers
# --------------------------------------------------------------------------


def clean_reason(text: str | None, fallback: str = "no output produced") -> str:
    """Last meaningful log line, skipping tqdm/progress noise.

    Without this a failure reason often ended up being a progress bar
    ("100%|####| 1/1 [00:00, 4.26it/s]"), which tells the reader nothing.
    """
    if not text:
        return fallback
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()]
    noise = ("%|", "it/s", "B/s", "s/it")
    lines = [ln for ln in lines if not any(tok in ln for tok in noise)]
    return lines[-1][:300] if lines else fallback


def human_duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        return False


# --------------------------------------------------------------------------
# Image helpers (import nibabel lazily)
# --------------------------------------------------------------------------

#: Axis meaning after nibabel.as_closest_canonical (RAS+):
#: axis 0 = L->R, axis 1 = P->A, axis 2 = I->S
LR_AXIS, AP_AXIS, SI_AXIS = 0, 1, 2


def load_canonical(path: str | Path):
    """Load a NIfTI and reorient to closest canonical RAS+.

    Every stage does this. The old notebook mixed canonical and non-canonical
    loads, so "axis 0 is left/right" was true in some cells and false in others —
    which silently mirrored the left/right asymmetry numbers that this whole
    case turns on.
    """
    import nibabel as nib

    return nib.as_closest_canonical(nib.load(str(path)))


def resample_mask_to(mask_img, ref_img):
    """Resample a label/mask volume onto ref_img's grid with nearest neighbour."""
    from nibabel.processing import resample_from_to

    if mask_img.shape == ref_img.shape and _affines_close(mask_img, ref_img):
        return mask_img
    return resample_from_to(mask_img, ref_img, order=0)


def _affines_close(a, b, tol: float = 1e-3) -> bool:
    try:
        import numpy as np

        return bool(np.allclose(a.affine, b.affine, atol=tol))
    except Exception:
        return False


def voxel_volume_mm3(img) -> float:
    zooms = img.header.get_zooms()[:3]
    vol = 1.0
    for z in zooms:
        vol *= float(z)
    return vol


def first_existing(paths: Iterable[str | Path]) -> Path | None:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def find_outputs(root: str | Path, pattern: str, exclude_dirs: Iterable[str] = ("input", "raw", "work")) -> list[Path]:
    """Collect tool outputs under root, excluding staging/input directories.

    The notebook repeatedly mistook its own *input* copy for an output (that is
    why TotalSpineSeg was reported as "failed" on successful runs), so exclusion
    is centralised here instead of being re-invented per stage.
    """
    root = Path(root)
    if not root.exists():
        return []
    bad = {d.lower() for d in exclude_dirs}
    out = []
    for p in sorted(root.rglob(pattern)):
        parts = {q.lower() for q in p.parts}
        if parts & bad:
            continue
        out.append(p)
    return out
