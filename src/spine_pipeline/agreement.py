"""Cross-tool segmentation agreement (notebook Cell 12)."""

import numpy as np


def dice(a, b):
    """Dice overlap of two boolean masks (0.0 when both empty)."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = int((a & b).sum())
    denom = int(a.sum()) + int(b.sum())
    return 2.0 * inter / denom if denom else 0.0
