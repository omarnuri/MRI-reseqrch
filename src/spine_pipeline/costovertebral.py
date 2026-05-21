"""Costovertebral-region bright-signal screen (notebook Cell 10)."""

import numpy as np


def bright_mask(stir, cv_region, frac=0.85, pct=99):
    """Bright-voxel mask inside `cv_region` using a percentile threshold."""
    if cv_region.any():
        thresh = np.percentile(stir[cv_region], pct)
    else:
        thresh = np.percentile(stir, pct)
    return (stir > frac * thresh) & cv_region


def side_bright_fractions(cv_region, bright):
    """Per-side (L/R, split on axis 0) region/bright voxel counts + fraction."""
    results = []
    mid_x = cv_region.shape[0] // 2
    for side, sl in [("L", slice(None, mid_x)), ("R", slice(mid_x, None))]:
        side_region = np.zeros_like(cv_region)
        side_region[sl] = cv_region[sl]
        side_bright = bright & side_region
        region_vox = int(side_region.sum())
        results.append({
            "side": side,
            "region_voxels": region_vox,
            "bright_voxels": int(side_bright.sum()),
            "bright_fraction": float(side_bright.sum()) / max(region_vox, 1),
        })
    return results
