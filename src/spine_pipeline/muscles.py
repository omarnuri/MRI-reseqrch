"""Paraspinal muscle L/R asymmetry + fatty-fraction proxy (notebook Cell 7)."""

import numpy as np


def normalize_t2(t2_img):
    """Normalize 1-99 percentile intensities to [0, 1]."""
    p1, p99 = np.percentile(t2_img, [1, 99])
    return np.clip((t2_img - p1) / (p99 - p1 + 1e-9), 0, 1)


def muscle_side_metrics(mask, t2_norm, fatty_thresh=0.6):
    """Left/right CSA, fatty fraction and asymmetry for one muscle mask.

    Sides are split on the mid-sagittal plane (axis 0). `t2_norm` is the
    normalized T2 volume (same shape as `mask`).
    """
    mask = mask.astype(bool)
    mid_x = mask.shape[0] // 2
    left = mask.copy()
    left[mid_x:, :, :] = False
    right = mask.copy()
    right[:mid_x, :, :] = False

    csa_left = int(left.sum())
    csa_right = int(right.sum())
    fatty_left = float(((t2_norm > fatty_thresh) & left).sum()) / max(csa_left, 1)
    fatty_right = float(((t2_norm > fatty_thresh) & right).sum()) / max(csa_right, 1)
    asym_pct = 100 * abs(csa_left - csa_right) / max((csa_left + csa_right) / 2, 1)
    return {
        "csa_left_voxels": csa_left,
        "csa_right_voxels": csa_right,
        "asymmetry_pct": round(asym_pct, 2),
        "fatty_frac_left": round(fatty_left, 3),
        "fatty_frac_right": round(fatty_right, 3),
    }
