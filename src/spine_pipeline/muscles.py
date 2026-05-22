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


def fatty_fraction_from_t1(mask_full, t1_volume, percentile_within=75):
    """Goutallier-style fatty-infiltration proxy from T1.

    Fat is brightest on T1 (white), muscle is mid-gray. Computing
    `(t2 > 0.6)` (the old approach) over a T2 image catches CSF/fluid as
    well as fat and is the reason fatty_frac came out 0.0 on findings_8.

    This function defines the "fatty" threshold relative to the in-muscle
    intensity distribution: the top `percentile_within`% of voxels inside
    the muscle mask are flagged. Returns left and right fatty fractions
    (split on axis 0 = L->R in canonical RAS) plus the threshold actually
    used, so the value is reproducible.
    """
    mask_full = mask_full.astype(bool)
    if not mask_full.any():
        return {
            "fatty_frac_left_t1": 0.0,
            "fatty_frac_right_t1": 0.0,
            "fat_threshold_intensity": None,
            "method": "T1_percentile_within_muscle",
        }
    in_muscle = t1_volume[mask_full]
    if in_muscle.size < 50:
        return {
            "fatty_frac_left_t1": 0.0,
            "fatty_frac_right_t1": 0.0,
            "fat_threshold_intensity": None,
            "method": "T1_percentile_within_muscle",
        }
    fat_threshold = float(np.percentile(in_muscle, percentile_within))

    mid_x = mask_full.shape[0] // 2
    left = mask_full.copy()
    left[mid_x:, :, :] = False
    right = mask_full.copy()
    right[:mid_x, :, :] = False

    fatty_left = float(((t1_volume > fat_threshold) & left).sum()) / max(int(left.sum()), 1)
    fatty_right = float(((t1_volume > fat_threshold) & right).sum()) / max(int(right.sum()), 1)

    return {
        "fatty_frac_left_t1": round(fatty_left, 3),
        "fatty_frac_right_t1": round(fatty_right, 3),
        "fat_threshold_intensity": round(fat_threshold, 1),
        "method": f"T1_percentile_within_muscle_p{percentile_within}",
    }

