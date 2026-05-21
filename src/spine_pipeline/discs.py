"""Disc dehydration ranking by T2 intensity (notebook Cell 9 fallback)."""

import numpy as np


def rank_discs(t2, disc_mask, min_voxels=50):
    """Rank discs by mean T2 (brightest = rank 1 = healthiest).

    `t2` and `disc_mask` are same-shaped arrays. Returns rows with a
    `relative_dehydration_rank` (1 = most hydrated ... N = most dehydrated).
    """
    rows = []
    for d_id in np.unique(disc_mask):
        if d_id == 0:
            continue
        m = disc_mask == d_id
        if m.sum() < min_voxels:
            continue
        vals = t2[m]
        rows.append({
            "disc_id": int(d_id),
            "mean_T2": float(vals.mean()),
            "voxel_count": int(m.sum()),
        })
    rows.sort(key=lambda r: -r["mean_T2"])
    for rank, r in enumerate(rows, 1):
        r["relative_dehydration_rank"] = rank
    return rows
