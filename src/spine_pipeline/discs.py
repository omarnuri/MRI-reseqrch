"""Disc dehydration ranking by T2 intensity (notebook Cell 9 fallback)."""

import numpy as np


def rank_discs(t2, disc_mask, min_voxels=50, label_min=0):
    """Rank discs by mean T2 (brightest = rank 1 = healthiest).

    `t2` and `disc_mask` are same-shaped arrays. Returns rows with a
    `relative_dehydration_rank` (1 = most hydrated ... N = most dehydrated).

    `label_min` ignores labels below this value. For TotalSpineSeg pass
    `label_min=63`: that scheme uses 1=cord, 2=canal/CSF, 11-50=vertebrae/sacrum
    and >=63=intervertebral discs, so without the filter the CSF-bright canal and
    the vertebral bodies get ranked as "discs".
    """
    rows = []
    for d_id in np.unique(disc_mask):
        if d_id == 0 or d_id < label_min:
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
