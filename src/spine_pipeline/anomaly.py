"""Per-vertebra T2 z-score anomaly fallback (notebook Cell 8)."""

import numpy as np

from .geometry import LABEL_NAMES


def zscore_anomaly(t2, vert, z_thresh=3.0, min_voxels=200, min_hi=5, top_n=20):
    """Flag vertebrae with many high T2 z-score outlier voxels (edema proxy).

    `t2` and `vert` are same-shaped arrays (intensity, int labels). Returns the
    top-N findings sorted by high-outlier voxel count.
    """
    findings = []
    for label_id in np.unique(vert):
        if label_id == 0 or label_id > 24:
            continue
        m = vert == label_id
        if m.sum() < min_voxels:
            continue
        vox = t2[m]
        mu, sd = float(vox.mean()), float(vox.std() + 1e-9)
        z = (vox - mu) / sd
        hi = int((z > z_thresh).sum())
        if hi > min_hi:
            findings.append({
                "label_id": int(label_id),
                "label_name": LABEL_NAMES.get(int(label_id), f"id_{int(label_id)}"),
                "high_outlier_voxels": hi,
                "mean_intensity": mu,
                "std_intensity": sd,
                "method": "per_vertebra_zscore_T2",
            })
    findings.sort(key=lambda x: -x["high_outlier_voxels"])
    return findings[:top_n]
