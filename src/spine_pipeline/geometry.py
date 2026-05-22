"""Vertebra geometry from segmentation masks + centroids (notebook Cell 6).

All functions assume a canonical RAS+ oriented mask, i.e. after
`nibabel.as_closest_canonical`: axis 0 = L->R, axis 1 = P->A, axis 2 = I->S.
"""

import numpy as np

LABEL_NAMES = {
    **{i: f"C{i}" for i in range(1, 8)},
    **{i: f"T{i - 7}" for i in range(8, 20)},
    **{i: f"L{i - 19}" for i in range(20, 25)},
}


def angle_deg(v1, v2):
    """Angle between two vectors in degrees (0..180)."""
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def parse_centroids(ctd):
    """Parse SPINEPS-style centroids (list-of-dicts or dict) into {label: xyz}."""
    out = {}
    if isinstance(ctd, list):
        for item in ctd:
            if isinstance(item, dict) and "label" in item:
                lbl = int(item["label"])
                out[lbl] = np.array(
                    [item.get("X", 0), item.get("Y", 0), item.get("Z", 0)],
                    dtype=float,
                )
    elif isinstance(ctd, dict):
        for k, v in ctd.items():
            try:
                lbl = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                out[lbl] = np.array(
                    [v.get("X", 0), v.get("Y", 0), v.get("Z", 0)], dtype=float
                )
            elif isinstance(v, (list, tuple)) and len(v) >= 3:
                out[lbl] = np.array(v[:3], dtype=float)
    return out


def centroid_global_metrics(centroids):
    """Approximate thoracic kyphosis and max lateral deviation from centroids."""
    metrics = {}
    if not centroids:
        return metrics
    th_labels = sorted(l for l in centroids if 8 <= l <= 19)
    if len(th_labels) >= 3:
        first, last = th_labels[0], th_labels[-1]
        mid = th_labels[len(th_labels) // 2]
        v_top = centroids[mid] - centroids[first]
        v_bottom = centroids[last] - centroids[mid]
        metrics["thoracic_kyphosis_approx_deg"] = angle_deg(v_top, v_bottom)
        coords = np.stack([centroids[l] for l in th_labels])
        p0, p1 = coords[0], coords[-1]
        lv = p1 - p0
        lv = lv / (np.linalg.norm(lv) + 1e-9)
        devs = [np.linalg.norm(p - (p0 + np.dot(p - p0, lv) * lv)) for p in coords]
        metrics["max_lateral_deviation_mm"] = float(max(devs))
    return metrics


def slab_si_extent_vox(slab2d):
    """SI voxel extent of any-positive column in a (ap_slab, si) 2D slab."""
    si_present = np.where(slab2d.any(axis=0))[0]
    return int(si_present[-1] - si_present[0] + 1) if len(si_present) >= 2 else 0


def vertebra_metrics(data, zooms, ap_axis=1, si_axis=2, lr_axis=0,
                     slab_thickness=3, min_voxels=100):
    """Per-vertebra geometry (AP width, ant/post heights, wedge angle).

    `data` is an int label volume in canonical RAS+; `zooms` are voxel sizes
    (mm). Returns a list of dict records with plain-Python floats so the result
    is JSON-serializable.
    """
    ap_mm = float(zooms[ap_axis])
    si_mm = float(zooms[si_axis])
    per_vertebra = []

    for label_id in np.unique(data):
        if label_id == 0 or label_id > 24:
            continue
        mask = data == label_id
        vox = int(mask.sum())
        if vox < min_voxels:
            continue

        sag = mask.any(axis=lr_axis)  # (n_ap, n_si)
        base = {
            "label_id": int(label_id),
            "label_name": LABEL_NAMES.get(int(label_id), f"id_{int(label_id)}"),
            "voxel_count": vox,
        }
        if sag.sum() < 20:
            per_vertebra.append(dict(base))
            continue

        ap_present = np.where(sag.any(axis=1))[0]
        if len(ap_present) < 3:
            per_vertebra.append(dict(base))
            continue
        ap_min, ap_max = int(ap_present[0]), int(ap_present[-1])

        post_slab = sag[ap_min:ap_min + slab_thickness, :]
        ant_slab = sag[max(ap_max - slab_thickness + 1, 0):ap_max + 1, :]
        post_h_vox = slab_si_extent_vox(post_slab)
        ant_h_vox = slab_si_extent_vox(ant_slab)
        ap_width_vox = ap_max - ap_min

        rec = dict(base)
        rec["ap_width_mm"] = round(float(ap_width_vox * ap_mm), 2)
        rec["post_height_mm"] = round(float(post_h_vox * si_mm), 2)
        rec["ant_height_mm"] = round(float(ant_h_vox * si_mm), 2)
        if ant_h_vox > 0 and post_h_vox > 0 and ap_width_vox > 0:
            ap_width_mm = ap_width_vox * ap_mm
            post_h_mm = post_h_vox * si_mm
            ant_h_mm = ant_h_vox * si_mm
            rec["ant_post_height_ratio"] = round(float(ant_h_mm / post_h_mm), 3)
            rec["wedge_angle_deg"] = round(
                float(np.degrees(np.arctan2(post_h_mm - ant_h_mm, ap_width_mm))), 2
            )
        per_vertebra.append(rec)

    return per_vertebra
