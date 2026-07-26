"""Stage `radiomics`: first-order + GLCM texture features per vertebral body.

pyradiomics is still pinned at 3.1.0 (May 2023) and does not build on current
Colab Pythons, so this stage uses it when importable and otherwise computes a
small, documented feature set with numpy. Either way the features are descriptive
only: with a single study there is no reference distribution to compare them
against, so they are exported for later use rather than interpreted here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import labels as L
from ..evidence import Evidence, Status
from ..pipeline import Context, SkipStage, StageResult
from ..utils import load_canonical, resample_mask_to, write_json

GLCM_LEVELS = 32


def run(ctx: Context) -> StageResult:
    cfg = ctx.config
    t2_sag = ctx.require_sequence("T2_SAG")
    spineps = ctx.stage_data("spineps")
    instance_masks = [Path(p) for p in spineps.get("instance_masks", [])]
    if not instance_masks:
        raise SkipStage("no SPINEPS vertebra instance mask")

    t2_img = load_canonical(t2_sag)
    t2 = np.asarray(t2_img.get_fdata(), dtype=float)
    inst = np.asarray(resample_mask_to(load_canonical(instance_masks[0]), t2_img)
                      .get_fdata()).astype(np.int32)

    quantised = _quantise(t2)
    rows = []
    for label_id in sorted(int(v) for v in np.unique(inst)):
        if not L.is_vertebra(label_id):
            continue
        mask = inst == label_id
        if int(mask.sum()) < 100:
            continue
        values = t2[mask]
        features = _first_order(values)
        features.update(_glcm_features(quantised, mask))
        rows.append({"label_id": label_id, "name": L.vertebra_name(label_id),
                     "voxel_count": int(mask.sum()), "features": features})

    if not rows:
        raise SkipStage("no vertebra large enough for texture features")

    payload = {
        "reference_image": t2_sag,
        "method": "numpy first-order + GLCM (levels=32) on the mid-mask sagittal slice",
        "n_vertebrae": len(rows),
        "vertebrae": rows,
        "interpretation_limits": [
            "Texture features are unitless and scanner-dependent. With one study there is "
            "no normative distribution, so nothing here is a finding — the values exist to "
            "be compared against a future follow-up on the same scanner and protocol.",
        ],
    }
    write_json(cfg.intermediate_dir / "radiomics.json", payload)
    return StageResult(name="radiomics", status=Status.OK, evidence=Evidence.MEASUREMENT,
                       data=payload)


def _quantise(volume: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(volume, [1, 99])
    scaled = np.clip((volume - p1) / (p99 - p1 + 1e-9), 0, 1)
    return (scaled * (GLCM_LEVELS - 1)).astype(np.uint8)


def _first_order(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std())
    centred = values - mean
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(float(np.median(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        # Fisher skewness / excess kurtosis without scipy (kept dependency-light).
        "skewness": round(float((centred ** 3).mean() / (std ** 3 + 1e-12)), 4),
        "kurtosis_excess": round(float((centred ** 4).mean() / (std ** 4 + 1e-12) - 3.0), 4),
        "coefficient_of_variation": round(float(std / (abs(mean) + 1e-12)), 4),
    }


def _glcm_features(quantised: np.ndarray, mask: np.ndarray) -> dict:
    """Contrast / homogeneity / energy from a 2-direction GLCM on one slice.

    The slice is the one containing most of the mask, not the middle of the
    volume: for a vertebra at the edge of the field of view the volume's middle
    slice may not contain the vertebra at all.
    """
    counts = mask.sum(axis=(1, 2))
    if counts.max() == 0:
        return {}
    idx = int(np.argmax(counts))
    patch = quantised[idx].astype(np.int32)
    mask2d = mask[idx]
    if int(mask2d.sum()) < 50:
        return {}

    glcm = np.zeros((GLCM_LEVELS, GLCM_LEVELS), dtype=float)
    for axis in (0, 1):
        a = patch
        m = mask2d
        if axis == 0:
            pairs_ok = m[:-1, :] & m[1:, :]
            i_vals, j_vals = a[:-1, :][pairs_ok], a[1:, :][pairs_ok]
        else:
            pairs_ok = m[:, :-1] & m[:, 1:]
            i_vals, j_vals = a[:, :-1][pairs_ok], a[:, 1:][pairs_ok]
        if i_vals.size == 0:
            continue
        np.add.at(glcm, (i_vals, j_vals), 1.0)
        np.add.at(glcm, (j_vals, i_vals), 1.0)  # symmetric
    total = glcm.sum()
    if total <= 0:
        return {}
    glcm /= total
    i_idx, j_idx = np.indices(glcm.shape)
    diff = np.abs(i_idx - j_idx)
    return {
        "glcm_contrast": round(float((glcm * diff ** 2).sum()), 5),
        "glcm_dissimilarity": round(float((glcm * diff).sum()), 5),
        "glcm_homogeneity": round(float((glcm / (1.0 + diff ** 2)).sum()), 5),
        "glcm_energy": round(float((glcm ** 2).sum()), 5),
        "glcm_entropy": round(float(-(glcm[glcm > 0] * np.log2(glcm[glcm > 0])).sum()), 5),
    }
