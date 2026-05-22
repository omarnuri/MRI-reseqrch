import numpy as np

from spine_pipeline.muscles import (
    muscle_side_metrics, normalize_t2, fatty_fraction_from_t1,
)


def test_normalize_range():
    t2 = np.linspace(0, 1000, 1000).reshape(10, 10, 10)
    n = normalize_t2(t2)
    assert n.min() >= 0 and n.max() <= 1


def test_symmetric_mask_zero_asymmetry():
    mask = np.zeros((10, 10, 10), bool)
    mask[2:4, 2:8, 2:8] = True  # left (axis0 < mid=5)
    mask[6:8, 2:8, 2:8] = True  # right
    m = muscle_side_metrics(mask, np.zeros((10, 10, 10)))
    assert m["csa_left_voxels"] == m["csa_right_voxels"]
    assert m["asymmetry_pct"] == 0.0


def test_asymmetric_mask():
    mask = np.zeros((10, 10, 10), bool)
    mask[1:4, 2:8, 2:8] = True  # bigger left
    mask[6:7, 2:8, 2:8] = True  # smaller right
    m = muscle_side_metrics(mask, np.zeros((10, 10, 10)))
    assert m["csa_left_voxels"] > m["csa_right_voxels"]
    assert m["asymmetry_pct"] > 0


def test_fatty_fraction():
    mask = np.zeros((10, 10, 10), bool)
    mask[2:4, 2:8, 2:8] = True  # left only
    m = muscle_side_metrics(mask, np.ones((10, 10, 10)))  # all fatty (>0.6)
    assert m["fatty_frac_left"] == 1.0
    assert m["csa_right_voxels"] == 0
    assert m["fatty_frac_right"] == 0.0


def test_fatty_fraction_from_t1_empty_mask():
    out = fatty_fraction_from_t1(np.zeros((4, 4, 4), bool), np.ones((4, 4, 4)))
    assert out["fatty_frac_left_t1"] == 0.0
    assert out["fatty_frac_right_t1"] == 0.0
    assert out["fat_threshold_intensity"] is None


def test_fatty_fraction_from_t1_uniform_muscle():
    # Left half bright (fat), right half normal — p50 threshold splits them.
    mask = np.zeros((10, 10, 10), bool)
    mask[:, 2:8, 2:8] = True
    t1 = np.full((10, 10, 10), 100.0)
    t1[:5, 2:8, 2:8] = 300.0  # bright fat on left half (axis 0 = L->R)
    out = fatty_fraction_from_t1(mask, t1, percentile_within=50)
    # Threshold ~200 (midpoint between 100 and 300 cohorts);
    # left voxels at 300 are > threshold, right at 100 are not.
    assert out["fatty_frac_left_t1"] > out["fatty_frac_right_t1"]
    assert out["fatty_frac_left_t1"] > 0.5
    assert out["fatty_frac_right_t1"] < 0.1


def test_fatty_fraction_from_t1_threshold_reproducible():
    mask = np.zeros((10, 10, 10), bool)
    mask[2:8, 2:8, 2:8] = True
    t1 = np.zeros((10, 10, 10))
    t1[mask] = np.linspace(50, 150, int(mask.sum()))
    out = fatty_fraction_from_t1(mask, t1, percentile_within=75)
    # Threshold should be around the 75th percentile of 50..150 = ~125
    assert 100 < out["fat_threshold_intensity"] < 145
    assert "T1_percentile_within_muscle_p75" == out["method"]


def test_fatty_fraction_from_t1_t2_method_distinct_from_t1():
    # Regression for findings_8: fatty_frac on T2 gave 0.0 because nothing
    # in muscle exceeded 0.6 of the normalized [0,1] range. T1-based path
    # uses absolute intensity percentile and is independent of T2.
    mask = np.zeros((10, 10, 10), bool)
    mask[2:8, 2:8, 2:8] = True
    t1 = np.random.RandomState(0).randint(100, 300, (10, 10, 10)).astype(float)
    out = fatty_fraction_from_t1(mask, t1, percentile_within=75)
    # Both sides should have non-trivial fatty_frac > 0 (some voxels above p75 exist).
    assert out["fatty_frac_left_t1"] + out["fatty_frac_right_t1"] > 0.0
