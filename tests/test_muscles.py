import numpy as np

from spine_pipeline.muscles import muscle_side_metrics, normalize_t2


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
