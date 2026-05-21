import numpy as np

from spine_pipeline.agreement import dice


def test_identical_masks_dice_one():
    a = np.zeros((5, 5, 5), bool)
    a[1:4, 1:4, 1:4] = True
    assert dice(a, a) == 1.0


def test_disjoint_masks_dice_zero():
    a = np.zeros((5, 5, 5), bool)
    a[0:2] = True
    b = np.zeros((5, 5, 5), bool)
    b[3:5] = True
    assert dice(a, b) == 0.0


def test_both_empty_dice_zero():
    a = np.zeros((5, 5, 5), bool)
    assert dice(a, a) == 0.0


def test_partial_overlap():
    a = np.zeros((4, 1, 1), bool)
    a[0:2] = True  # 2 voxels
    b = np.zeros((4, 1, 1), bool)
    b[1:3] = True  # 2 voxels, overlap 1
    assert abs(dice(a, b) - 0.5) < 1e-9
