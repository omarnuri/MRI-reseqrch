import numpy as np

from spine_pipeline.costovertebral import bright_mask, side_bright_fractions


def test_bright_mask_flags_hotspot():
    stir = np.zeros((10, 10, 10))
    region = np.zeros((10, 10, 10), bool)
    region[2:8, 2:8, 2:8] = True
    stir[region] = 100
    stir[3, 3, 3] = 1000
    bm = bright_mask(stir, region, frac=0.85, pct=99)
    assert bm[3, 3, 3]
    assert bm.sum() >= 1
    # bright voxels are confined to the region
    assert not bm[~region].any()


def test_side_bright_fractions_split():
    region = np.zeros((10, 10, 10), bool)
    region[1:4, 2:8, 2:8] = True  # left (axis0 < mid=5)
    region[6:9, 2:8, 2:8] = True  # right
    bright = np.zeros((10, 10, 10), bool)
    bright[2, 3, 3] = True  # one left-side bright voxel
    by = {r["side"]: r for r in side_bright_fractions(region, bright)}
    assert by["L"]["bright_voxels"] == 1
    assert by["R"]["bright_voxels"] == 0
    assert by["L"]["region_voxels"] > 0
    assert 0 <= by["L"]["bright_fraction"] <= 1
