import numpy as np

from spine_pipeline.discs import rank_discs


def test_orders_by_intensity_brightest_first():
    mask = np.zeros((10, 10, 10), np.int32)
    mask[0:5, 0:5, 0:2] = 1
    mask[0:5, 0:5, 2:4] = 2
    mask[0:5, 0:5, 4:6] = 3
    t2 = np.zeros((10, 10, 10))
    t2[mask == 1] = 300  # brightest -> rank 1 (most hydrated)
    t2[mask == 2] = 200
    t2[mask == 3] = 100  # darkest -> rank 3 (most dehydrated)
    ranks = {r["disc_id"]: r["relative_dehydration_rank"]
             for r in rank_discs(t2, mask, min_voxels=10)}
    assert ranks == {1: 1, 2: 2, 3: 3}


def test_skips_small_discs():
    mask = np.zeros((10, 10, 10), np.int32)
    mask[0, 0, 0] = 1
    assert rank_discs(np.ones((10, 10, 10)), mask, min_voxels=50) == []


def test_reports_voxel_count_and_mean():
    mask = np.zeros((10, 10, 10), np.int32)
    mask[0:5, 0:5, 0:2] = 1
    t2 = np.full((10, 10, 10), 42.0)
    rows = rank_discs(t2, mask, min_voxels=10)
    assert rows[0]["voxel_count"] == 50
    assert abs(rows[0]["mean_T2"] - 42.0) < 1e-9


def test_label_min_filters_cord_canal_vertebrae():
    # TotalSpineSeg scheme: 1=cord, 2=canal/CSF (brightest!), 25=vertebra, >=63=discs.
    # Regression for the bug where the CSF-bright canal ranked as the #1 "disc".
    mask = np.zeros((10, 10, 12), np.int32)
    mask[0:5, 0:5, 0:2] = 1     # cord
    mask[0:5, 0:5, 2:4] = 2     # canal / CSF
    mask[0:5, 0:5, 4:6] = 25    # vertebra body
    mask[0:5, 0:5, 6:8] = 63    # real disc
    mask[0:5, 0:5, 8:10] = 82   # real disc
    t2 = np.zeros((10, 10, 12))
    t2[mask == 2] = 500   # canal brightest — must NOT become rank-1 disc
    t2[mask == 1] = 250
    t2[mask == 25] = 200
    t2[mask == 63] = 180
    t2[mask == 82] = 150
    rows = rank_discs(t2, mask, min_voxels=10, label_min=63)
    assert {r["disc_id"] for r in rows} == {63, 82}
    rank1 = next(r for r in rows if r["relative_dehydration_rank"] == 1)
    assert rank1["disc_id"] == 63  # brightest *disc*, not the canal
