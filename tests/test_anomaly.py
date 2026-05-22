import numpy as np

from spine_pipeline.anomaly import zscore_anomaly


def _label_volume():
    vert = np.zeros((12, 12, 12), np.int32)
    vert[2:10, 2:10, 2:10] = 8  # 512 voxels
    return vert


def test_detects_bright_outliers():
    vert = _label_volume()
    t2 = np.full((12, 12, 12), 100.0)
    coords = np.argwhere(vert == 8)[:20]
    for x, y, z in coords:
        t2[x, y, z] = 5000.0
    findings = zscore_anomaly(t2, vert, min_hi=0)
    assert len(findings) == 1
    assert findings[0]["label_id"] == 8
    assert findings[0]["high_outlier_voxels"] > 0


def test_no_findings_for_uniform_intensity():
    vert = _label_volume()
    t2 = np.full((12, 12, 12), 50.0)
    assert zscore_anomaly(t2, vert) == []


def test_skips_small_label():
    vert = np.zeros((12, 12, 12), np.int32)
    vert[0, 0, 0] = 8
    t2 = np.zeros((12, 12, 12))
    assert zscore_anomaly(t2, vert) == []


def test_findings_sorted_descending():
    vert = np.zeros((12, 24, 12), np.int32)
    vert[2:10, 2:10, 2:10] = 8
    vert[2:10, 14:22, 2:10] = 9
    t2 = np.full((12, 24, 12), 100.0)
    # label 8 gets fewer outliers than label 9
    for x, y, z in np.argwhere(vert == 8)[:10]:
        t2[x, y, z] = 5000.0
    for x, y, z in np.argwhere(vert == 9)[:30]:
        t2[x, y, z] = 5000.0
    findings = zscore_anomaly(t2, vert, min_hi=0)
    counts = [f["high_outlier_voxels"] for f in findings]
    assert counts == sorted(counts, reverse=True)
