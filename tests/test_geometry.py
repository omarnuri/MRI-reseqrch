import json

import numpy as np

from spine_pipeline.geometry import (
    LABEL_NAMES,
    angle_deg,
    body_heights_vox,
    centroid_global_metrics,
    col_si_extents,
    parse_centroids,
    slab_si_extent_vox,
    vertebra_metrics,
)
from spine_pipeline.serialization import to_jsonable


def test_angle_deg_orthogonal():
    assert abs(angle_deg(np.array([1, 0, 0]), np.array([0, 1, 0])) - 90) < 1e-6


def test_angle_deg_parallel():
    assert angle_deg(np.array([1, 1, 0]), np.array([2, 2, 0])) < 0.01


def test_parse_centroids_list_of_dicts():
    ctd = [{"label": 8, "X": 1, "Y": 2, "Z": 3},
           {"label": 9, "X": 4, "Y": 5, "Z": 6}]
    c = parse_centroids(ctd)
    assert set(c) == {8, 9}
    assert np.allclose(c[8], [1, 2, 3])


def test_parse_centroids_dict_xyz():
    assert np.allclose(parse_centroids({"8": {"X": 1, "Y": 2, "Z": 3}})[8], [1, 2, 3])


def test_parse_centroids_dict_list():
    assert np.allclose(parse_centroids({"8": [1, 2, 3, 99]})[8], [1, 2, 3])


def test_parse_centroids_ignores_non_int_keys():
    assert set(parse_centroids({"direction": [1, 0, 0], "8": [1, 2, 3]})) == {8}


def test_label_names():
    assert LABEL_NAMES[1] == "C1"
    assert LABEL_NAMES[8] == "T1"
    assert LABEL_NAMES[20] == "L1"


def test_centroid_metrics_straight_column():
    centroids = {l: np.array([0.0, 0.0, float(l)]) for l in range(8, 14)}
    m = centroid_global_metrics(centroids)
    assert m["thoracic_kyphosis_approx_deg"] < 0.01
    assert m["max_lateral_deviation_mm"] < 1e-6


def test_centroid_metrics_empty():
    assert centroid_global_metrics({}) == {}


def test_slab_si_extent():
    slab = np.zeros((3, 10), bool)
    slab[0, 2:7] = True
    assert slab_si_extent_vox(slab) == 5


def test_slab_si_extent_empty():
    assert slab_si_extent_vox(np.zeros((3, 10), bool)) == 0


def _one_vertebra_volume():
    # canonical RAS+: axis0 L->R, axis1 P->A, axis2 I->S
    data = np.zeros((20, 30, 40), np.int32)
    data[5:15, 8:22, 10:30] = 8
    return data


def test_vertebra_metrics_symmetric_box():
    recs = vertebra_metrics(_one_vertebra_volume(), zooms=(1.0, 1.0, 1.0))
    assert len(recs) == 1
    r = recs[0]
    assert r["label_id"] == 8 and r["label_name"] == "T1"
    assert r["ap_width_mm"] > 0
    assert r["post_height_mm"] > 0 and r["ant_height_mm"] > 0
    # symmetric block => ratio ~1, wedge ~0
    assert abs(r["ant_post_height_ratio"] - 1.0) < 0.05
    assert abs(r["wedge_angle_deg"]) < 1.0


def test_vertebra_metrics_json_serializable_with_float32_zooms():
    # Regression for the crash: float32 zooms must not produce float32 outputs.
    zooms = np.array([0.5, 0.6, 0.7], dtype=np.float32)  # like header.get_zooms()
    recs = vertebra_metrics(_one_vertebra_volume(), zooms=zooms)
    json.dumps(to_jsonable({"global_metrics": {}, "per_vertebra": recs}))
    assert isinstance(recs[0]["ap_width_mm"], float)
    assert isinstance(recs[0]["wedge_angle_deg"], float)


def test_vertebra_metrics_skips_tiny_labels():
    data = np.zeros((20, 30, 40), np.int32)
    data[0, 0, 0] = 8  # 1 voxel, below min_voxels
    assert vertebra_metrics(data, (1, 1, 1)) == []


def test_vertebra_metrics_ignores_labels_over_24():
    data = np.zeros((20, 30, 40), np.int32)
    data[5:15, 8:22, 10:30] = 50
    assert vertebra_metrics(data, (1, 1, 1)) == []


def test_col_si_extents_per_column():
    sag = np.zeros((4, 10), bool)
    sag[1, 2:7] = True   # SI extent 5
    sag[2, 0:10] = True  # SI extent 10
    e = col_si_extents(sag)
    assert list(e) == [0, 5, 10, 0]


def test_body_heights_uniform_block():
    # axis0=L->R, axis1=P->A, axis2=I->S; a clean body block => ant ~ post
    body = np.zeros((6, 20, 30), bool)
    body[1:5, 4:16, 5:25] = True   # AP span 12, SI height 20
    post_h, ant_h, span = body_heights_vox(body, lr_axis=0)
    assert span == 12
    assert abs(post_h - 20) < 1.5 and abs(ant_h - 20) < 1.5


def test_body_heights_detects_anterior_wedge():
    # SI height shrinks toward the anterior (high AP index) => posterior taller
    body = np.zeros((6, 24, 40), bool)
    for j in range(4, 20):
        h = 30 - (j - 4)            # 30 (posterior) down to 15 (anterior)
        body[1:5, j, 5:5 + h] = True
    post_h, ant_h, span = body_heights_vox(body, lr_axis=0)
    assert post_h > ant_h           # positive (anterior) wedge — Scheuermann pattern


def test_body_heights_too_small_returns_none():
    body = np.zeros((6, 20, 30), bool)
    body[2, 2:5, 5:8] = True        # < 6 AP columns
    assert body_heights_vox(body, lr_axis=0) is None
