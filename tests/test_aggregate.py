import json

import numpy as np

from spine_pipeline.aggregate import build_findings, merge_per_vertebra
from spine_pipeline.serialization import to_jsonable


def test_merge_attaches_radiomics_and_anomaly():
    geom = {"per_vertebra": [{"label_id": 8, "voxel_count": 100}]}
    radiomics = [{"label_id": 8, "features": {"mean": 1.0}}]
    anomaly = {"top_findings": [
        {"label_id": 8, "high_outlier_voxels": 5, "method": "z"}]}
    merged = merge_per_vertebra(geom, radiomics, anomaly)
    assert merged[0]["radiomics"] == {"mean": 1.0}
    assert merged[0]["anomaly_T2"]["high_outlier_voxels"] == 5


def test_merge_no_match_leaves_empty_radiomics():
    geom = {"per_vertebra": [{"label_id": 8}]}
    merged = merge_per_vertebra(geom, [], {"top_findings": []})
    assert merged[0]["radiomics"] == {}
    assert "anomaly_T2" not in merged[0]


def test_build_findings_is_json_serializable():
    geom = {
        "global_metrics": {"k": np.float32(1.0)},
        "per_vertebra": [{"label_id": 8, "ap_width_mm": np.float32(12.0)}],
    }
    findings = build_findings(
        tool_status={"SPINEPS": {"status": "ok"}},
        geom=geom,
        muscle={},
        anomaly={"top_findings": [], "method": None},
        disc={"discs": [], "method": None},
        cv=[],
        radiomics_data=[],
        agreement_data={},
        sequences_used={"T2_SAG": "a.nii.gz"},
    )
    json.dumps(to_jsonable(findings))  # must not raise
    assert findings["patient_id"] == "anon"
    assert findings["tool_status"]["SPINEPS"]["status"] == "ok"
