"""Final findings assembly (notebook Cell 13)."""


def merge_per_vertebra(geom, radiomics_data, anomaly):
    """Merge geometry, radiomics and anomaly findings per vertebra label."""
    rad_by_label = {r["label_id"]: r["features"] for r in radiomics_data}
    merged = []
    for v in geom.get("per_vertebra", []):
        lid = v["label_id"]
        rec = dict(v)
        rec["radiomics"] = rad_by_label.get(lid, {})
        for af in anomaly.get("top_findings", []):
            if af["label_id"] == lid:
                rec["anomaly_T2"] = {
                    "high_outlier_voxels": af.get("high_outlier_voxels"),
                    "method": af.get("method"),
                }
                break
        merged.append(rec)
    return merged


def build_findings(tool_status, geom, muscle, anomaly, disc, cv,
                   radiomics_data, agreement_data, sequences_used,
                   patient_id="anon", pipeline_version="2026-05-21"):
    """Assemble the structured findings dict written to results/findings.json."""
    per_vert_merged = merge_per_vertebra(geom, radiomics_data, anomaly)
    return {
        "patient_id": patient_id,
        "pipeline_version": pipeline_version,
        "tool_status": tool_status,
        "global_metrics": geom.get("global_metrics", {}),
        "per_vertebra": per_vert_merged,
        "per_disc": disc.get("discs", []),
        "disc_grading_method": disc.get("method"),
        "paraspinal_muscles": muscle,
        "costovertebral": cv,
        "top_anomalies": anomaly.get("top_findings", []),
        "anomaly_method": anomaly.get("method"),
        "cross_tool_agreement": agreement_data,
        "sequences_used": sequences_used,
    }
