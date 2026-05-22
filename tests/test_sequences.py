from spine_pipeline.sequences import (
    classify, pick_sequence, pick_any_orientation, pick_stir_or_fatsat,
)


def test_classify_stir_coronal():
    assert classify({"SeriesDescription": "T2 COR STIR"}) == ("STIR", "coronal")


def test_classify_t2_fs_sagittal():
    assert classify({"SeriesDescription": "T2 SAG FS"}) == ("T2_FS", "sagittal")


def test_classify_t2_sagittal():
    assert classify({"SeriesDescription": "T2 SAG"}) == ("T2", "sagittal")


def test_classify_t1_axial():
    assert classify({"SeriesDescription": "T1 AX"}) == ("T1", "axial")


def test_classify_tra_is_axial():
    assert classify({"SeriesDescription": "T2 TRA"})[1] == "axial"


def test_classify_by_te_tr_t2():
    assert classify({"EchoTime": 90, "RepetitionTime": 3000})[0] == "T2"


def test_classify_by_te_tr_t1():
    assert classify({"EchoTime": 10, "RepetitionTime": 500})[0] == "T1"


def test_classify_empty_is_unknown():
    assert classify({}) == ("unknown", "unknown")


def test_classify_uses_protocol_name():
    assert classify({"ProtocolName": "stir cor"})[0] == "STIR"


def test_pick_sequence_match():
    info = [
        {"sequence": "T2", "orientation": "sagittal", "nifti": "a.nii.gz"},
        {"sequence": "T1", "orientation": "sagittal", "nifti": "b.nii.gz"},
    ]
    assert pick_sequence(info, "T2", "sagittal") == "a.nii.gz"


def test_pick_sequence_no_match():
    info = [{"sequence": "T2", "orientation": "sagittal", "nifti": "a.nii.gz"}]
    assert pick_sequence(info, "STIR", "sagittal") is None


def test_pick_any_orientation_falls_back():
    info = [{"sequence": "STIR", "orientation": "coronal", "nifti": "c.nii.gz"}]
    nii, orient = pick_any_orientation(info, "STIR")
    assert nii == "c.nii.gz"
    assert orient == "coronal"


def test_pick_any_orientation_prefers_sagittal():
    info = [
        {"sequence": "STIR", "orientation": "coronal", "nifti": "c.nii.gz"},
        {"sequence": "STIR", "orientation": "sagittal", "nifti": "s.nii.gz"},
    ]
    nii, orient = pick_any_orientation(info, "STIR")
    assert nii == "s.nii.gz"
    assert orient == "sagittal"


def test_pick_any_orientation_none():
    info = [{"sequence": "T2", "orientation": "sagittal", "nifti": "a.nii.gz"}]
    nii, orient = pick_any_orientation(info, "STIR")
    assert nii is None
    assert orient is None


def test_pick_stir_or_fatsat_prefers_real_stir():
    info = [
        {"sequence": "STIR", "orientation": "coronal", "nifti": "stir.nii.gz"},
        {"sequence": "T2_FS", "orientation": "sagittal", "nifti": "fs.nii.gz"},
    ]
    nii, orient, source = pick_stir_or_fatsat(info)
    assert source == "STIR"
    assert nii == "stir.nii.gz"


def test_pick_stir_or_fatsat_falls_back_to_t2fs():
    info = [{"sequence": "T2_FS", "orientation": "sagittal", "nifti": "fs.nii.gz"}]
    nii, orient, source = pick_stir_or_fatsat(info)
    assert source == "T2_FS"
    assert orient == "sagittal"


def test_pick_stir_or_fatsat_none():
    info = [{"sequence": "T2", "orientation": "sagittal", "nifti": "a.nii.gz"}]
    assert pick_stir_or_fatsat(info) == (None, None, None)


def test_findings_8_scenario_coronal_stir_is_picked():
    # Reproduces the user's findings_8 data: only coronal STIR is available.
    # Previous code returned None and the entire STIR path stayed inactive.
    info = [
        {"sequence": "T2", "orientation": "sagittal", "nifti": "t2.nii.gz"},
        {"sequence": "STIR", "orientation": "coronal", "nifti": "stir_cor.nii.gz"},
    ]
    assert pick_sequence(info, "STIR", "sagittal") is None
    nii, orient, source = pick_stir_or_fatsat(info)
    assert nii == "stir_cor.nii.gz"
    assert orient == "coronal"
    assert source == "STIR"
