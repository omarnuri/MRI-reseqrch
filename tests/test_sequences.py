from spine_pipeline.sequences import classify, pick_sequence


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
