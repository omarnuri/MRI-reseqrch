import json

from spine_pipeline.outputs import clean_reason, detect_seg_outputs, load_json


def test_detect_finds_nonstandard_output_names(tmp_path):
    # TotalSpineSeg writes the labeled seg using the INPUT basename (no
    # 'seg'/'labels'/'cord' token), which the old patterns missed.
    out = tmp_path / "out" / "step2_output"
    out.mkdir(parents=True)
    f = out / "T2_SAG_4.nii.gz"
    f.write_bytes(b"x")
    assert f in detect_seg_outputs(tmp_path / "out")


def test_detect_excludes_input_copy(tmp_path):
    out = tmp_path / "out"
    inp = out / "input_copy"
    inp.mkdir(parents=True)
    (out / "result.nii.gz").write_bytes(b"x")
    (inp / "T2_SAG_4.nii.gz").write_bytes(b"x")
    names = [p.name for p in detect_seg_outputs(out, exclude_dir=inp)]
    assert "result.nii.gz" in names
    assert "T2_SAG_4.nii.gz" not in names


def test_detect_empty(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert detect_seg_outputs(out) == []


def test_clean_reason_skips_progress_bar():
    text = ("Predicting T2_SAG_4\n"
            "done with T2_SAG_4\n"
            "100%|##########| 1/1 [00:00<00:00,  4.26it/s]")
    assert clean_reason(text) == "done with T2_SAG_4"


def test_clean_reason_handles_carriage_returns():
    assert clean_reason("step1\rstep2\rreal error message") == "real error message"


def test_clean_reason_empty_returns_fallback():
    assert clean_reason("") == "no output produced"
    assert clean_reason("   \n  ") == "no output produced"


def test_load_json_ok(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"x": 1}))
    assert load_json(p) == {"x": 1}


def test_load_json_missing_returns_default(tmp_path):
    assert load_json(tmp_path / "nope.json", default={}) == {}
