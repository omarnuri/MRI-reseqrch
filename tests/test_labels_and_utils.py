"""Label handling, JSON safety and output discovery."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spinelab import labels as L
from spinelab.utils import (
    clean_reason,
    find_outputs,
    parse_cli_flags,
    read_json,
    to_jsonable,
    write_json,
)


class TestParseCliFlags:
    """Commands are built from what the installed tool accepts, not from its docs."""

    HELP = """
    usage: sct_detect_compression [-h] -s SEG -discfile FILE [-o OUTPUT] [-v {0,1,2}]

    optional arguments:
      -h, --help     show this help message and exit
      -s SEG         spinal cord segmentation
      -discfile FILE labels, one voxel per disc
      --qc-dataset   name of the dataset
    """

    def test_finds_single_and_double_dash_options(self):
        flags = parse_cli_flags(self.HELP)
        assert {"-s", "-discfile", "-o", "-v", "-h", "--help", "--qc-dataset"} <= flags

    def test_does_not_invent_flags_from_prose(self):
        flags = parse_cli_flags(self.HELP)
        assert "-i" not in flags and "-ascor" not in flags

    def test_empty_help_is_an_empty_set_not_a_crash(self):
        assert parse_cli_flags(None) == set()
        assert parse_cli_flags("") == set()


class TestVertebraNaming:
    def test_thoracic_and_lumbar_mapping(self):
        assert L.vertebra_name(8) == "T1"
        assert L.vertebra_name(18) == "T11"
        assert L.vertebra_name(19) == "T12"
        assert L.vertebra_name(20) == "L1"

    def test_round_trip(self):
        for name in ("C7", "T1", "T11", "L5"):
            assert L.vertebra_name(L.vertebra_label(name)) == name

    def test_unknown_name_is_not_guessed(self):
        assert L.vertebra_label("T13") is None

    def test_unknown_label_is_marked_as_raw(self):
        assert L.vertebra_name(99).startswith("id_")

    def test_sacrum_is_not_a_vertebra_for_measurement(self):
        assert L.is_vertebra(24) is True
        assert L.is_vertebra(25) is False


class TestSemanticLabelGroups:
    def test_posterior_elements_exclude_body_cord_and_canal(self):
        for excluded in (L.VERTEBRA_CORPUS, L.VERTEBRA_CORPUS_BORDER,
                         L.SPINAL_CORD, L.SPINAL_CANAL):
            assert excluded not in L.POSTERIOR_ELEMENTS

    def test_facet_labels_are_side_specific(self):
        assert set(L.FACET_LEFT) == {45, 47}
        assert set(L.FACET_RIGHT) == {46, 48}
        assert not set(L.FACET_LEFT) & set(L.FACET_RIGHT)

    def test_costal_processes_are_side_specific(self):
        assert L.COSTAL_LEFT == (43,)
        assert L.COSTAL_RIGHT == (44,)

    def test_midline_structures_are_not_in_either_side(self):
        for label in L.POSTERIOR_MIDLINE:
            assert label not in L.FACET_LEFT + L.FACET_RIGHT


class TestCorpusResolution:
    def test_picks_whichever_corpus_label_is_present(self):
        sem = np.zeros((10, 10, 10), dtype=np.int32)
        sem[2:8, 2:8, 2:8] = L.VERTEBRA_CORPUS  # 50, not the hard-coded 49
        assert L.resolve_corpus_label(sem) == L.VERTEBRA_CORPUS

    def test_prefers_the_larger_candidate(self):
        sem = np.zeros((10, 10, 10), dtype=np.int32)
        sem[0:6, 0:6, 0:6] = L.VERTEBRA_CORPUS_BORDER
        sem[8:9, 8:9, 8:9] = L.VERTEBRA_CORPUS
        assert L.resolve_corpus_label(sem) == L.VERTEBRA_CORPUS_BORDER

    def test_returns_none_when_absent(self):
        assert L.resolve_corpus_label(np.zeros((5, 5, 5), dtype=np.int32)) is None


class TestDiscLabels:
    def test_disc_threshold_excludes_cord_csf_and_vertebrae(self):
        assert L.TSS_SPINAL_CORD < L.TSS_DISC_LABEL_MIN
        assert L.TSS_CSF < L.TSS_DISC_LABEL_MIN
        assert max(k for k in L.TSS_VERTEBRAE) < L.TSS_DISC_LABEL_MIN

    def test_named_levels(self):
        assert L.tss_disc_name(81) == "T10-T11"
        assert L.tss_disc_name(100) == "L5-S"

    def test_unknown_disc_label_is_marked(self):
        assert L.tss_disc_name(64000).startswith("disc_label_")


class TestJsonSafety:
    def test_numpy_scalars_stay_numbers(self):
        payload = {"a": np.float32(1.5), "b": np.int64(3), "c": np.array([1, 2])}
        out = json.loads(json.dumps(to_jsonable(payload)))
        assert out == {"a": pytest.approx(1.5), "b": 3, "c": [1, 2]}

    def test_zooms_tuple_is_serialisable(self):
        zooms = tuple(np.float32(v) for v in (3.3, 0.6, 0.6))
        assert json.dumps(to_jsonable({"z": zooms}))

    def test_paths_and_sets(self, tmp_path):
        out = to_jsonable({"p": tmp_path, "s": {"b", "a"}})
        assert out["s"] == ["a", "b"]
        assert isinstance(out["p"], str)

    def test_round_trip_through_disk(self, tmp_path):
        path = write_json(tmp_path / "x" / "y.json", {"v": np.float64(2.5)})
        assert read_json(path)["v"] == 2.5

    def test_read_json_missing_returns_default(self, tmp_path):
        assert read_json(tmp_path / "nope.json", {"d": 1}) == {"d": 1}


class TestCleanReason:
    def test_progress_bars_are_skipped(self):
        text = "done with X\n100%|##########| 1/1 [00:00<00:00,  4.26it/s]"
        assert clean_reason(text) == "done with X"

    def test_empty_gives_fallback(self):
        assert clean_reason("", "nothing") == "nothing"

    def test_only_noise_gives_fallback(self):
        assert clean_reason("50%|#####| 1/2 [00:01<00:01,  1.00s/it]", "nothing") == "nothing"


class TestFindOutputs:
    def test_excludes_input_and_raw_directories(self, tmp_path):
        (tmp_path / "input").mkdir()
        (tmp_path / "step2_output").mkdir()
        (tmp_path / "input" / "study.nii.gz").write_bytes(b"x")
        (tmp_path / "step2_output" / "seg.nii.gz").write_bytes(b"x")
        found = find_outputs(tmp_path, "*.nii.gz")
        assert [p.name for p in found] == ["seg.nii.gz"]

    def test_missing_root_is_empty(self, tmp_path):
        assert find_outputs(tmp_path / "nope", "*.nii.gz") == []
