"""Which TotalSpineSeg file is "the segmentation" — decided once, by the producer.

Three stages each guessed this from the file names and each guessed differently, on a
run where TotalSpineSeg had succeeded:

* `discs` matched "step2" and got `step2_input`, a binary mask, then reported that the
  study had no disc labels;
* `canal` matched "canal" and got `step1_canal`, a soft map with 8999 distinct values,
  and measured a thresholded probability as if it were a segmentation;
* `agreement` matched "cord" and got `step1_cord`, another soft map, and produced a
  cord Dice of 0.12 — a statement about the file choice, not about either model.
"""

from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from spinelab.stages.seg_totalspineseg import (  # noqa: E402
    LABEL_DIRS, MAX_LABELS, MIN_LABELS, _pick_label_volume)


def _write(path, values):
    """A volume whose distinct values are exactly `values`."""
    values = list(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((8, 8, max(8, len(values))), dtype=np.float32)
    for i, v in enumerate(values):
        data[0, 0, i] = v
    data[1:4, 1:4, 1:4] = values[-1] if values else 0
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _tree(root, *, step2_output=True):
    """The layout TotalSpineSeg really writes."""
    out = []
    out.append(_write(root / "step1_canal" / "s.nii.gz", np.linspace(0, 1, 300)))
    out.append(_write(root / "step1_cord" / "s.nii.gz", np.linspace(0, 1, 250)))
    out.append(_write(root / "step1_output" / "s.nii.gz", list(range(0, 19)) + [92]))
    out.append(_write(root / "step2_input" / "s.nii.gz", [0, 1]))
    if step2_output:
        out.append(_write(root / "step2_output" / "s.nii.gz",
                          list(range(0, 30)) + [63, 100]))
    out.append(_write(root / "step2_raw" / "s.nii.gz", list(range(0, 10))))
    return sorted(out)


class TestPickLabelVolume:
    def test_step2_output_wins_over_step2_input(self, tmp_path):
        chosen, note = _pick_label_volume(_tree(tmp_path))
        assert chosen.parent.name == "step2_output"
        assert "step2_output" in note

    def test_a_soft_probability_map_is_never_chosen(self, tmp_path):
        # step1_canal and step1_cord have hundreds of distinct values. Thresholding
        # one of those at 0.5 is what produced the 0.12 cord Dice.
        chosen, _ = _pick_label_volume(_tree(tmp_path))
        assert chosen.parent.name not in ("step1_canal", "step1_cord")

    def test_step1_output_is_the_fallback_when_step2_is_missing(self, tmp_path):
        chosen, note = _pick_label_volume(_tree(tmp_path, step2_output=False))
        assert chosen.parent.name == "step1_output"
        assert "step1_output" in note

    def test_nothing_usable_is_reported_rather_than_guessed(self, tmp_path):
        only_soft = [_write(tmp_path / "step1_cord" / "s.nii.gz", np.linspace(0, 1, 300))]
        chosen, note = _pick_label_volume(only_soft)
        assert chosen is None
        assert "step2_output" in note and "step1_output" in note

    def test_a_volume_without_disc_labels_is_not_a_label_volume(self, tmp_path):
        # The discriminator is reaching the disc range: a coarse mask that stops at 10
        # is not the volume the disc, canal and cord stages need.
        shallow = [_write(tmp_path / "step2_output" / "s.nii.gz", list(range(0, 11)))]
        chosen, _ = _pick_label_volume(shallow)
        assert chosen is None

    def test_the_search_order_is_final_then_coarse(self):
        assert LABEL_DIRS == ("step2_output", "step1_output")

    def test_the_value_count_window_excludes_both_failure_shapes(self):
        assert MIN_LABELS > 2, "a binary step2_input must not qualify"
        assert MAX_LABELS < 250, "a soft map with hundreds of values must not qualify"

    def test_an_unreadable_file_does_not_stop_the_search(self, tmp_path):
        broken = tmp_path / "step2_output" / "s.nii.gz"
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"not a nifti")
        good = _write(tmp_path / "step1_output" / "s.nii.gz", list(range(0, 19)) + [92])
        chosen, _ = _pick_label_volume([broken, good])
        assert chosen == good


class TestConsumersUseTheNamedVolume:
    """No stage may go back to matching on path substrings."""

    @pytest.mark.parametrize("module", ["discs", "canal", "agreement"])
    def test_no_stage_matches_step2_or_cord_in_a_path(self, module):
        from pathlib import Path

        import spinelab

        source = (Path(spinelab.__file__).parent / "stages" / f"{module}.py").read_text(
            encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        for pattern in ('"step2" in', '"cord" in str', '"canal" in str'):
            assert pattern not in code, (
                f"{module}.py is guessing at the TotalSpineSeg layout again")
        assert 'get("label_volume")' in code
