"""Recovering an acquisition that dcm2niix delivered as several volumes.

Real case from this study: one axial series of 66 slices in the DICOM comes out of
the default conversion as 31 + 35 NIfTI volumes. Picking one of them analyses half
the spine.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from spinelab.config import Config
from spinelab.sequences import Series, describe_series
from spinelab.stages.ingest import _describe_file, _recover_split_series


def _write_volume(path, n_slices, *, echo=103.2, description="T2 AX"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((64, 64, n_slices), dtype=np.int16)
    affine = np.diag([0.49, 0.49, 4.0, 1.0])
    # Axial acquisition: the slice direction (third column) points along z.
    nib.save(nib.Nifti1Image(data, affine), str(path))
    sidecar = str(path).replace(".nii.gz", ".json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump({"SeriesDescription": description, "EchoTime": echo,
                   "RepetitionTime": 13728.0}, fh)
    return path


@pytest.fixture
def cfg(tmp_path):
    config = Config(work_dir=tmp_path / "work")
    config.ensure_dirs()
    return config


def _parts(cfg, counts, *, echo=103.2, description="T2 AX"):
    out = []
    for i, n in enumerate(counts):
        path = _write_volume(cfg.nifti_dir / f"T2_AX_7_T2_AX_i{i:05d}.nii.gz", n,
                             echo=echo, description=description)
        out.append(_describe_file(path))
    return out


class TestRecovery:
    def test_merged_volume_replaces_the_parts(self, cfg):
        parts = _parts(cfg, (31, 35))
        assert [p.n_slices for p in parts] == [31, 35]
        _write_volume(cfg.nifti_dir / "merged" / "T2_AX_7_T2_AX.nii.gz", 66)
        merged, superseded = _recover_split_series(cfg, cfg.dicom_dir, parts)
        assert len(merged) == 1
        assert merged[0].n_slices == 66
        assert {s.path for s in superseded} == {p.path for p in parts}
        assert any("merged volume" in n for n in merged[0].notes)

    def test_a_short_merged_volume_is_rejected(self, cfg):
        # Fewer slices than the parts together: not the parts put back together.
        parts = _parts(cfg, (31, 35))
        _write_volume(cfg.nifti_dir / "merged" / "T2_AX_7_T2_AX.nii.gz", 35)
        merged, superseded = _recover_split_series(cfg, cfg.dicom_dir, parts)
        assert merged == [] and superseded == []

    def test_a_different_echo_is_rejected(self, cfg):
        # A series that genuinely holds two acquisitions must stay split rather than
        # be merged into an interleaved volume.
        parts = _parts(cfg, (31, 35), echo=103.2)
        _write_volume(cfg.nifti_dir / "merged" / "T2_AX_7_T2_AX.nii.gz", 66, echo=15.0)
        merged, _ = _recover_split_series(cfg, cfg.dicom_dir, parts)
        assert merged == []

    def test_nothing_happens_without_a_split(self, cfg):
        single = _parts(cfg, (66,))
        merged, superseded = _recover_split_series(cfg, cfg.dicom_dir, single)
        assert merged == [] and superseded == []

    def test_a_different_plane_is_not_matched(self, cfg):
        parts = _parts(cfg, (31, 35))
        # Sagittal merged volume: the slice direction runs left-right.
        path = cfg.nifti_dir / "merged" / "sag.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        affine = np.zeros((4, 4))
        affine[0, 2], affine[1, 0], affine[2, 1], affine[3, 3] = 3.5, 0.66, 0.66, 1.0
        nib.save(nib.Nifti1Image(np.zeros((64, 64, 70), dtype=np.int16), affine), str(path))
        with open(str(path).replace(".nii.gz", ".json"), "w", encoding="utf-8") as fh:
            json.dump({"SeriesDescription": "T2 AX", "EchoTime": 103.2}, fh)
        merged, _ = _recover_split_series(cfg, cfg.dicom_dir, parts)
        assert merged == []


class TestSliceCountFromRealGeometry:
    def test_axial_stack(self, cfg):
        path = _write_volume(cfg.nifti_dir / "ax.nii.gz", 66)
        series = _describe_file(path)
        assert series.n_slices == 66
        assert series.plane == "axial"

    def test_sidecar_echo_time_is_read(self, cfg):
        path = _write_volume(cfg.nifti_dir / "ax.nii.gz", 66, echo=103.2)
        assert _describe_file(path).echo_time_ms == pytest.approx(103.2)


def test_series_dataclass_keeps_notes_separate():
    a = Series(path="a", name="a")
    b = Series(path="b", name="b")
    a.notes.append("x")
    assert b.notes == []


def test_describe_series_survives_a_missing_affine():
    s = describe_series({"SeriesDescription": "T2 SAG"}, path="p", name="p",
                        shape=(512, 512, 17), voxel_mm=(0.66, 0.66, 3.5), normal=None)
    assert s.n_slices == 17
    assert s.plane == "sagittal"          # falls back to the description
