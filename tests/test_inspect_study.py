"""Series inspection straight from DICOM headers, on synthetic DICOM files."""

from __future__ import annotations

import pytest

from spinelab.inspect_study import _slice_normal, format_table, inspect

pydicom = pytest.importorskip("pydicom")


def _write_series(directory, *, uid, number, description, n=12, iop=(1, 0, 0, 0, 0, -1),
                  te=90.0, tr=3500.0, ti=None, thickness=3.3, scan_options=None,
                  image_type=("ORIGINAL", "PRIMARY", "M", "NORM")):
    """Minimal but valid single-frame MR DICOM instances."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = uid
        ds.StudyInstanceUID = "1.2.3.4"
        ds.SeriesNumber = number
        ds.InstanceNumber = i + 1
        ds.Modality = "MR"
        ds.SeriesDescription = description
        ds.ProtocolName = description
        ds.ImageOrientationPatient = list(iop)
        ds.ImagePositionPatient = [float(i) * thickness, 0.0, 0.0]
        ds.PixelSpacing = [0.7, 0.7]
        ds.SliceThickness = thickness
        ds.Rows = 320
        ds.Columns = 320
        ds.EchoTime = te
        ds.RepetitionTime = tr
        ds.MagneticFieldStrength = 1.5
        ds.ImageType = list(image_type)
        ds.PatientName = "TEST^SUBJECT"
        ds.PatientID = "TEST001"
        if ti is not None:
            ds.InversionTime = ti
        if scan_options is not None:
            ds.ScanOptions = scan_options
        ds.save_as(str(directory / f"{number:02d}_{i:03d}.dcm"), enforce_file_format=True)


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    # Sagittal T2: row along A-P, column along S-I -> normal along L-R.
    _write_series(root / "s2", uid="1.1", number=2, description="T2 SAG TSE",
                  iop=(0, 1, 0, 0, 0, -1))
    # Coronal STIR: row along L-R, column along S-I -> normal along A-P.
    _write_series(root / "s5", uid="1.2", number=5, description="T2 COR STIR",
                  iop=(1, 0, 0, 0, 0, -1), ti=150.0, n=15)
    # A 3-slice localiser that must never be treated as an analysis input.
    _write_series(root / "s1", uid="1.3", number=1, description="3pl localizer", n=3,
                  image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"))
    return root


class TestSliceNormal:
    def test_sagittal_acquisition(self):
        assert _slice_normal((0, 1, 0, 0, 0, -1)) == pytest.approx([-1.0, 0.0, 0.0])

    def test_coronal_acquisition(self):
        assert _slice_normal((1, 0, 0, 0, 0, -1)) == pytest.approx([0.0, 1.0, 0.0])

    def test_axial_acquisition(self):
        assert _slice_normal((1, 0, 0, 0, 1, 0)) == pytest.approx([0.0, 0.0, 1.0])

    def test_malformed_tag(self):
        assert _slice_normal((1, 0)) is None
        assert _slice_normal(None) is None


class TestInspectDirectory:
    def test_groups_files_into_series(self, study):
        report = inspect(study)
        assert report["n_series"] == 3
        by_number = {s["series_number"]: s for s in report["series"]}
        assert by_number[2]["n_instances"] == 12
        assert by_number[5]["n_instances"] == 15

    def test_planes_come_from_the_geometry(self, study):
        by_number = {s["series_number"]: s for s in inspect(study)["series"]}
        assert by_number[2]["plane"] == "sagittal"
        assert by_number[5]["plane"] == "coronal"

    def test_stir_is_detected_as_fat_suppressed(self, study):
        by_number = {s["series_number"]: s for s in inspect(study)["series"]}
        assert by_number[5]["fat_saturated"] is True
        assert by_number[5]["sequence_label"] == "T2 FS coronal"

    def test_plain_tse_is_not_fat_suppressed(self, study):
        by_number = {s["series_number"]: s for s in inspect(study)["series"]}
        assert by_number[2]["fat_saturated"] is False

    def test_localizer_is_flagged(self, study):
        by_number = {s["series_number"]: s for s in inspect(study)["series"]}
        assert by_number[1]["localizer"] is True

    def test_says_what_the_study_can_and_cannot_answer(self, study):
        notes = " ".join(inspect(study)["what_this_means"])
        assert "fat-suppressed series present in: coronal" in notes
        assert "not sagittal" in notes
        assert "no axial series" in notes

    def test_a_study_without_fat_sat_is_called_out(self, tmp_path):
        root = tmp_path / "plain"
        _write_series(root / "s2", uid="2.1", number=2, description="T2 SAG TSE",
                      iop=(0, 1, 0, 0, 0, -1))
        notes = " ".join(inspect(root)["what_this_means"])
        assert "no fat-suppressed series" in notes

    def test_table_renders(self, study):
        text = format_table(inspect(study))
        assert "T2 COR STIR" in text
        assert "localizer" in text


class TestInspectZip:
    def test_reads_a_zip_archive(self, study, tmp_path):
        import zipfile

        archive = tmp_path / "study.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for path in study.rglob("*.dcm"):
                zf.write(path, path.relative_to(study))
        report = inspect(archive)
        assert report["n_series"] == 3
        assert report["n_files_unreadable"] == 0
