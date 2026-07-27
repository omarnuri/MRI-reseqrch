"""Sequence identification: the bugs that decided which images got analysed."""

from __future__ import annotations

from spinelab.sequences import (
    CONTRAST_T1,
    CONTRAST_T2,
    PLANE_AXIAL,
    PLANE_CORONAL,
    PLANE_SAGITTAL,
    Series,
    build_picks,
    classify_contrast,
    describe_series,
    has_fat_saturation,
    is_localizer,
    pick,
    pick_fat_saturated,
    plane_from_direction,
)


class TestFatSaturation:
    def test_plain_fse_is_not_fat_saturated(self):
        # The regression that mattered: 'fs' inside 'FSE' made the old pipeline
        # treat an ordinary T2 fast-spin-echo as fat-suppressed, and its bright
        # voxels were then read as oedema.
        assert has_fat_saturation({"SeriesDescription": "T2 SAG FSE"}) is False
        assert has_fat_saturation({"SeriesDescription": "T2 TSE sag"}) is False
        assert has_fat_saturation({"SeriesDescription": "t2_tse_fs_sag"}) is True

    def test_recognises_common_techniques(self):
        for desc in ("T2 COR STIR", "TIRM sag", "T2 SPAIR ax", "T1 fatsat",
                     "T2 fat_sat sag", "T2 Dixon water"):
            assert has_fat_saturation({"SeriesDescription": desc}) is True, desc

    def test_inversion_time_plus_ir_token(self):
        assert has_fat_saturation({"SeriesDescription": "sag ir", "InversionTime": 0.15}) is True

    def test_scan_options_flag(self):
        assert has_fat_saturation({"SeriesDescription": "T2 sag", "ScanOptions": "FS"}) is True


class TestContrast:
    def test_from_description(self):
        assert classify_contrast({"SeriesDescription": "T2 SAG"}) == CONTRAST_T2
        assert classify_contrast({"SeriesDescription": "T1 sag se"}) == CONTRAST_T1

    def test_stir_counts_as_t2_weighted(self):
        assert classify_contrast({"SeriesDescription": "COR STIR"}) == CONTRAST_T2

    def test_falls_back_to_te_tr(self):
        assert classify_contrast({"EchoTime": 90, "RepetitionTime": 3500}) == CONTRAST_T2
        assert classify_contrast({"EchoTime": 12, "RepetitionTime": 600}) == CONTRAST_T1

    def test_seconds_are_normalised_to_ms(self):
        # dcm2niix sidecars report seconds; DICOM reports milliseconds.
        assert classify_contrast({"EchoTime": 0.09, "RepetitionTime": 3.5}) == CONTRAST_T2


class TestPlane:
    def test_from_slice_normal(self):
        assert plane_from_direction([1, 0, 0]) == PLANE_SAGITTAL
        assert plane_from_direction([0, 1, 0]) == PLANE_CORONAL
        assert plane_from_direction([0, 0, 1]) == PLANE_AXIAL
        assert plane_from_direction([0.05, 0.1, 0.99]) == PLANE_AXIAL

    def test_description_words_do_not_leak(self):
        # 'relax' contains 'ax'; the old substring test called this axial.
        s = describe_series({"SeriesDescription": "T2 relax sag"}, path="a.nii.gz", name="a",
                            shape=(15, 320, 320), voxel_mm=(3.3, 0.6, 0.6), normal=[1, 0, 0])
        assert s.plane == PLANE_SAGITTAL

    def test_slice_count_comes_from_the_third_axis(self):
        # Real dcm2niix output for this study: 512 x 512 x 17 sagittal. Mapping the
        # plane to an axis index reported 512 "slices", which then drove selection.
        s = describe_series({"SeriesDescription": "T2 SAG"}, path="a.nii.gz", name="a",
                            shape=(512, 512, 17), voxel_mm=(0.66, 0.66, 3.5),
                            normal=[1, 0, 0])
        assert s.n_slices == 17

    def test_slice_count_for_a_coronal_stack(self):
        s = describe_series({"SeriesDescription": "T2 COR STIR", "InversionTime": 100},
                            path="c.nii.gz", name="c", shape=(512, 512, 23),
                            voxel_mm=(0.625, 0.625, 4.0), normal=[0, 1, 0])
        assert s.n_slices == 23
        assert s.plane == PLANE_CORONAL
        assert s.fat_sat is True


class TestLocalizer:
    def test_detects_survey_series(self):
        assert is_localizer({"SeriesDescription": "3pl localizer"}) is True
        assert is_localizer({"ImageType": ["ORIGINAL", "PRIMARY", "LOCALIZER"]}) is True
        assert is_localizer({"SeriesDescription": "T2 sag"}) is False

    def test_vendor_names_for_a_positioning_scan(self):
        # From the real study: "Scano_SAG" is a scout and contains none of the usual
        # words, so a token list alone would have let it into the analysis.
        for desc in ("Scano_SAG", "scanogram", "topogram sag"):
            assert is_localizer({"SeriesDescription": desc}) is True, desc

    def test_thick_short_slab_is_a_localizer_whatever_it_is_called(self):
        assert is_localizer({"SeriesDescription": "series 1", "SliceThickness": 10.0},
                            n_slices=5) is True

    def test_a_real_series_is_not_caught_by_the_geometric_rule(self):
        # 4 mm x 23 slices (the coronal STIR) and 3.5 mm x 17 (the sagittal T2).
        assert is_localizer({"SeriesDescription": "T2 COR STIR", "SliceThickness": 4.0},
                            n_slices=23) is False
        assert is_localizer({"SeriesDescription": "T2 SAG", "SliceThickness": 3.5},
                            n_slices=17) is False

    def test_thickness_alone_is_not_enough(self):
        # A thick series with many slices is a legitimate acquisition choice.
        assert is_localizer({"SeriesDescription": "T2 ax", "SliceThickness": 8.0},
                            n_slices=40) is False


def _series(**kw) -> Series:
    base = dict(path=kw.get("name", "x") + ".nii.gz", name=kw.get("name", "x"),
                contrast=CONTRAST_T2, plane=PLANE_SAGITTAL, n_slices=15,
                voxel_mm=(3.3, 0.6, 0.6), fat_sat=False, localizer=False)
    base.update(kw)
    return Series(**base)


class TestPicking:
    def test_localizer_is_never_picked(self):
        localizer = _series(name="loc", n_slices=3, localizer=True)
        real = _series(name="real", n_slices=15)
        assert pick([localizer, real], contrast=CONTRAST_T2, plane=PLANE_SAGITTAL) is real

    def test_prefers_more_slices_then_finer_resolution(self):
        coarse = _series(name="coarse", n_slices=12, voxel_mm=(4.0, 1.2, 1.2))
        rich = _series(name="rich", n_slices=20, voxel_mm=(3.0, 0.6, 0.6))
        assert pick([coarse, rich], contrast=CONTRAST_T2) is rich

    def test_fat_sat_preference_order_is_sag_then_ax_then_cor(self):
        cor = _series(name="cor", plane=PLANE_CORONAL, fat_sat=True)
        ax = _series(name="ax", plane=PLANE_AXIAL, fat_sat=True)
        assert pick_fat_saturated([cor, ax]) is ax
        assert pick_fat_saturated([cor]) is cor

    def test_build_picks_reports_missing_fat_sat_as_a_limitation(self):
        picks = build_picks([_series(name="t2sag")])
        assert picks["T2_SAG"] is not None
        assert picks["FATSAT_BEST"] is None
        assert any("fat-suppressed" in x for x in picks["limitations"])

    def test_build_picks_flags_non_sagittal_fat_sat(self):
        picks = build_picks([_series(name="t2sag"),
                             _series(name="stir", plane=PLANE_CORONAL, fat_sat=True)])
        assert picks["FATSAT_PLANE"] == PLANE_CORONAL
        assert any("coronal" in x for x in picks["limitations"])

    def test_build_picks_flags_missing_axial(self):
        picks = build_picks([_series(name="t2sag")])
        assert any("no axial T2" in x for x in picks["limitations"])

    def test_a_split_acquisition_is_reported_not_silently_halved(self):
        # dcm2niix delivered this study's 66-slice axial series as 31 + 35 volumes.
        # Picking one of them analyses half the spine, so the split must be stated.
        parts = [_series(name="ax_a", plane=PLANE_AXIAL, n_slices=31),
                 _series(name="ax_b", plane=PLANE_AXIAL, n_slices=35)]
        picks = build_picks([_series(name="t2sag")] + parts)
        assert picks["T2_AX"] == "ax_b.nii.gz"      # the larger part is used
        split_note = [x for x in picks["limitations"] if "separate volumes" in x]
        assert split_note, picks["limitations"]
        assert "31, 35" in split_note[0] and "35 slices) is analysed" in split_note[0]

    def test_a_single_volume_series_produces_no_split_warning(self):
        picks = build_picks([_series(name="t2sag"),
                             _series(name="ax", plane=PLANE_AXIAL, n_slices=66)])
        assert not any("separate volumes" in x for x in picks["limitations"])
