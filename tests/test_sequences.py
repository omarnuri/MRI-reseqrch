"""Sequence identification: the bugs that decided which images got analysed."""

from __future__ import annotations

import pytest

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
    group_stations,
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


class TestLocalizerWithoutADescription:
    """The 2026-08-21 Siemens study: the exporter stripped every SeriesDescription.

    Nothing textual is left, so a positioning scan has to be recognised from its
    timing alone. Values below are the real ones from that study.
    """

    SCOUT = {"SeriesDescription": "", "ProtocolName": "",
             "EchoTime": 2.38, "RepetitionTime": 4.2, "SliceThickness": 1.7}

    def test_a_fast_gradient_echo_survey_is_a_localizer(self):
        # 112 slices at 1.7 mm passes every geometric rule there was, and with
        # sidecar timings (seconds) it classifies as T1 — i.e. it would have been
        # picked as the T1 sagittal volume, beating the real 14-slice T1 TSE.
        assert is_localizer(self.SCOUT, n_slices=112) is True

    def test_the_same_rule_in_sidecar_units(self):
        # dcm2niix writes seconds; DICOM headers carry milliseconds. Both forms of
        # the same scan must be rejected.
        scout_seconds = {"EchoTime": 0.00238, "RepetitionTime": 0.0042,
                         "SliceThickness": 1.7}
        assert is_localizer(scout_seconds, n_slices=112) is True

    def test_a_single_slice_projection_slab_is_a_localizer(self):
        # Series 20 and 25: one 52 mm slab, the MIP of the 3D SPACE acquisition.
        assert is_localizer({"EchoTime": 437, "RepetitionTime": 3000,
                             "SliceThickness": 52.0}, n_slices=1) is True

    @pytest.mark.parametrize("meta,n", [
        ({"EchoTime": 96, "RepetitionTime": 2540, "SliceThickness": 3.0}, 14),    # T2 TSE sag
        ({"EchoTime": 9.5, "RepetitionTime": 703, "SliceThickness": 3.0}, 14),    # T1 TSE sag
        ({"EchoTime": 437, "RepetitionTime": 3000, "SliceThickness": 1.3}, 40),   # 3D SPACE FS
        ({"EchoTime": 84, "RepetitionTime": 6050, "SliceThickness": 3.0}, 35),    # axial T2
    ])
    def test_the_diagnostic_series_of_that_study_survive(self, meta, n):
        assert is_localizer(meta, n_slices=n) is False

    def test_a_composed_image_with_no_timing_at_all_is_not_an_analysis_input(self):
        # Series 9 and 11: the stitched whole-body positioning image. No
        # ImageOrientationPatient, no TE/TR — a derived picture, not an acquisition.
        assert is_localizer({"SeriesDescription": ""}, n_slices=109) is True

    def test_a_fat_suppressed_gradient_echo_is_not_rejected(self):
        # A Dixon/VIBE volume has scout-like timing but is a diagnostic sequence,
        # and SPINEPS has a model for it. Fat-suppression evidence protects it.
        assert is_localizer({"EchoTime": 2.4, "RepetitionTime": 6.0,
                             "ScanOptions": "FS", "SliceThickness": 1.5},
                            n_slices=64) is False


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


class TestStations:
    """A study can cover two levels of the spine in one session.

    The 2026-08-21 study does: an upper block (T2/T1 sagittal + a fat-suppressed
    3D block + an axial) and a lower one. Every pick here is a single volume per
    role, so without grouping the pipeline analyses whichever block happens to
    have one more slice and says nothing about the other.
    """

    def _upper(self):
        return [
            _series(name="t2_up", z_range_mm=(-260.0, -40.0)),
            _series(name="t1_up", contrast=CONTRAST_T1, z_range_mm=(-260.0, -40.0)),
            _series(name="ax_up", plane=PLANE_AXIAL, n_slices=35,
                    z_range_mm=(-300.0, -170.0)),
        ]

    def _lower(self):
        return [
            _series(name="t2_low", n_slices=18, z_range_mm=(-640.0, -420.0)),
            _series(name="ax_low", plane=PLANE_AXIAL, n_slices=27,
                    z_range_mm=(-776.0, -588.0)),
        ]

    def test_overlapping_volumes_form_one_station(self):
        stations = group_stations(self._upper())
        assert len(stations) == 1
        assert {s.name for s in stations[0].series} == {"t2_up", "t1_up", "ax_up"}

    def test_two_blocks_are_two_stations_numbered_from_the_top(self):
        stations = group_stations(self._upper() + self._lower())
        assert [st.id for st in stations] == [1, 2]
        assert {s.name for s in stations[0].series} == {"t2_up", "t1_up", "ax_up"}
        assert {s.name for s in stations[1].series} == {"t2_low", "ax_low"}
        assert stations[0].z_range_mm[1] > stations[1].z_range_mm[1]

    def test_a_localizer_never_joins_a_diagnostic_station(self):
        # A whole-spine positioning scan overlaps every diagnostic block; letting it
        # in would make one station out of the whole study and put a scout in the
        # candidate list for the analysis volumes.
        scout = _series(name="scout", localizer=True, z_range_mm=(-900.0, 0.0))
        stations = group_stations(self._upper() + [scout])
        diagnostic = [st for st in stations if not st.survey_only]
        assert len(diagnostic) == 1
        assert "scout" not in {s.name for s in diagnostic[0].series}

    def test_survey_blocks_can_be_left_out_entirely(self):
        scout = _series(name="scout", localizer=True, z_range_mm=(-900.0, 0.0))
        stations = group_stations(self._upper() + [scout], include_survey=False)
        assert len(stations) == 1

    def test_a_volume_without_geometry_is_reported_not_dropped(self):
        stations = group_stations(self._upper() + [_series(name="no_geom")])
        assert len(stations) == 1
        assert "no_geom" in stations[0].unplaced

    def test_picks_can_be_restricted_to_one_station(self):
        series = self._upper() + self._lower()
        # Unrestricted, the larger lower block wins the sagittal T2 role.
        assert build_picks(series)["T2_SAG"] == "t2_low.nii.gz"
        picks = build_picks(series, station=group_stations(series)[0])
        assert picks["T2_SAG"] == "t2_up.nii.gz"
        assert picks["T2_AX"] == "ax_up.nii.gz"
        assert picks["station"] == 1

    def test_a_survey_block_becomes_its_own_station_after_the_diagnostic_ones(self):
        # The 2026-08-21 study images the thoracic spine only on the AutoAlign
        # survey. Grouped together with the diagnostic volumes it would vanish:
        # a whole-spine positioning scan overlaps everything.
        scouts = [_series(name="scout_thoracic", localizer=True, n_slices=112,
                          z_range_mm=(-618.0, -220.0))]
        stations = group_stations(self._upper() + self._lower() + scouts)
        assert [(st.id, st.survey_only) for st in stations] == [(1, False), (2, False), (3, True)]
        assert stations[2].label.endswith("(survey only)")

    def test_a_survey_station_is_not_chosen_by_contrast_and_says_what_it_is(self):
        series = self._upper() + [_series(name="scout", localizer=True, n_slices=112,
                                          contrast=CONTRAST_T1, z_range_mm=(-618.0, -220.0))]
        stations = group_stations(series)
        picks = build_picks(series, station=stations[-1], stations=stations)
        assert picks["survey_only"] is True
        assert picks["T2_SAG"] == "scout.nii.gz"   # a localiser, chosen deliberately
        assert picks["FATSAT_BEST"] is None
        assert any("POSITIONING SCANS" in x for x in picks["limitations"])

    def test_the_gap_between_diagnostic_blocks_is_measured(self):
        series = self._upper() + self._lower()
        stations = group_stations(series)
        picks = build_picks(series, station=stations[0], stations=stations)
        gap = [x for x in picks["limitations"] if "no diagnostic series covers" in x]
        assert gap, picks["limitations"]
        assert "120 mm of spine" in gap[0]   # -420 (lower top) to -300 (upper bottom)

    def test_the_unanalysed_station_is_named_as_a_limitation(self):
        series = self._upper() + self._lower()
        picks = build_picks(series, station=group_stations(series)[0])
        note = [x for x in picks["limitations"] if "station" in x.lower()]
        assert note, picks["limitations"]
        assert "--station 2" in note[0]
