"""Array maths, on synthetic phantoms with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from spinelab import labels as L
from spinelab.analysis import (
    asymmetry_ratio,
    binary_dilate,
    binary_erode,
    body_heights,
    bright_fraction,
    canal_area_profile,
    classify_compression,
    curvature_metrics,
    disc_labels_for_sct,
    facet_interface,
    dice,
    sct_disc_value,
    label_agreement,
    longest_run,
    midline_index,
    mirror_side_labels,
    centroids_si,
    label_table,
    levels_from_disc_anchors,
    mean_thickness_mm,
    modified_z,
    percentile_of,
    sct_level_index,
    sct_level_name,
    segmental_angles_deg,
    slab_area_mm2,
    slice_counts,
    robust_threshold,
    screen_region,
    summarise_distribution,
    side_masks_from_labels,
    split_by_midline,
    wedge_angle_deg,
)


def make_body(*, ap=20, si_anterior=16, si_posterior=16, lr=8, shape=(24, 40, 40)):
    """A block vertebral body in canonical RAS (axis0=L->R, 1=P->A, 2=I->S).

    Anterior = high index on axis 1. Heights are set per end so wedging is known.
    """
    vol = np.zeros(shape, dtype=bool)
    x0 = (shape[0] - lr) // 2
    y0 = (shape[1] - ap) // 2
    z_centre = shape[2] // 2
    for j in range(ap):
        # linear ramp from posterior height to anterior height
        frac = j / max(ap - 1, 1)
        height = int(round(si_posterior + frac * (si_anterior - si_posterior)))
        z0 = z_centre - height // 2
        vol[x0:x0 + lr, y0 + j, z0:z0 + height] = True
    return vol


class TestBodyHeights:
    def test_symmetric_body_has_equal_heights(self):
        heights = body_heights(make_body(si_anterior=16, si_posterior=16))
        assert heights is not None
        assert heights.anterior_voxels == pytest.approx(heights.posterior_voxels, abs=1)

    def test_anterior_wedging_is_detected(self):
        heights = body_heights(make_body(si_anterior=10, si_posterior=16))
        assert heights.anterior_voxels < heights.posterior_voxels

    def test_returns_none_for_a_sliver(self):
        vol = np.zeros((10, 10, 10), dtype=bool)
        vol[5, 5, 5] = True
        assert body_heights(vol) is None

    def test_rounded_corners_do_not_dominate(self):
        # A body whose extreme columns are 1 voxel tall: the edge inset must keep
        # those out of the height estimate (the old code let them in and produced
        # 2-9 mm "posterior heights").
        vol = make_body(si_anterior=16, si_posterior=16)
        vol[:, 10, :] = False
        vol[:, 10, 20] = True
        heights = body_heights(vol)
        assert heights.posterior_voxels > 10


class TestWedgeAngle:
    def test_sign_convention_positive_means_anterior_shorter(self):
        assert wedge_angle_deg(20.0, 15.0, 30.0) > 0
        assert wedge_angle_deg(15.0, 20.0, 30.0) < 0

    def test_equal_heights_give_zero(self):
        assert wedge_angle_deg(18.0, 18.0, 30.0) == pytest.approx(0.0)

    def test_known_value(self):
        # atan2(5, 30) = 9.46 deg
        assert wedge_angle_deg(20.0, 15.0, 30.0) == pytest.approx(9.46, abs=0.05)

    def test_rejects_nonpositive(self):
        assert wedge_angle_deg(0.0, 15.0, 30.0) is None


class TestLongestRun:
    def test_counts_contiguous_only(self):
        values = [6, 6, 1, 6, 6, 6, 2]
        assert longest_run(values, lambda v: v >= 5) == 3

    def test_none_values_break_the_run(self):
        values = [6, None, 6]
        assert longest_run(values, lambda v: v is not None and v >= 5) == 1


class TestModifiedZ:
    def test_constant_region_yields_no_outliers(self):
        # MAD == 0. The old code divided by 1e-6 instead, so one stray voxel became
        # a z-score in the thousands and produced 5009 "outlier voxels".
        values = np.array([100.0] * 50 + [400.0])
        z = modified_z(values)
        assert np.all(z == 0)

    def test_detects_a_real_outlier(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(100, 5, 500), [400.0]])
        z = modified_z(values)
        assert (z > 3.5).sum() == 1


class TestScreenRegion:
    def test_low_coverage_region_is_excluded_not_ranked(self):
        image = np.zeros((10, 10, 10))
        image[:2] = 100.0  # only 20% of the mask carries signal
        mask = np.ones((10, 10, 10), dtype=bool)
        result = screen_region(image, mask, label=18, name="T11", min_voxels=10)
        assert result.excluded_reason == "below_fov_coverage_threshold"
        assert result.outlier_voxels == 0

    def test_uniform_region_reports_no_outliers(self):
        image = np.full((10, 10, 10), 100.0)
        mask = np.ones((10, 10, 10), dtype=bool)
        result = screen_region(image, mask, label=18, name="T11", min_voxels=10)
        assert result.excluded_reason is None
        assert result.outlier_voxels == 0

    def test_bright_focus_is_counted(self):
        rng = np.random.default_rng(1)
        image = rng.normal(100, 4, (12, 12, 12))
        image[2:5, 2:5, 2:5] = 300.0
        mask = np.ones((12, 12, 12), dtype=bool)
        result = screen_region(image, mask, label=18, name="T11", min_voxels=10)
        # The 27 planted voxels, plus at most a couple of noise voxels from the
        # normal background (1728 samples at a 3.5 modified-z cut).
        assert 27 <= result.outlier_voxels <= 32

    def test_too_small_region_returns_none(self):
        image = np.ones((5, 5, 5))
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 0, 0] = True
        assert screen_region(image, mask, label=1, name="C1") is None


class TestBrightFraction:
    def test_threshold_is_computed_inside_the_roi(self):
        image = np.zeros((10, 10, 10))
        image[:5] = 10.0     # ROI
        image[5:] = 1000.0   # bright CSF-like structure outside the ROI
        roi = np.zeros((10, 10, 10), dtype=bool)
        roi[:5] = True
        stats = bright_fraction(image, roi, percentile=95.0)
        # With a whole-volume percentile the ROI would contain zero bright voxels
        # (or all of them); inside the ROI the threshold tracks the ROI itself.
        assert stats["threshold"] == 10.0
        assert stats["roi_voxels"] == 500

    def test_empty_roi_is_safe(self):
        stats = bright_fraction(np.ones((4, 4, 4)), np.zeros((4, 4, 4), dtype=bool))
        assert stats["bright_fraction"] == 0.0


class TestRobustThreshold:
    def test_threshold_from_reference_tissue_survives_a_dominant_bright_side(self):
        # Half of the paired region is uniformly bright. An in-region percentile
        # lands inside that bright population and reports "nothing is bright" on
        # either side; a reference-tissue threshold still separates them.
        rng = np.random.default_rng(3)
        marrow = rng.normal(100, 5, 4000)
        left = rng.normal(100, 5, 1000)
        right = np.full(1000, 300.0)
        threshold = robust_threshold(marrow, k=3.0)
        assert threshold is not None and 110 < threshold < 160
        assert (left > threshold).mean() < 0.05
        assert (right > threshold).mean() == 1.0

        pooled = np.concatenate([left, right])
        saturated = float(np.percentile(pooled, 95))
        assert (right > saturated).mean() == 0.0  # the failure mode being avoided

    def test_returns_none_for_a_constant_reference(self):
        assert robust_threshold(np.full(100, 50.0)) is None

    def test_returns_none_for_a_tiny_reference(self):
        assert robust_threshold(np.array([1.0, 2.0, 3.0])) is None

    def test_explicit_threshold_overrides_percentile(self):
        image = np.zeros((6, 6, 6))
        image[:3] = 100.0
        image[3:] = 200.0
        roi = np.ones((6, 6, 6), dtype=bool)
        stats = bright_fraction(image, roi, threshold=150.0)
        assert stats["threshold"] == 150.0
        assert stats["bright_fraction"] == pytest.approx(0.5)


class TestAsymmetry:
    def test_ratio_and_direction(self):
        out = asymmetry_ratio(10.0, 20.0)
        assert out["ratio"] == 2.0
        assert out["higher_side"] == "right"
        assert out["diff_pct"] == pytest.approx(66.67, abs=0.01)

    def test_zero_side_reports_undefined_instead_of_inf(self):
        out = asymmetry_ratio(0.0, 5.0)
        assert out["ratio"] is None
        assert "undefined" in out["note"]

    def test_both_zero(self):
        out = asymmetry_ratio(0.0, 0.0)
        assert out["higher_side"] is None


class TestSides:
    def test_labels_decide_sides_not_array_geometry(self):
        # Put the LEFT-labelled structure on the high-index side of the array. A
        # midline split would call it right; the labels must win.
        sem = np.zeros((20, 10, 10), dtype=np.int32)
        sem[15:18, 4:6, 4:6] = L.SUPERIOR_ARTICULAR_LEFT
        sem[2:5, 4:6, 4:6] = L.SUPERIOR_ARTICULAR_RIGHT
        left, right = side_masks_from_labels(sem, L.FACET_LEFT, L.FACET_RIGHT)
        assert left.sum() == 3 * 2 * 2
        assert right.sum() == 3 * 2 * 2
        assert left[16, 4, 4] and not right[16, 4, 4]

    def test_midline_follows_anatomy_not_the_array_centre(self):
        mask = np.zeros((40, 10, 10), dtype=bool)
        mask[28:32] = True  # spinous process sits off-centre in the FOV
        assert midline_index(mask) == pytest.approx(29.5, abs=0.6)

    def test_split_by_midline_maps_low_index_to_left_in_ras(self):
        mask = np.ones((10, 4, 4), dtype=bool)
        left, right = split_by_midline(mask, 5)
        assert left[:5].all() and not left[5:].any()
        assert right[5:].all() and not right[:5].any()


class TestMorphology:
    def test_dilation_grows_by_one_voxel_per_iteration(self):
        mask = np.zeros((9, 9, 9), dtype=bool)
        mask[4, 4, 4] = True
        assert binary_dilate(mask, 1).sum() == 7        # centre + 6 neighbours
        assert binary_dilate(mask, 2).sum() == 25

    def test_dilation_does_not_wrap_around_the_volume(self):
        # np.roll would grow the last slice into the first one.
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[4, 2, 2] = True
        grown = binary_dilate(mask, 1)
        assert not grown[0].any()

    def test_erosion_removes_the_boundary_layer(self):
        mask = np.zeros((9, 9, 9), dtype=bool)
        mask[2:7, 2:7, 2:7] = True
        assert binary_erode(mask, 1).sum() == 3 ** 3

    def test_erosion_does_not_eat_the_volume_faces(self):
        # A body touching the field of view must lose its rim to anatomy, not to
        # the array border.
        mask = np.ones((6, 6, 6), dtype=bool)
        assert binary_erode(mask, 1).all()

    def test_zero_iterations_is_identity(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[1, 1, 1] = True
        assert np.array_equal(binary_dilate(mask, 0), mask)
        assert np.array_equal(binary_erode(mask, 0), mask)


class TestFacetInterface:
    def _masks(self):
        """Two stacked vertebrae, each with articular processes, one voxel apart."""
        sem = np.zeros((12, 12, 24), dtype=np.int32)
        inst = np.zeros((12, 12, 24), dtype=np.int32)
        # upper vertebra = label 16, occupies z 4..12; lower = 17, z 14..22
        inst[4:8, 4:8, 4:12] = 16
        inst[4:8, 4:8, 14:22] = 17
        sem[4:8, 4:8, 8:12] = L.INFERIOR_ARTICULAR_LEFT   # bottom of the upper one
        sem[4:8, 4:8, 14:18] = L.SUPERIOR_ARTICULAR_LEFT  # top of the lower one
        return sem, inst

    def test_interface_lies_between_the_two_processes(self):
        sem, inst = self._masks()
        roi = facet_interface(sem, inst, upper_label=16, lower_label=17,
                              inferior_process=L.INFERIOR_ARTICULAR_LEFT,
                              superior_process=L.SUPERIOR_ARTICULAR_LEFT, dilate=2)
        assert roi.any()
        zs = np.flatnonzero(roi.any(axis=(0, 1)))
        # The gap is z 12..14; the interface must sit there, not inside the bones.
        assert zs.min() >= 10 and zs.max() <= 15

    def test_no_interface_when_a_process_is_missing(self):
        sem, inst = self._masks()
        sem[sem == L.SUPERIOR_ARTICULAR_LEFT] = 0
        roi = facet_interface(sem, inst, upper_label=16, lower_label=17,
                              inferior_process=L.INFERIOR_ARTICULAR_LEFT,
                              superior_process=L.SUPERIOR_ARTICULAR_LEFT)
        assert not roi.any()

    def test_sides_do_not_leak_into_each_other(self):
        sem, inst = self._masks()
        roi_right = facet_interface(sem, inst, upper_label=16, lower_label=17,
                                    inferior_process=L.INFERIOR_ARTICULAR_RIGHT,
                                    superior_process=L.SUPERIOR_ARTICULAR_RIGHT)
        assert not roi_right.any()

    def test_non_adjacent_levels_do_not_form_a_joint(self):
        sem, inst = self._masks()
        roi = facet_interface(sem, inst, upper_label=16, lower_label=99,
                              inferior_process=L.INFERIOR_ARTICULAR_LEFT,
                              superior_process=L.SUPERIOR_ARTICULAR_LEFT)
        assert not roi.any()


class TestMirrorTTA:
    """Mirror test-time augmentation: does the model keep sides straight?"""

    def _semantic(self):
        sem = np.zeros((20, 8, 8), dtype=np.int32)
        sem[2:5, 3:5, 3:5] = L.SUPERIOR_ARTICULAR_LEFT     # 45
        sem[15:18, 3:5, 3:5] = L.SUPERIOR_ARTICULAR_RIGHT   # 46
        sem[6:8, 3:5, 3:5] = L.COSTAL_PROCESS_LEFT          # 43
        sem[12:14, 3:5, 3:5] = L.COSTAL_PROCESS_RIGHT       # 44
        sem[9:11, 3:5, 3:5] = L.SPINOSUS_PROCESS            # 42, midline
        return sem

    def test_swap_is_an_involution(self):
        sem = self._semantic()
        assert np.array_equal(mirror_side_labels(mirror_side_labels(sem)), sem)

    def test_midline_labels_are_untouched(self):
        sem = self._semantic()
        swapped = mirror_side_labels(sem)
        assert np.array_equal(swapped == L.SPINOSUS_PROCESS, sem == L.SPINOSUS_PROCESS)

    def test_a_side_consistent_model_round_trips(self):
        # The full mirror-TTA path. The key subtlety: a model reads sides off the
        # *appearance* of the volume it is given, so on a mirrored study the
        # structure that is anatomically left now looks right and gets the right
        # label. Simulating the model as identity-on-labels would be wrong.
        sem = self._semantic()

        def perfect_model(volume):
            """Labels by appearance: geometry as given, sides named by position."""
            return mirror_side_labels(volume) if volume is mirrored else volume

        mirrored = np.flip(sem, axis=0)          # what the segmenter is fed
        prediction = perfect_model(mirrored)      # its output, sides named by appearance
        restored = mirror_side_labels(np.flip(prediction, axis=0))
        assert np.array_equal(restored, sem)

    def test_a_model_that_confuses_sides_is_detected(self):
        sem = self._semantic()
        confused = sem.copy()
        # Simulate a model that put the right facet label on the left structure.
        confused[confused == L.SUPERIOR_ARTICULAR_LEFT] = L.SUPERIOR_ARTICULAR_RIGHT
        agreement = label_agreement(sem, confused, (45, 46, 47, 48))
        assert agreement["per_label_dice"]["45"] == 0.0
        assert agreement["min_dice"] == 0.0


class TestLabelAgreement:
    def test_identical_volumes_score_one(self):
        sem = np.zeros((6, 6, 6), dtype=np.int32)
        sem[1:3] = 45
        out = label_agreement(sem, sem, (45,))
        assert out["per_label_dice"]["45"] == 1.0
        assert out["mean_dice"] == 1.0

    def test_absent_label_is_none_not_zero(self):
        sem = np.zeros((4, 4, 4), dtype=np.int32)
        out = label_agreement(sem, sem, (45,))
        assert out["per_label_dice"]["45"] is None
        assert out["mean_dice"] is None

    def test_reports_the_weakest_label(self):
        a = np.zeros((10, 4, 4), dtype=np.int32)
        b = np.zeros((10, 4, 4), dtype=np.int32)
        a[0:4] = 45
        b[0:4] = 45          # perfect
        a[5:9] = 46
        b[8:9] = 46          # poor
        out = label_agreement(a, b, (45, 46))
        assert out["weakest_label"] == "46"
        assert out["per_label_dice"]["45"] == 1.0


class TestCanal:
    def test_profile_trims_partial_end_slices(self):
        mask = np.zeros((20, 20, 12), dtype=bool)
        mask[5:15, 5:15, :] = True     # 100 voxels per slice
        mask[5:7, 5:7, 0] = True       # first slice partially covered
        mask[5:15, 5:15, 0] = False
        mask[5:7, 5:7, 0] = True
        profile = canal_area_profile(mask, voxel_area_mm2=1.0, min_area_mm2=3.0)
        # The 4 mm^2 end slice must not become "the narrowest point".
        assert profile["min_area_mm2"] == 100.0
        assert profile["max_narrowing_pct"] == 0.0

    def test_detects_a_genuine_mid_narrowing(self):
        mask = np.zeros((20, 20, 12), dtype=bool)
        mask[5:15, 5:15, :] = True
        mask[5:15, 5:15, 6] = False
        mask[5:11, 5:11, 6] = True     # 36 vs 100 mm^2 in the middle
        profile = canal_area_profile(mask, voxel_area_mm2=1.0, min_area_mm2=3.0)
        assert profile["min_area_mm2"] == 36.0
        assert profile["max_narrowing_pct"] == pytest.approx(64.0, abs=0.1)
        assert profile["slice_index_of_min"] == 6

    def test_too_few_slices_is_reported_not_guessed(self):
        mask = np.zeros((10, 10, 3), dtype=bool)
        mask[2:4, 2:4, 1] = True
        profile = canal_area_profile(mask, 1.0)
        assert "note" in profile


class TestCurvature:
    def test_straight_chain_has_near_zero_angle(self):
        centroids = {8 + i: np.array([0.0, 0.0, float(i) * 10]) for i in range(12)}
        out = curvature_metrics(centroids, L.THORACIC_LABELS)
        assert out["chain_angle_deg"] == pytest.approx(0.0, abs=1e-6)
        assert out["max_lateral_deviation_mm"] == pytest.approx(0.0, abs=1e-6)

    def test_bent_chain_reports_angle_and_deviation(self):
        centroids = {}
        for i in range(12):
            y = 0.0 if i < 6 else float(i - 5) * 5
            centroids[8 + i] = np.array([0.0, y, float(i) * 10])
        out = curvature_metrics(centroids, L.THORACIC_LABELS)
        assert out["chain_angle_deg"] > 5
        assert out["max_lateral_deviation_mm"] > 1

    def test_too_few_levels_returns_empty(self):
        assert curvature_metrics({8: np.zeros(3)}, L.THORACIC_LABELS) == {}


class TestDice:
    def test_identical_masks(self):
        a = np.zeros((5, 5, 5), dtype=bool)
        a[1:4, 1:4, 1:4] = True
        assert dice(a, a) == 1.0

    def test_disjoint_masks(self):
        a = np.zeros((5, 5, 5), dtype=bool)
        b = np.zeros((5, 5, 5), dtype=bool)
        a[0] = True
        b[4] = True
        assert dice(a, b) == 0.0

    def test_empty_pair_is_zero_not_nan(self):
        z = np.zeros((3, 3, 3), dtype=bool)
        assert dice(z, z) == 0.0


class TestSctDiscValue:
    """SCT numbers a disc by the vertebra below it: C3/C4 is 4, C6/C7 is 7.

    Getting this off by one would move every measurement one level and the result
    would still look entirely plausible, which is why it is tested rather than
    trusted. The two anchors come from the sct_detect_compression documentation.
    """

    @pytest.mark.parametrize("name,value", [
        ("C2-C3", 3), ("C3-C4", 4), ("C4-C5", 5), ("C5-C6", 6), ("C6-C7", 7),
        ("C7-T1", 8), ("T1-T2", 9), ("T12-L1", 20), ("L1-L2", 21), ("L5-S", 25),
    ])
    def test_known_levels(self, name, value):
        assert sct_disc_value(name) == value

    def test_separator_and_case_do_not_matter(self):
        assert sct_disc_value("c5/c6") == sct_disc_value("C5-C6") == 6

    def test_unknown_names_are_none_not_a_guess(self):
        for name in ("", "sacrum", "disc_label_99", "X1-X2", None):
            assert sct_disc_value(name) is None


class TestDiscLabelsForSct:
    """`-discfile` wants exactly one voxel per disc, at its posterior edge."""

    def _volume(self):
        # Two discs, canonical RAS: axis0 L->R, axis1 P->A, axis2 I->S.
        vol = np.zeros((10, 12, 20), dtype=np.int32)
        vol[3:7, 2:8, 5:8] = 64      # C3-C4 -> 4
        vol[3:7, 3:9, 12:15] = 67    # C6-C7 -> 7
        return vol

    def test_one_voxel_per_disc_with_sct_numbering(self):
        out = disc_labels_for_sct(self._volume(), L.TSS_DISCS)
        assert sorted(int(v) for v in np.unique(out) if v) == [4, 7]
        for value in (4, 7):
            assert int((out == value).sum()) == 1

    def test_the_voxel_sits_at_the_posterior_edge_and_mid_level(self):
        out = disc_labels_for_sct(self._volume(), L.TSS_DISCS)
        lr, ap, si = (int(c[0]) for c in np.nonzero(out == 4))
        assert ap == 2          # posterior = lowest index on the P->A axis
        assert si in (6, 7)     # middle of the 5..7 span
        assert lr in (4, 5)     # middle of the 3..6 span

    def test_labels_outside_the_map_are_dropped(self):
        vol = np.zeros((6, 6, 6), dtype=np.int32)
        vol[1:3, 1:3, 1:3] = 999
        assert not disc_labels_for_sct(vol, L.TSS_DISCS).any()

    def test_levels_can_be_restricted(self):
        out = disc_labels_for_sct(self._volume(), L.TSS_DISCS, keep_values=range(4, 8))
        assert sorted(int(v) for v in np.unique(out) if v) == [4, 7]
        out = disc_labels_for_sct(self._volume(), L.TSS_DISCS, keep_values=(7,))
        assert sorted(int(v) for v in np.unique(out) if v) == [7]

    def test_an_empty_volume_gives_an_empty_label_file(self):
        out = disc_labels_for_sct(np.zeros((4, 4, 4), dtype=np.int32), L.TSS_DISCS)
        assert out.shape == (4, 4, 4) and not out.any()


class TestClassifyCompression:
    """Thresholds from the sct_detect_compression documentation: 0.345 and 0.451.

    The boundaries are inclusive on the 'possible' side, so they are tested
    explicitly — a `>` where a `>=` belongs silently upgrades a borderline level.
    """

    @pytest.mark.parametrize("p,expected", [
        (0.0, "no"), (0.344, "no"),
        (0.345, "possible"), (0.40, "possible"), (0.451, "possible"),
        (0.452, "yes"), (1.0, "yes"),
    ])
    def test_categories(self, p, expected):
        assert classify_compression(p) == expected

    def test_missing_probability_is_none(self):
        assert classify_compression(None) is None


class TestLevelMorphometry:
    """The measurements a reference cohort and this study share.

    A synthetic spine: alternating vertebra and disc slabs stacked along the
    superior-inferior axis (canonical axis 2), with label values chosen to be
    nothing like any real convention — the point of the level mapping is that it
    does not read them.
    """

    @staticmethod
    def _spine(n_vertebrae: int = 5, vert_mm: int = 10, disc_mm: int = 4):
        """Returns (volume, anchors, expected {label: level index})."""
        height = n_vertebrae * (vert_mm + disc_mm) + disc_mm
        volume = np.zeros((6, 6, height), dtype=np.int32)
        anchors: dict[int, float] = {}
        expected: dict[int, int] = {}
        z = height
        # Built head-first: level 8 is T1, and each vertebra sits below its own
        # disc, which is the numbering the anchors carry.
        for step, level in enumerate(range(8, 8 + n_vertebrae)):
            disc_label = 700 + step
            volume[1:5, 1:5, z - disc_mm:z] = disc_label
            anchors[level] = float(z - disc_mm / 2.0)
            z -= disc_mm
            vert_label = 900 - step
            volume[1:5, 1:5, z - vert_mm:z] = vert_label
            expected[vert_label] = level
            z -= vert_mm
        return volume, anchors, expected

    def test_levels_come_from_the_anchors_not_the_label_values(self):
        volume, anchors, expected = self._spine()
        vertebrae, discs, note = levels_from_disc_anchors(centroids_si(volume), anchors)
        # The last vertebra has no anchor below it, so it cannot be placed.
        assert vertebrae == {k: v for k, v in expected.items() if v < max(expected.values())}
        assert sorted(discs.values()) == sorted(anchors)[:len(discs)]
        assert "anchored" not in note and "placed from" in note

    def test_a_disc_sitting_on_an_anchor_is_not_counted_as_a_vertebra(self):
        volume, anchors, _ = self._spine()
        _, discs, _ = levels_from_disc_anchors(centroids_si(volume), anchors)
        assert set(discs) <= {700, 701, 702, 703, 704}

    def test_anchors_out_of_order_are_refused(self):
        volume, anchors, _ = self._spine()
        upside_down = {k: -z for k, z in anchors.items()}
        vertebrae, _, note = levels_from_disc_anchors(centroids_si(volume), upside_down)
        assert vertebrae == {} and "head to foot" in note

    def test_a_single_anchor_is_refused(self):
        volume, anchors, _ = self._spine()
        first = sorted(anchors)[0]
        vertebrae, _, note = levels_from_disc_anchors(centroids_si(volume),
                                                     {first: anchors[first]})
        assert vertebrae == {} and "fewer than two" in note

    def test_two_labels_in_one_gap_place_no_level_there(self):
        volume, anchors, expected = self._spine()
        # Split one vertebra in two along left-right: the gap is now ambiguous.
        victim = max(expected, key=lambda label: expected[label] == 8)
        volume[1:3][volume[1:3] == victim] = 999
        vertebrae, _, _ = levels_from_disc_anchors(centroids_si(volume), anchors)
        assert 8 not in vertebrae.values()

    def test_mean_thickness_is_volume_over_footprint(self):
        volume = np.zeros((4, 9, 10), dtype=np.int32)
        volume[1:3, 1:3, 2:5] = 5             # 2x2 footprint, 3 slices
        coords = label_table(volume)[5]
        assert mean_thickness_mm(coords, si_mm=2.0, ap_size=9) == pytest.approx(6.0)
        assert mean_thickness_mm(np.zeros((0, 3), dtype=int), 1.0, 9) is None

    def test_mean_thickness_does_not_collide_two_columns_into_one(self):
        # The footprint is a set of (left-right, anterior-posterior) columns. With
        # the wrong multiplier two of them share a key, the footprint shrinks and
        # the thickness comes out too large.
        volume = np.zeros((3, 40, 6), dtype=np.int32)
        volume[0, 39, 1:3] = 7                # column (0, 39)
        volume[1, 0, 1:3] = 7                 # column (1, 0) — collides at *3
        coords = label_table(volume)[7]
        assert mean_thickness_mm(coords, si_mm=1.0, ap_size=40) == pytest.approx(2.0)

    def test_slab_area_is_the_median_over_the_range(self):
        mask = np.zeros((4, 4, 6), dtype=bool)
        mask[0:2, 0:2, 2] = True              # 4 voxels
        mask[0:3, 0:2, 3] = True              # 6 voxels
        mask[0:4, 0:2, 4] = True              # 8 voxels
        counts = slice_counts(mask)
        assert counts.tolist() == [0, 0, 4, 6, 8, 0]
        assert slab_area_mm2(counts, 2, 4, voxel_area_mm2=2.0) == pytest.approx(12.0)
        assert slab_area_mm2(counts, 0, 1, voxel_area_mm2=2.0) is None

    def test_label_table_groups_every_voxel_by_label(self):
        volume = np.zeros((3, 3, 3), dtype=np.int32)
        volume[0, 0, 0] = 4
        volume[1, :, :] = 9
        table = label_table(volume)
        assert sorted(table) == [4, 9]
        assert table[4].tolist() == [[0, 0, 0]] and len(table[9]) == 9
        assert label_table(np.zeros((2, 2, 2), dtype=np.int32)) == {}

    def test_segmental_angle_sign_follows_the_direction_of_the_bend(self):
        straight = {1: np.array([0.0, 0.0, 2.0]), 2: np.array([0.0, 0.0, 1.0]),
                    3: np.array([0.0, 0.0, 0.0])}
        assert segmental_angles_deg(straight)[2] == pytest.approx(0.0, abs=1e-6)
        kyphotic = {**straight, 2: np.array([0.0, -1.0, 1.0])}
        lordotic = {**straight, 2: np.array([0.0, 1.0, 1.0])}
        assert segmental_angles_deg(kyphotic)[2] == pytest.approx(90.0)
        assert segmental_angles_deg(lordotic)[2] == pytest.approx(-90.0)

    def test_level_names_and_indices_round_trip(self):
        assert sct_level_index("T7") == 14 and sct_level_name(14) == "T7"
        assert sct_level_index("c1") == 1 and sct_level_index("L5") == 24
        assert sct_level_index("T13") is None and sct_level_name(99) is None


class TestSummariseDistribution:
    def test_known_sample(self):
        out = summarise_distribution(list(range(101)))
        assert out["n"] == 101
        assert out["median"] == 50.0
        assert out["iqr"] == [25.0, 75.0]
        assert out["p5_p95"] == [5.0, 95.0]
        assert (out["min"], out["max"]) == (0.0, 100.0)

    def test_too_few_subjects_is_none_not_a_smaller_summary(self):
        assert summarise_distribution(list(range(9))) is None
        assert summarise_distribution(list(range(10))) is not None

    def test_non_finite_values_reduce_n_and_nothing_else(self):
        values = list(range(10)) + [float("nan"), float("inf")]
        out = summarise_distribution(values)
        assert out["n"] == 10 and out["max"] == 9.0

    def test_min_n_is_overridable_for_callers_that_state_why(self):
        assert summarise_distribution([1.0, 2.0, 3.0], min_n=3)["n"] == 3


class TestPercentileOf:
    def test_position_in_a_uniform_sample(self):
        sample = list(range(100))          # 0..99
        assert percentile_of(-1, sample) == 0.0
        assert percentile_of(50, sample) == 50.5   # 50 below, itself counted as half
        assert percentile_of(1000, sample) == 100.0

    def test_ties_count_as_half_so_a_constant_sample_lands_mid(self):
        assert percentile_of(5, [5] * 20) == 50.0

    def test_missing_value_or_empty_reference_is_none(self):
        assert percentile_of(None, [1, 2, 3]) is None
        assert percentile_of(float("nan"), [1, 2, 3]) is None
        assert percentile_of(1.0, []) is None
        assert percentile_of(1.0, [float("nan")]) is None
