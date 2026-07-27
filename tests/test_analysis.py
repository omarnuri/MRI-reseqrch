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
    curvature_metrics,
    facet_interface,
    dice,
    label_agreement,
    longest_run,
    midline_index,
    mirror_side_labels,
    modified_z,
    robust_threshold,
    screen_region,
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
