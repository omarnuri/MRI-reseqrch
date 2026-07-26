"""End-to-end run of the analysis stages on a synthetic study.

No GPU, no DICOM and no segmentation tool: the segmentation stages are pre-seeded
with masks built by hand, which is exactly what lets the downstream logic be
tested for the behaviours that matter here — that a left/right difference is
attributed to the correct side, that a missing fat-suppressed sequence produces a
refusal rather than a number, and that the report never contains a conclusion the
data cannot support.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from spinelab import labels as L
from spinelab.config import Config
from spinelab.pipeline import run_pipeline

# Geometry of the phantom: axis 0 = L->R (16 slices), 1 = P->A, 2 = I->S.
LR, AP, SI = 16, 64, 72
VOXEL = (3.3, 0.7, 0.7)
LEVELS = (16, 17, 18, 19, 20)  # T9, T10, T11, T12, L1
BODY_AP = slice(24, 48)
BODY_LR = slice(5, 11)
FACET_AP = slice(14, 22)
LEFT_LR = slice(1, 4)     # low index = patient LEFT in RAS
RIGHT_LR = slice(12, 15)


def _affine():
    a = np.eye(4)
    a[0, 0], a[1, 1], a[2, 2] = VOXEL
    return a


def _save(data, path):
    nib.save(nib.Nifti1Image(np.asarray(data), _affine()), str(path))
    return str(path)


def _level_slices():
    """SI extent of each vertebra, 12 voxels tall with a 2-voxel disc gap."""
    out = {}
    z = 4
    for label in LEVELS:
        out[label] = slice(z, z + 12)
        z += 14
    return out


def build_masks():
    instance = np.zeros((LR, AP, SI), dtype=np.int16)
    semantic = np.zeros((LR, AP, SI), dtype=np.int16)
    for label, z in _level_slices().items():
        instance[BODY_LR, BODY_AP, z] = label
        instance[LEFT_LR, FACET_AP, z] = label
        instance[RIGHT_LR, FACET_AP, z] = label
        semantic[BODY_LR, BODY_AP, z] = L.VERTEBRA_CORPUS_BORDER
        semantic[LEFT_LR, FACET_AP, z] = L.SUPERIOR_ARTICULAR_LEFT
        semantic[RIGHT_LR, FACET_AP, z] = L.SUPERIOR_ARTICULAR_RIGHT
        semantic[7:9, 20:24, z] = L.SPINAL_CORD
    return instance, semantic


def build_image(*, bright_level: int | None = None, bright_side: str | None = None,
                rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    image = rng.normal(100.0, 4.0, (LR, AP, SI))
    image = np.clip(image, 20.0, None)  # everything is in-field
    levels = _level_slices()
    if bright_level is not None:
        z = levels[bright_level]
        image[6:10, 30:42, z.start + 3:z.start + 9] = 320.0
    if bright_side is not None:
        lr = LEFT_LR if bright_side == "left" else RIGHT_LR
        for z in levels.values():
            image[lr, FACET_AP, z] = 300.0
    return image


def seed_study(tmp_path, *, fatsat: bool, bright_level=None, bright_side=None,
               left_out_of_field: bool = False) -> Config:
    cfg = Config(work_dir=tmp_path / "work", subject_id="phantom",
                 stages=("geometry", "marrow", "posterior", "radiomics", "report"))
    cfg.ensure_dirs()

    instance, semantic = build_masks()
    inst_path = _save(instance, cfg.nifti_dir / "instance.nii.gz")
    sem_path = _save(semantic, cfg.nifti_dir / "semantic.nii.gz")

    t2 = build_image(rng_seed=1)
    t2_path = _save(t2, cfg.nifti_dir / "t2_sag.nii.gz")

    picks = {"T2_SAG": t2_path, "T1_SAG": None, "T2_AX": None,
             "FATSAT_BEST": None, "FATSAT_PLANE": None, "FATSAT_LABEL": None,
             "limitations": []}
    if fatsat:
        stir = build_image(bright_level=bright_level, bright_side=bright_side, rng_seed=2)
        if left_out_of_field:
            stir[LEFT_LR] = 0.0
        picks.update({
            "FATSAT_BEST": _save(stir, cfg.nifti_dir / "stir_sag.nii.gz"),
            "FATSAT_PLANE": "sagittal",
            "FATSAT_LABEL": "T2 FS",
        })
    else:
        picks["limitations"].append(
            "no fat-suppressed sequence (STIR/TIRM/SPAIR/Dixon-water) in this study: "
            "bone-marrow and peri-facet oedema are NOT assessable")

    _write_stage(cfg, "ingest", {"picks": picks, "n_series": 2, "series": [
        {"description": "T2 SAG", "sequence_label": "T2", "plane": "sagittal",
         "n_slices": LR, "voxel_mm": list(VOXEL), "localizer": False, "name": "t2_sag.nii.gz"},
    ]})
    _write_stage(cfg, "spineps", {"instance_masks": [inst_path],
                                  "semantic_masks": [sem_path],
                                  "centroid_files": []})
    return cfg


def _write_stage(cfg: Config, name: str, data: dict):
    payload = {"name": name, "status": "ok", "evidence": "model", "reason": None,
               "data": data, "artifacts": [], "duration_s": 0.0}
    path = cfg.stage_dir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------


class TestFullRun:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("study")
        cfg = seed_study(tmp, fatsat=True, bright_level=18, bright_side="right")
        results = run_pipeline(cfg, log=lambda *_: None)
        return cfg, results

    def test_every_stage_completed(self, run):
        _, results = run
        for name in ("geometry", "marrow", "posterior", "radiomics", "report"):
            assert results[name].status.value in ("ok", "partial"), \
                f"{name}: {results[name].status} {results[name].reason}"

    def test_geometry_reports_named_levels_not_raw_ids(self, run):
        _, results = run
        names = results["geometry"].data["levels_measured"]
        assert names == ["T9", "T10", "T11", "T12", "L1"]

    def test_geometry_measures_on_the_vertebral_body(self, run):
        _, results = run
        data = results["geometry"].data
        assert data["corpus_label_used"] == L.VERTEBRA_CORPUS_BORDER
        assert all(v["body_based"] for v in data["per_vertebra"])

    def test_straight_phantom_shows_no_scheuermann_pattern(self, run):
        _, results = run
        data = results["geometry"].data
        assert data["scheuermann_pattern"] is False
        assert abs(data["max_wedge_angle_deg"]) < 5

    def test_body_heights_are_plausible_not_a_few_millimetres(self, run):
        # 12 voxels x 0.7 mm = 8.4 mm; the old full-mask measurement returned
        # posterior heights of 2-9 mm because the slab landed on the processes.
        _, results = run
        for v in results["geometry"].data["per_vertebra"]:
            assert 7.0 <= v["posterior_height_mm"] <= 9.0
            assert 7.0 <= v["anterior_height_mm"] <= 9.0

    def test_marrow_finds_the_planted_level(self, run):
        _, results = run
        data = results["marrow"].data
        assert data["top_candidates"][0] == "T11"
        assert data["restricted_to_vertebral_body"] is True

    def test_marrow_states_its_limits(self, run):
        _, results = run
        assert results["marrow"].evidence.value == "heuristic"
        assert any("not oedema" in x or "not oedema" in x.lower()
                   for x in results["marrow"].data["interpretation_limits"])

    def test_posterior_attributes_brightness_to_the_correct_side(self, run):
        _, results = run
        facet = results["posterior"].data["groups"]["facet"]
        assert facet["status"] == "measured"
        assert facet["bright_fraction_comparison"]["higher_side"] == "right"

    def test_posterior_reports_per_level_addresses(self, run):
        _, results = run
        levels = [r["level"] for r in results["posterior"].data["groups"]["facet"]["per_level"]]
        assert levels == ["T9", "T10", "T11", "T12", "L1"]

    def test_report_is_written_and_carries_the_disclaimer(self, run):
        cfg, results = run
        html = (cfg.results_dir / "report.html").read_text(encoding="utf-8")
        assert "НЕ диагноз" in html
        assert "ЭВРИСТИКА" in html
        assert "Чего эти данные не могут показать" in html

    def test_report_does_not_invent_a_diagnosis(self, run):
        cfg, _ = run
        html = (cfg.results_dir / "report.html").read_text(encoding="utf-8").lower()
        # Wording from the old report that asserted findings the data never
        # supported. "фасеточный синдром" itself is allowed — it appears in the
        # section explaining that this study cannot establish it.
        for phrase in ("острейший", "эпицентр", "вердикт", "идеально объясняют",
                       "скрытая патология", "нейросеть нашла"):
            assert phrase not in html, phrase

    def test_findings_bundle_records_evidence_levels(self, run):
        cfg, _ = run
        bundle = json.loads((cfg.results_dir / "findings.json").read_text(encoding="utf-8"))
        assert bundle["stages"]["posterior"]["evidence"] in ("heuristic", "not_diagnostic")
        assert bundle["stages"]["geometry"]["evidence"] == "measurement"

    def test_rerun_resumes_from_cache(self, run):
        cfg, _ = run
        again = run_pipeline(cfg, log=lambda *_: None)
        assert again["geometry"].status.value == "cached"
        # The report is always regenerated, so it can never go stale.
        assert again["report"].status.value == "ok"


class TestWithoutFatSuppression:
    def test_marrow_refuses_instead_of_producing_numbers(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=False)
        results = run_pipeline(cfg, log=lambda *_: None)
        marrow = results["marrow"]
        assert marrow.status.value == "skipped"
        assert "fat-suppressed" in marrow.reason
        assert marrow.data == {}

    def test_posterior_is_marked_not_diagnostic(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=False)
        results = run_pipeline(cfg, log=lambda *_: None)
        posterior = results["posterior"]
        assert posterior.evidence.value == "not_diagnostic"
        assert any("NO fat-suppressed" in x
                   for x in posterior.data["interpretation_limits"])

    def test_report_says_the_stage_did_not_run(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=False)
        run_pipeline(cfg, log=lambda *_: None)
        html = (cfg.results_dir / "report.html").read_text(encoding="utf-8")
        assert "Этап не дал результата" in html


class TestRegisteredMasks:
    """When `register` has moved the masks, downstream stages must use those."""

    def _seed_with_registration(self, tmp_path, *, applied: bool):
        cfg = seed_study(tmp_path, fatsat=True, bright_level=18, bright_side="right")
        cfg.stages = ("posterior", "marrow")
        # A register stage result pointing at the same masks (identity transform):
        # the numbers must not change, but the provenance flag must.
        spineps = json.loads((cfg.stage_dir / "spineps.json").read_text(encoding="utf-8"))["data"]
        _write_stage(cfg, "register", {
            "space": "fatsat",
            "applied": applied,
            "instance_mask": spineps["instance_masks"][0],
            "semantic_mask": spineps["semantic_masks"][0],
            "registration": {"translation_magnitude_mm": 1.2 if applied else 0.0,
                             "rotation_deg": 0.4 if applied else 0.0,
                             "reason": None if applied else "did not improve the metric"},
        })
        return cfg

    def test_motion_correction_is_recorded_when_applied(self, tmp_path):
        cfg = self._seed_with_registration(tmp_path, applied=True)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["posterior"].data["masks_motion_corrected"] is True
        assert results["marrow"].data["masks_motion_corrected"] is True

    def test_unapplied_registration_is_not_claimed_as_corrected(self, tmp_path):
        cfg = self._seed_with_registration(tmp_path, applied=False)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["posterior"].data["masks_motion_corrected"] is False
        assert any("header geometry" in x
                   for x in results["posterior"].data["interpretation_limits"])

    def test_without_registration_the_caveat_is_stated(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=True, bright_side="right")
        cfg.stages = ("posterior",)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["posterior"].data["masks_motion_corrected"] is False
        assert any("no motion correction" in x
                   for x in results["posterior"].data["interpretation_limits"])


class TestQualityProfile:
    def test_crosscheck_is_off_in_the_standard_profile(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=True)
        cfg.stages = ("crosscheck",)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["crosscheck"].status.value == "skipped"
        assert "quality" in results["crosscheck"].reason

    def test_register_skips_cleanly_without_simpleitk_or_fatsat(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=False)
        cfg.stages = ("register",)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["register"].status.value == "skipped"
        assert "fat-suppressed" in results["register"].reason


class TestCoverageGuard:
    def test_side_comparison_is_refused_when_one_side_is_out_of_field(self, tmp_path):
        # This is the coronal-STIR-on-a-sagittal-grid situation that produced the
        # original "9x brighter on the right".
        cfg = seed_study(tmp_path, fatsat=True, bright_side="right",
                         left_out_of_field=True)
        results = run_pipeline(cfg, log=lambda *_: None)
        facet = results["posterior"].data["groups"]["facet"]
        assert facet["status"] == "not_comparable"
        assert "coverage" in facet["reason"] or "in-field" in facet["reason"]
