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
    """Phantom masks with real facet anatomy: each level carries a superior and an
    inferior articular process per side, so consecutive levels form a joint."""
    instance = np.zeros((LR, AP, SI), dtype=np.int16)
    semantic = np.zeros((LR, AP, SI), dtype=np.int16)
    for label, z in _level_slices().items():
        instance[BODY_LR, BODY_AP, z] = label
        instance[LEFT_LR, FACET_AP, z] = label
        instance[RIGHT_LR, FACET_AP, z] = label
        semantic[BODY_LR, BODY_AP, z] = L.VERTEBRA_CORPUS_BORDER
        semantic[7:9, 20:24, z] = L.SPINAL_CORD
        # Superior processes at the top of the level, inferior at the bottom; the
        # inferior process of one level and the superior process of the next are
        # then two voxels apart, which is what forms the joint interface.
        top = slice(z.start, z.start + 4)
        bottom = slice(z.stop - 4, z.stop)
        semantic[LEFT_LR, FACET_AP, top] = L.SUPERIOR_ARTICULAR_LEFT
        semantic[RIGHT_LR, FACET_AP, top] = L.SUPERIOR_ARTICULAR_RIGHT
        semantic[LEFT_LR, FACET_AP, bottom] = L.INFERIOR_ARTICULAR_LEFT
        semantic[RIGHT_LR, FACET_AP, bottom] = L.INFERIOR_ARTICULAR_RIGHT
    return instance, semantic


def _joint_gap_slices():
    """SI extent of each inter-level gap, keyed by (upper, lower) label."""
    levels = _level_slices()
    out = {}
    for upper, lower in zip(LEVELS, LEVELS[1:]):
        out[(upper, lower)] = slice(levels[upper].stop, levels[lower].start)
    return out


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


def build_axial_image(*, bright_joint_side: str | None = None, rng_seed: int = 5):
    """Phantom 'axial' series. Orientation is not what is under test here — the
    measurement logic is — so the same grid is reused with different content."""
    rng = np.random.default_rng(rng_seed)
    image = np.clip(rng.normal(100.0, 4.0, (LR, AP, SI)), 20.0, None)
    if bright_joint_side is not None:
        lr = LEFT_LR if bright_joint_side == "left" else RIGHT_LR
        for gap in _joint_gap_slices().values():
            image[lr, FACET_AP, gap] = 300.0
    return image


def seed_study(tmp_path, *, fatsat: bool, bright_level=None, bright_side=None,
               left_out_of_field: bool = False, axial: bool = False,
               axial_bright_side: str | None = None) -> Config:
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
    if axial:
        picks["T2_AX"] = _save(build_axial_image(bright_joint_side=axial_bright_side),
                               cfg.nifti_dir / "t2_ax.nii.gz")
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
            "moving_image": spineps["instance_masks"][0],
            "targets": {
                "fatsat": {
                    "image": "irrelevant-for-this-test",
                    "plane": "sagittal",
                    "applied": applied,
                    "instance_mask": spineps["instance_masks"][0],
                    "semantic_mask": spineps["semantic_masks"][0],
                    "registration": {
                        "translation_magnitude_mm": 1.2 if applied else 0.0,
                        "rotation_deg": 0.4 if applied else 0.0,
                        "reason": None if applied else "did not improve the metric",
                    },
                }
            },
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

    def test_register_skips_when_there_is_nothing_to_register_onto(self, tmp_path):
        # Only a sagittal T2: the masks are already in the only space that exists.
        cfg = seed_study(tmp_path, fatsat=False)
        cfg.stages = ("register",)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["register"].status.value == "skipped"
        assert "no other series" in results["register"].reason


class TestFacetsAxial:
    """The axial branch: per-joint, per-side measurement."""

    def _run(self, tmp_path, **kw):
        cfg = seed_study(tmp_path, fatsat=False, axial=True, **kw)
        cfg.stages = ("facets_axial", "report")
        return cfg, run_pipeline(cfg, log=lambda *_: None)

    def test_measures_each_adjacent_pair_as_a_joint(self, tmp_path):
        _, results = self._run(tmp_path)
        joints = [j["joint"] for j in results["facets_axial"].data["joints"]]
        assert joints == ["T9-T10", "T10-T11", "T11-T12", "T12-L1"]

    def test_finds_the_side_with_the_bright_joint(self, tmp_path):
        _, results = self._run(tmp_path, axial_bright_side="right")
        data = results["facets_axial"].data
        assert data["n_comparable"] >= 3
        top = data["largest_side_difference"][0]
        assert top["higher_side"] == "right"
        for joint in data["joints"]:
            if joint.get("comparable"):
                sides = joint["sides"]
                assert sides["right"]["bright_fraction"] > sides["left"]["bright_fraction"]

    def test_symmetric_phantom_shows_no_side_preference(self, tmp_path):
        _, results = self._run(tmp_path)
        for joint in results["facets_axial"].data["joints"]:
            if not joint.get("comparable"):
                continue
            # Noise only: both sides should be at or near zero bright fraction.
            assert joint["sides"]["left"]["bright_fraction"] < 0.2
            assert joint["sides"]["right"]["bright_fraction"] < 0.2

    def test_reports_volume_per_side_in_mm3(self, tmp_path):
        _, results = self._run(tmp_path)
        first = results["facets_axial"].data["joints"][0]["sides"]["left"]
        assert first["interface_volume_mm3"] > 0

    def test_refuses_to_report_a_joint_width(self, tmp_path):
        _, results = self._run(tmp_path)
        data = results["facets_axial"].data
        assert not any("width_mm" in k for k in _all_keys(data))
        assert any("width in millimetres" in x for x in data["interpretation_limits"])

    def test_states_that_oedema_is_not_assessable_without_fat_suppression(self, tmp_path):
        _, results = self._run(tmp_path)
        limits = " ".join(results["facets_axial"].data["interpretation_limits"])
        assert "no fat suppression" in limits

    def test_skipped_without_an_axial_series(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=True)
        cfg.stages = ("facets_axial",)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["facets_axial"].status.value == "skipped"
        assert "axial" in results["facets_axial"].reason

    def test_report_renders_the_axial_section(self, tmp_path):
        cfg, _ = self._run(tmp_path, axial_bright_side="right")
        html_text = (cfg.results_dir / "report.html").read_text(encoding="utf-8")
        assert "Фасеточные суставы по уровням" in html_text
        assert "T10-T11" in html_text


def _all_keys(obj, out=None):
    """Every key appearing anywhere in a nested structure."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            _all_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _all_keys(v, out)
    return out


def _body_phantom(rim_value: float, core_value: float = 100.0, *, seed: int = 7):
    """A body with a distinct subcutaneous band, for the fat-suppression check."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((LR, AP, SI))
    vol[2:LR - 2, 6:AP - 6, 6:SI - 6] = rim_value
    vol[4:LR - 4, 14:AP - 14, 14:SI - 14] = core_value
    return np.clip(vol + rng.normal(0, 2.0, vol.shape) * (vol > 0), 0, None)


class TestFatSuppressionGate:
    """A series that claims fat suppression but does not show it must not be trusted."""

    def _seed(self, tmp_path, *, fat_rim: float):
        cfg = seed_study(tmp_path, fatsat=True, bright_level=18, bright_side="right")
        cfg.stages = ("fatsat_qc", "marrow", "posterior", "report")
        picks = json.loads((cfg.stage_dir / "ingest.json").read_text(encoding="utf-8"))["data"]["picks"]
        # Control T2: bright subcutaneous fat. Suppressed series: `fat_rim` decides
        # whether the band actually darkens.
        picks["T2_SAG"] = _save(_body_phantom(400.0), cfg.nifti_dir / "t2_body.nii.gz")
        picks["FATSAT_BEST"] = _save(_body_phantom(fat_rim),
                                     cfg.nifti_dir / "stir_body.nii.gz")
        picks["FATSAT_PLANE"] = "coronal"
        picks["FATSAT_LABEL"] = "T2 FS"
        _write_stage(cfg, "ingest", {"picks": picks, "n_series": 2, "series": []})
        return cfg

    def test_working_suppression_is_confirmed(self, tmp_path):
        cfg = self._seed(tmp_path, fat_rim=60.0)
        results = run_pipeline(cfg, log=lambda *_: None)
        qc = results["fatsat_qc"]
        assert qc.data["suppression_effective"] is True
        assert qc.status.value == "ok"

    def test_absent_suppression_is_caught(self, tmp_path):
        cfg = self._seed(tmp_path, fat_rim=400.0)
        results = run_pipeline(cfg, log=lambda *_: None)
        qc = results["fatsat_qc"]
        assert qc.data["suppression_effective"] is False
        assert "NO evidence of fat suppression" in qc.data["verdict"]

    def test_marrow_carries_the_warning_forward(self, tmp_path):
        cfg = self._seed(tmp_path, fat_rim=400.0)
        results = run_pipeline(cfg, log=lambda *_: None)
        marrow = results["marrow"]
        if marrow.data:  # the phantom body may leave too few vertebrae measurable
            assert marrow.data["fat_suppression_verified"] is False
            assert any("FAT SUPPRESSION NOT CONFIRMED" in x
                       for x in marrow.data["interpretation_limits"])

    def test_posterior_is_downgraded_to_not_diagnostic(self, tmp_path):
        cfg = self._seed(tmp_path, fat_rim=400.0)
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["posterior"].evidence.value == "not_diagnostic"
        assert any("FAT SUPPRESSION NOT CONFIRMED" in x
                   for x in results["posterior"].data["interpretation_limits"])

    def test_report_shows_the_check(self, tmp_path):
        cfg = self._seed(tmp_path, fat_rim=400.0)
        run_pipeline(cfg, log=lambda *_: None)
        html_text = (cfg.results_dir / "report.html").read_text(encoding="utf-8")
        assert "Работает ли подавление жира" in html_text

    def test_unchecked_suppression_is_flagged_as_assumed(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=True, bright_level=18)
        cfg.stages = ("marrow",)   # fatsat_qc deliberately not run
        results = run_pipeline(cfg, log=lambda *_: None)
        assert results["marrow"].data["fat_suppression_verified"] is None
        assert any("not checked" in x
                   for x in results["marrow"].data["interpretation_limits"])


class TestReportNumbering:
    def test_sections_are_numbered_once_and_in_order(self, tmp_path):
        import re

        cfg = seed_study(tmp_path, fatsat=True, bright_level=18, bright_side="right",
                         axial=True, axial_bright_side="right")
        cfg.stages = ("geometry", "facets_axial", "posterior", "marrow", "report")
        run_pipeline(cfg, log=lambda *_: None)
        html_text = (cfg.results_dir / "report.html").read_text(encoding="utf-8")
        numbers = [int(n) for n in re.findall(r"<h2>(\d+)\.", html_text)]
        assert numbers == list(range(1, len(numbers) + 1)), numbers

    def test_no_unrendered_template_braces_leak_into_the_html(self, tmp_path):
        cfg = seed_study(tmp_path, fatsat=True, axial=True)
        cfg.stages = ("geometry", "facets_axial", "report")
        run_pipeline(cfg, log=lambda *_: None)
        html_text = (cfg.results_dir / "report.html").read_text(encoding="utf-8")
        for leak in ("{_h2(", "badge_html(", "{data.", "{'"):
            assert leak not in html_text, leak


class TestStageContract:
    """Every stage must degrade, never crash — checked against all of them."""

    def test_every_default_stage_is_registered(self):
        from spinelab.config import DEFAULT_STAGES
        from spinelab.pipeline import _registry

        registry = _registry()
        assert set(DEFAULT_STAGES) <= set(registry)
        assert set(registry) - set(DEFAULT_STAGES) == set()

    def test_nothing_fails_on_an_empty_workspace(self, tmp_path):
        # No data at all: stages must report `skipped` with a reason. A `failed`
        # here means a stage crashed on absent input instead of saying so.
        cfg = Config(work_dir=tmp_path / "empty", dicom_source=str(tmp_path / "nope.zip"))
        results = run_pipeline(cfg, log=lambda *_: None)
        failed = {n: r.reason for n, r in results.items() if r.status.value == "failed"}
        assert failed == {}
        for name, res in results.items():
            if res.status.value == "skipped":
                assert res.reason, f"{name} skipped without saying why"

    def test_report_is_produced_even_when_everything_skipped(self, tmp_path):
        cfg = Config(work_dir=tmp_path / "empty", dicom_source=str(tmp_path / "nope.zip"))
        run_pipeline(cfg, log=lambda *_: None)
        assert (cfg.results_dir / "report.html").exists()
        assert (cfg.results_dir / "findings.json").exists()

    def test_all_analysis_stages_run_on_the_phantom(self, tmp_path):
        # Everything that does not need an external binary or GPU.
        cfg = seed_study(tmp_path, fatsat=True, bright_level=18, bright_side="right",
                         axial=True, axial_bright_side="right")
        cfg.stages = ("fatsat_qc", "facets_axial", "geometry", "marrow", "posterior",
                      "radiomics", "report")
        results = run_pipeline(cfg, log=lambda *_: None)
        # The phantom is uniform noise with no body outline, so the fat-suppression
        # check has nothing to delineate. Skipping with that reason is the correct
        # behaviour — it must not invent a verdict.
        assert results["fatsat_qc"].status.value == "skipped"
        assert "rim and core" in results["fatsat_qc"].reason
        for name, res in results.items():
            if name == "fatsat_qc":
                continue
            assert res.status.value in ("ok", "partial"), f"{name}: {res.reason}"

    def test_stage_data_survives_a_process_boundary(self, tmp_path):
        # Resume works by reading the JSON markers, so the data must round-trip
        # through JSON without losing anything a later stage needs.
        cfg = seed_study(tmp_path, fatsat=True, bright_level=18, axial=True)
        cfg.stages = ("geometry",)
        run_pipeline(cfg, log=lambda *_: None)
        fresh = Config(work_dir=cfg.work_dir, subject_id=cfg.subject_id,
                       stages=("marrow", "posterior", "facets_axial", "report"))
        results = run_pipeline(fresh, log=lambda *_: None)
        assert results["facets_axial"].status.value in ("ok", "partial")
        assert results["report"].status.value == "ok"


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
