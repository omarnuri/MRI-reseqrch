"""Run configuration: where data lives, where caches live, which stages run.

Designed around one hard constraint: **the operator has slow internet.** So
* nothing large is ever downloaded to the operator's machine — Colab pulls it;
* every model weight lands in a Google Drive cache directory, so a re-run after a
  disconnect re-uses ~6 GB of weights instead of re-downloading them;
* every stage writes a JSON marker, so a run resumes where it stopped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .utils import set_tool_env

DEFAULT_STAGES = (
    "ingest",         # DICOM -> NIfTI + sequence inventory
    "spineps",        # vertebra instances + semantic subregions (posterior elements)
    "totalspineseg",  # cord, canal, discs
    "totalsegmentator",  # paraspinal muscles, ribs
    "seg_sct",        # Spinal Cord Toolbox cord + canal — the masks the cut-offs assume
    "register",       # masks -> fat-suppressed and axial series (rigid motion correction)
    "fatsat_qc",      # did the fat suppression actually work? gates the oedema branch
    "crosscheck",     # independent vertebra model, per-level reliability (quality profile)
    "facets_axial",   # per-joint, per-side facet measurement on the axial T2
    "geometry",       # wedge angles, kyphosis proxy, lateral deviation
    "muscles",        # paraspinal CSA / volume asymmetry
    "marrow",         # robust intensity outlier screen inside vertebral bodies
    "posterior",      # facet + costovertebral side comparison (label-based)
    "canal",          # canal cross-sectional area profile
    "compression",    # cervical canal against published cut-offs (the only calibrated numbers)
    "normative",      # per-level position in an open whole-spine cohort (ds005616)
    "discs",          # per-level disc signal ranking
    "radiomics",      # first-order + GLCM texture per vertebra
    "agreement",      # cross-tool sanity check (cord Dice)
    "report",         # single HTML report with evidence levels
)


@dataclass
class Config:
    """Everything a run needs. Serialised into results/run_config.json."""

    # --- inputs -----------------------------------------------------------
    dicom_source: str = ""
    """Directory or .zip containing the DICOM study. On Colab prefer a path on
    Drive (e.g. /content/drive/MyDrive/mri/study.zip) — never a public repo."""

    subject_id: str = "anon"

    station: int = 0
    """Which craniocaudal block of the study to analyse; 0 = choose automatically.

    A session can image two levels of the spine (the 2026-08-21 study images two),
    and every sequence pick is one volume per role. Runs are therefore per station,
    and each writes to its own results directory so the second does not overwrite
    the first."""

    # --- workspace --------------------------------------------------------
    work_dir: Path = Path("/content/spine_work")
    cache_dir: Path | None = None
    """Persistent cache for model weights. Set to a Drive path to survive VM
    resets — this is the single biggest win on a slow connection."""

    # --- behaviour --------------------------------------------------------
    stages: tuple[str, ...] = DEFAULT_STAGES
    force: tuple[str, ...] = ()
    """Stages to re-run even if a completed marker exists."""

    device: str = "auto"
    """Where the segmentation models run: "auto", "cuda" or "cpu".

    This used to be a `gpu_required` flag that was assigned and never read, so
    `--no-gpu` changed nothing; the segmentation stages skipped only because the
    CLIs were absent, which looked like the same thing and is not. Now the choice
    reaches the tools: SPINEPS gets `-cpu`, TotalSegmentator gets `device=`, and a
    CPU run is slow but real — which is what makes the pipeline debuggable off a
    GPU host at all.
    """

    timeout_spineps_s: int = 2400
    timeout_tss_s: int = 2400
    timeout_ts_s: int = 2400
    timeout_sct_s: int = 1800
    """Spinal Cord Toolbox. Lower than the others on purpose: its models are small
    and a run that takes half an hour has gone wrong rather than gone slowly."""

    quality: bool = False
    """Spend GPU time on reliability rather than speed: enables the independent
    cross-check model and mirror test-time augmentation. Worth it on an A100,
    pointless on 4 GB. Does not change any measurement definition — only how much
    we know about how reliable the masks are."""

    tta_mirror: bool = False
    """Run the segmenter a second time on a left-right mirrored copy, swap the
    side-specific labels back, and report agreement. This is the direct test of
    whether the model's LEFT/RIGHT assignment is stable — the one property the
    facet question depends on. Requires `quality`."""

    # --- analysis parameters (all explicit, none buried in a cell) --------
    #
    # Every constant below states where it comes from. "OURS" means exactly that:
    # chosen here, with no literature behind it. That distinction is the difference
    # between a measurement and a house rule, and a reader of the report cannot make
    # it unless the code does. See docs/research/reading-list-2026-08.md §4 for the
    # catalogue, including which published scales were derived on the lumbar spine
    # and therefore may not be carried over to the thoracic one.
    marrow_robust_z: float = 3.5
    """Modified z-score threshold (0.6745*(x-median)/MAD) for a bright voxel.

    OURS. A conventional robust-outlier cut-off, not a radiological criterion. It
    ranks levels within one study; it does not define oedema."""
    marrow_min_outlier_voxels: int = 5
    """OURS. Below this a "cluster" is single-voxel noise."""
    posterior_reference_k: float = 3.0
    """Bright-signal threshold for the facet/costal regions, expressed as
    median + k*sigma_MAD of a reference tissue (vertebral marrow).

    OURS. No published scale exists for thoracic facet or costovertebral joints on
    MRI — the ESSR-Arthritis consensus (Eur Radiol 2025) supplies *definitions* for
    reporting them, not a grading scale. So this number sorts regions by brightness
    and can never be phrased as a finding."""
    posterior_bright_percentile: float = 95.0
    """Fallback only, used when no reference tissue is available. Never a
    whole-volume percentile: that highlights CSF and subcutaneous fat. OURS."""
    facet_dilate_voxels: int = 2
    """How far each articular process is grown before intersecting the two to form
    the joint-interface ROI. 2 voxels on a 0.49 mm axial grid is about 1 mm. OURS."""
    muscle_asymmetry_pct_threshold: float = 10.0
    """OURS. A reporting threshold for left-right paraspinal difference, not a
    clinical one: asymmetry of this size is common in asymptomatic people, more so
    in a former athlete."""
    wedge_scheuermann_deg: float = 5.0
    wedge_scheuermann_run: int = 3
    """Scheuermann's rule as classically stated — anterior wedging of at least 5° at
    three or more adjacent thoracic levels. The rule is standard; the primary source
    has not been checked against in this repository, so it is cited as convention
    rather than as a verified reference."""
    canal_stenosis_pct: float = 33.0
    """Relative narrowing against the study's own median canal area.

    OURS, and deliberately relative: clinical stenosis is defined on absolute area
    together with cord signal and symptoms, and no absolute threshold can be applied
    to one study with no normative cohort. The one calibrated exception in this
    project is the cervical canal, where `stages/compression.py` uses published
    cut-offs from external cohorts."""
    min_series_slices: int = 5
    """Series with fewer slices are localisers/scouts and never analysis inputs."""

    notes: dict = field(default_factory=dict)

    # --- derived paths ----------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.work_dir / "data"

    @property
    def dicom_dir(self) -> Path:
        return self.data_dir / "dicom"

    @property
    def nifti_dir(self) -> Path:
        return self.data_dir / "nifti"

    @property
    def results_dir(self) -> Path:
        if self.station:
            return self.work_dir / "results" / f"station-{self.station}"
        return self.work_dir / "results"

    @property
    def stage_dir(self) -> Path:
        return self.results_dir / "stages"

    @property
    def intermediate_dir(self) -> Path:
        return self.results_dir / "intermediate"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def weights_dir(self) -> Path:
        """Where model weights live. Drive-backed when cache_dir is set."""
        base = self.cache_dir or self.work_dir
        return Path(base) / "weights"

    def resolve_device(self) -> str:
        """"auto" -> what is actually available. Returns "cuda" or "cpu".

        Timeouts are not adjusted here: a CPU run of the same model is roughly an
        order of magnitude slower, and silently stretching the limit would hide a
        genuinely hung process. `spinelab run --device cpu` is expected to be given
        a longer `--timeout` deliberately.
        """
        if self.device in ("cuda", "gpu"):
            return "cuda"
        if self.device == "cpu":
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 — no torch at all means no GPU path
            return "cpu"

    def ensure_dirs(self) -> None:
        for d in (
            self.work_dir, self.data_dir, self.dicom_dir, self.nifti_dir,
            self.results_dir, self.stage_dir, self.intermediate_dir,
            self.figures_dir, self.weights_dir,
        ):
            Path(d).mkdir(parents=True, exist_ok=True)

    def export_env(self) -> dict[str, str]:
        """Point every upstream tool at the persistent weights cache.

        These env vars are the documented cache locations of the tools we call:
        SPINEPS honours SPINEPS_SEGMENTOR_MODELS, TotalSegmentator honours
        TOTALSEG_HOME_DIR, nnU-Net honours nnUNet_results, and totalspineseg
        keeps its data under TOTALSPINESEG_DATA.
        """
        w = self.weights_dir
        env = {
            "SPINEPS_SEGMENTOR_MODELS": str(w / "spineps"),
            "TOTALSEG_HOME_DIR": str(w / "totalsegmentator"),
            "TOTALSPINESEG_DATA": str(w / "totalspineseg"),
            "nnUNet_results": str(w / "totalsegmentator" / "nnunet" / "results"),
            "nnUNet_raw": str(w / "nnunet_raw"),
            "nnUNet_preprocessed": str(w / "nnunet_preprocessed"),
            # HuggingFace / torch caches too: BiomedCLIP-class downloads are the
            # other multi-GB item that must not be fetched twice.
            "HF_HOME": str(w / "huggingface"),
            "TORCH_HOME": str(w / "torch"),
        }
        for path in env.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        # Also recorded outside os.environ: `import totalspineseg` sets
        # nnUNet_results to a *relative* './nnUNet_results' at module level, so
        # os.environ stops being a reliable record of this decision the moment
        # anything probes the stack. See utils.child_env.
        set_tool_env(env)
        return env

    def to_dict(self) -> dict:
        d = asdict(self)
        for key, value in list(d.items()):
            if isinstance(value, Path):
                d[key] = str(value)
        d["stages"] = list(self.stages)
        d["force"] = list(self.force)
        return d


def from_dict(payload: dict) -> Config:
    """Build a Config from a plain dict (Colab form values, JSON, CLI)."""
    known = {f for f in Config.__dataclass_fields__}
    kwargs = {}
    for key, value in payload.items():
        if key not in known:
            continue
        if key in ("work_dir", "cache_dir") and value is not None:
            value = Path(value)
        if key in ("stages", "force") and value is not None:
            value = tuple(value)
        kwargs[key] = value
    return Config(**kwargs)
