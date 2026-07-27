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

DEFAULT_STAGES = (
    "ingest",         # DICOM -> NIfTI + sequence inventory
    "spineps",        # vertebra instances + semantic subregions (posterior elements)
    "totalspineseg",  # cord, canal, discs
    "totalsegmentator",  # paraspinal muscles, ribs
    "register",       # masks -> fat-suppressed and axial series (rigid motion correction)
    "fatsat_qc",      # did the fat suppression actually work? gates the oedema branch
    "crosscheck",     # independent vertebra model, per-level reliability (quality profile)
    "facets_axial",   # per-joint, per-side facet measurement on the axial T2
    "geometry",       # wedge angles, kyphosis proxy, lateral deviation
    "muscles",        # paraspinal CSA / volume asymmetry
    "marrow",         # robust intensity outlier screen inside vertebral bodies
    "posterior",      # facet + costovertebral side comparison (label-based)
    "canal",          # canal cross-sectional area profile
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
    marrow_robust_z: float = 3.5
    """Modified z-score threshold (0.6745*(x-median)/MAD) for a bright voxel."""
    marrow_min_outlier_voxels: int = 5
    posterior_reference_k: float = 3.0
    """Bright-signal threshold for the facet/costal regions, expressed as
    median + k*sigma_MAD of a reference tissue (vertebral marrow)."""
    posterior_bright_percentile: float = 95.0
    """Fallback only, used when no reference tissue is available. Never a
    whole-volume percentile: that highlights CSF and subcutaneous fat."""
    facet_dilate_voxels: int = 2
    """How far each articular process is grown before intersecting the two to form
    the joint-interface ROI. 2 voxels on a 0.49 mm axial grid is about 1 mm."""
    muscle_asymmetry_pct_threshold: float = 10.0
    wedge_scheuermann_deg: float = 5.0
    wedge_scheuermann_run: int = 3
    canal_stenosis_pct: float = 33.0
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
