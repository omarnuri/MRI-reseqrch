"""A reference distribution for spine geometry, built from an open cohort.

Every other measurement in this project compares the subject to himself, because
there was nothing to compare him to. The search for that something is written up
in `docs/research/` and it ended in one usable result: **OpenNeuro ds005616**,
sixty-odd volunteers scanned head-to-torso on 3 T Siemens with a 3D T2w SPACE, and
released under CC0 with *manual* whole-spine segmentations and manual disc-level
labels. Only the masks are needed here — about 17 MB for the whole cohort, no
images at all — so this is cheap enough to build once and keep in the Drive cache.

Two rules shape the whole module.

**The same code measures both sides.** `measure_label_volume` is called on a
cohort subject and on this study's own TotalSpineSeg output, and neither gets a
special case. A reference distribution built by different code from the one that
measures the patient is not a reference, it is a coincidence.

**Nothing here is a threshold.** The output is a distribution, and the stage that
consumes it reports a position inside that distribution. This project reserves
"which side of the line" for `Evidence.CALIBRATED`, which means a cut-off
published by the people who assembled the cohort. Nobody published a cut-off on
ds005616; we computed the spread ourselves, and that is a weaker thing which has
to keep saying so.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import labels as L
from .analysis import (
    MIN_REFERENCE_N,
    centroids_si,
    label_table,
    levels_from_disc_anchors,
    mean_thickness_mm,
    sct_level_index,
    sct_level_name,
    segmental_angles_deg,
    slab_area_mm2,
    slice_counts,
    summarise_distribution,
)
from .runlog import get_logger
from .utils import AP_AXIS, LR_AXIS, SI_AXIS, load_canonical, write_json

log = get_logger(__name__)

DATASET = "ds005616"
CONTRAST = "T2w"
S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET}"

#: The four mask files used per subject. Nothing else is downloaded — in
#: particular no image, which is what keeps the whole cohort under 20 MB.
MASK_FILES = {
    "spine": f"{CONTRAST}_label-spine_dseg",
    "discs": f"{CONTRAST}_labels-disc-manual",
    "canal": f"{CONTRAST}_label-canal_seg",
    "cord": f"{CONTRAST}_label-SC_seg",
}

#: What travels with every number this cohort produces. Verified against the
#: dataset's own README and dataset_description.json, not quoted from a paper.
COHORT = {
    "dataset": DATASET,
    "name": "whole-spine (OpenNeuro)",
    "doi": "doi:10.18112/openneuro.ds005616.v1.1.4",
    "license": "CC0",
    "field_strength_t": 3.0,
    "scanners": "Siemens MAGNETOM Tim Trio and Verio",
    "sequence": "3D T2w SPACE, whole spine, 1 mm isotropic",
    "segmentation": "manual whole-spine labels and manual disc-level labels "
                    "(derivatives/labels, distributed with the dataset)",
    "age_years": "21-56 (mean 27.1, SD 6.5)",
    "sex": "32 M, 18 F, 10 undisclosed",
    "population": "volunteers; the dataset records incidental disc levels per "
                  "subject in participants.tsv rather than claiming a screened "
                  "normal cohort",
    "url": f"https://openneuro.org/datasets/{DATASET}",
}

#: Measurements defined identically on any whole-vertebra label volume. Heights
#: and wedge angles are deliberately absent: they need the vertebral body alone,
#: and a whole-spine label volume does not separate it from the posterior
#: elements. `stages/geometry.py` measures those, against this subject himself.
METRICS = ("canal_area_mm2", "cord_area_mm2", "cord_canal_ratio",
           "disc_height_mm", "segmental_angle_deg")

_TIMEOUT_S = 120


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def cohort_dir(cache_dir) -> Path:
    return Path(cache_dir) / "normative" / DATASET


def cohort_json(cache_dir) -> Path:
    return Path(cache_dir) / "normative" / f"{DATASET}-{CONTRAST}.json"


def _download(url: str, dest: Path) -> bool:
    """One file, atomically. A partial file left behind is worse than no file."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
            part.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False                      # subject simply has no such file
        log.warning("%s: HTTP %s", url, exc.code)
        return False
    except Exception as exc:                  # noqa: BLE001 — network, reported not raised
        log.warning("%s: %s", url, exc)
        part.unlink(missing_ok=True)
        return False
    part.replace(dest)
    return True


def subject_ids(cache_dir) -> list[str]:
    """Subject list from the cohort's own participants.tsv."""
    path = cohort_dir(cache_dir) / "participants.tsv"
    if not _download(f"{S3_BASE}/participants.tsv", path):
        return []
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        first = line.split("\t")[0].strip()
        if first.startswith("sub-"):
            ids.append(first)
    return ids


def fetch(cache_dir, subjects: list[str] | None = None) -> dict[str, dict[str, Path]]:
    """Download the mask derivatives. Idempotent — an interrupted build resumes."""
    subjects = subjects or subject_ids(cache_dir)
    out: dict[str, dict[str, Path]] = {}
    for subject in subjects:
        paths: dict[str, Path] = {}
        for role, suffix in MASK_FILES.items():
            name = f"{subject}_{suffix}.nii.gz"
            dest = cohort_dir(cache_dir) / subject / name
            url = f"{S3_BASE}/derivatives/labels/{subject}/anat/{name}"
            if _download(url, dest):
                paths[role] = dest
        # The spine labels and the disc labels are the two that carry the level
        # numbering; without both, this subject cannot be placed on the spine at
        # all and contributes nothing.
        if "spine" in paths and "discs" in paths:
            out[subject] = paths
        else:
            log.info("%s: no %s whole-spine labels, skipped", subject, CONTRAST)
    return out


# --------------------------------------------------------------------------
# Measuring
# --------------------------------------------------------------------------


def anchors_from_disc_points(volume: np.ndarray) -> dict[int, float]:
    """The manual disc file as {level index: superior-inferior position}.

    Its label values already are this project's level numbering — a disc named by
    the vertebra below it, C1 = 1 — which is what makes the whole cohort placeable
    on the spine without reading a single label value out of the segmentation.
    """
    return {int(k): v for k, v in centroids_si(volume).items() if 1 <= int(k) <= 25}


def measure_label_volume(vertebrae: np.ndarray, index_map: dict[int, int],
                         zooms, discs: np.ndarray | None = None,
                         disc_index_map: dict[int, int] | None = None,
                         canal: np.ndarray | None = None,
                         cord: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Per-level morphometry, keyed by anatomical name ('T7').

    `index_map` maps the label values of `vertebrae` onto level indices counting
    C1 = 1, so two volumes in unrelated label spaces are compared level for level.
    Everything missing is simply absent from the result — a level not covered by
    the field of view has no entry, never a zero.

    `disc_height_mm` at a level is the disc *above* it, because that is what "a
    disc is named by the vertebra below it" means and it is the convention the SCT
    interop already uses. So `T7 -> disc_height_mm` is the T6/T7 disc, on both
    sides of the comparison.
    """
    lr_mm, ap_mm, si_mm = (float(zooms[LR_AXIS]), float(zooms[AP_AXIS]), float(zooms[SI_AXIS]))
    voxel_area = lr_mm * ap_mm
    ap_size = int(vertebrae.shape[AP_AXIS])
    out: dict[str, dict[str, float]] = {}

    table = label_table(vertebrae)
    disc_table = table if discs is vertebrae else (
        label_table(discs) if discs is not None else {})
    canal_counts = slice_counts(canal) if canal is not None else None
    cord_counts = slice_counts(cord) if cord is not None else None

    centroids = {label: table[label].mean(axis=0)
                 for label in index_map if label in table}
    angles = segmental_angles_deg(centroids)
    disc_by_level = {index: label for label, index in (disc_index_map or {}).items()}

    for label, index in sorted(index_map.items(), key=lambda kv: kv[1]):
        name = sct_level_name(index)
        coords = table.get(label)
        if name is None or coords is None or coords.size == 0:
            continue
        z = coords[:, SI_AXIS]
        z_low, z_high = int(z.min()), int(z.max())
        entry: dict[str, float] = {}
        canal_area = (slab_area_mm2(canal_counts, z_low, z_high, voxel_area)
                      if canal_counts is not None else None)
        cord_area = (slab_area_mm2(cord_counts, z_low, z_high, voxel_area)
                     if cord_counts is not None else None)
        if canal_area is not None:
            entry["canal_area_mm2"] = canal_area
        if cord_area is not None:
            entry["cord_area_mm2"] = cord_area
        if canal_area and cord_area:
            entry["cord_canal_ratio"] = round(cord_area / canal_area, 4)
        if label in angles:
            entry["segmental_angle_deg"] = angles[label]
        disc_coords = disc_table.get(disc_by_level.get(index))
        if disc_coords is not None:
            thickness = mean_thickness_mm(disc_coords, si_mm, ap_size)
            if thickness is not None:
                entry["disc_height_mm"] = round(thickness, 3)
        if entry:
            out[name] = entry
    return out


def measure_subject(paths: dict[str, Path]) -> tuple[dict | None, str]:
    """One cohort subject, from its four mask files."""
    spine_img = load_canonical(paths["spine"])
    spine = np.asarray(spine_img.get_fdata()).astype(np.int32)
    disc_manual = np.asarray(
        load_canonical(paths["discs"]).get_fdata()).astype(np.int32)

    vertebra_map, disc_map, note = levels_from_disc_anchors(
        centroids_si(spine), anchors_from_disc_points(disc_manual))
    if not vertebra_map:
        return None, note

    def _binary(role: str):
        if role not in paths:
            return None
        return np.asarray(load_canonical(paths[role]).get_fdata()) > 0.5

    levels = measure_label_volume(
        spine, vertebra_map, spine_img.header.get_zooms(),
        discs=spine, disc_index_map=disc_map,
        canal=_binary("canal"), cord=_binary("cord"))
    if not levels:
        return None, "no level could be measured"
    return levels, note


# --------------------------------------------------------------------------
# Building the reference
# --------------------------------------------------------------------------


def build(cache_dir, subjects: list[str] | None = None,
          min_n: int = MIN_REFERENCE_N) -> dict:
    """Download, measure, summarise, and write the reference JSON."""
    started = time.time()
    available = fetch(cache_dir, subjects)
    if not available:
        raise RuntimeError(
            f"no {DATASET} masks could be downloaded — check the connection, then "
            f"re-run; the download is resumable and lives in {cohort_dir(cache_dir)}")

    per_level: dict[str, dict[str, list[float]]] = {}
    used, skipped = [], []
    for subject, paths in sorted(available.items()):
        try:
            levels, note = measure_subject(paths)
        except Exception as exc:                       # noqa: BLE001
            levels, note = None, f"{type(exc).__name__}: {exc}"
        if levels is None:
            skipped.append({"subject": subject, "reason": note})
            log.info("%s skipped: %s", subject, note)
            continue
        used.append(subject)
        for name, entry in levels.items():
            for metric, value in entry.items():
                per_level.setdefault(name, {}).setdefault(metric, []).append(float(value))
        log.info("%s measured: %d levels (%s)", subject, len(levels), note)

    levels_out: dict[str, dict] = {}
    for name, metrics in per_level.items():
        summaries = {}
        for metric in METRICS:
            summary = summarise_distribution(metrics.get(metric, []), min_n=min_n)
            if summary is not None:
                summary["values"] = [round(v, 3) for v in sorted(metrics[metric])]
                summaries[metric] = summary
        if summaries:
            levels_out[name] = summaries

    payload = {
        "cohort": {**COHORT,
                   "subjects_measured": len(used),
                   "subjects_skipped": len(skipped),
                   "min_subjects_per_level": min_n,
                   "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "build_seconds": round(time.time() - started, 1)},
        "metrics": list(METRICS),
        "levels": levels_out,
        "skipped": skipped,
        "what_this_is_not": [
            "Not a published normative reference: the spread here was computed by "
            "this project from the cohort's masks, and no cut-off derived from it "
            "has ever been validated against an outcome.",
            "Not a screened healthy sample: participants.tsv records incidental "
            "disc levels in many of these volunteers.",
            f"Acquired at 3 T, 1 mm isotropic 3D T2w. Any study compared against it "
            f"at a different field strength or slice thickness carries that "
            f"difference into every number.",
        ],
    }
    write_json(cohort_json(cache_dir), payload)
    log.info("normative reference: %d subjects, %d levels -> %s",
             len(used), len(levels_out), cohort_json(cache_dir))
    return payload


def load(cache_dir) -> dict | None:
    """The cached reference, or None if it has not been built."""
    import json

    path = cohort_json(cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                            # noqa: BLE001
        log.warning("cannot read %s: %s", path, exc)
        return None


def tss_index_map(volume: np.ndarray) -> dict[int, int]:
    """Level index for each TotalSpineSeg vertebra label present in a volume.

    TotalSpineSeg's numbering is not a constant offset (C1=11, T1=21, L1=41), so
    it comes from `labels.TSS_VERTEBRAE` by name rather than from the disc-anchored
    derivation used on the reference cohort.
    """
    out: dict[int, int] = {}
    for label in np.unique(volume):
        index = sct_level_index(L.TSS_VERTEBRAE.get(int(label)))
        if index is not None:
            out[int(label)] = index
    return out


def tss_disc_index_map(volume: np.ndarray) -> dict[int, int]:
    """Same, for TotalSpineSeg disc labels, named by the vertebra below."""
    from .analysis import sct_disc_value

    out: dict[int, int] = {}
    for label in np.unique(volume):
        value = sct_disc_value(L.TSS_DISCS.get(int(label)))
        if value is not None:
            out[int(label)] = value
    return out
