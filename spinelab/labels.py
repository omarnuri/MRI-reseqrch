"""Label spaces of the upstream segmentation tools.

Every magic number in the old notebook lived inline next to the code that used
it, which is how `sub >= 40` ended up sweeping the vertebral body, the cord and
the CSF-filled canal into a "posterior elements" mask. The values below are
transcribed from the upstream sources so they can be checked against them:

* SPINEPS semantic labels == ``TPTBox.core.vert_constants.Location``
  (https://github.com/Hendrik-code/TPTBox, ``Location`` enum)
* SPINEPS instance labels == vertebra index 1..25 (C1=1 … L5=24, sacrum=25)
* TotalSpineSeg labels == ``totalspineseg/resources/labels_maps/tss_map.json``

Verified against upstream main on 2026-07-26 (SPINEPS 2.0.0,
totalspineseg 20260623).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# SPINEPS / TPTBox semantic (subregion) labels
# --------------------------------------------------------------------------
VERTEBRA_FULL = 40
ARCUS_VERTEBRAE = 41
SPINOSUS_PROCESS = 42
COSTAL_PROCESS_LEFT = 43
COSTAL_PROCESS_RIGHT = 44
SUPERIOR_ARTICULAR_LEFT = 45
SUPERIOR_ARTICULAR_RIGHT = 46
INFERIOR_ARTICULAR_LEFT = 47
INFERIOR_ARTICULAR_RIGHT = 48
# TPTBox comments 49 as "actual corpus body" and 50 as Vertebra_Corpus. Which of
# the two a given SPINEPS model writes has changed between releases, so never
# hard-code one: use resolve_corpus_label() against the actual volume.
VERTEBRA_CORPUS_BORDER = 49
VERTEBRA_CORPUS = 50
CORPUS_CANDIDATES = (VERTEBRA_CORPUS_BORDER, VERTEBRA_CORPUS)
ENDPLATE_SUPERIOR = 52
ENDPLATE_INFERIOR = 53
SPINAL_CORD = 60
SPINAL_CANAL = 61
ENDPLATE = 62
RIB_LEFT = 63
RIB_RIGHT = 64
VERTEBRA_DISC = 100

#: Facet (zygapophyseal) joint articular processes, split by side. These are the
#: structures that matter for a facet-arthropathy question.
FACET_LEFT = (SUPERIOR_ARTICULAR_LEFT, INFERIOR_ARTICULAR_LEFT)
FACET_RIGHT = (SUPERIOR_ARTICULAR_RIGHT, INFERIOR_ARTICULAR_RIGHT)

#: Costal / transverse processes, split by side — the vertebral half of the
#: costovertebral and costotransverse joints.
COSTAL_LEFT = (COSTAL_PROCESS_LEFT,)
COSTAL_RIGHT = (COSTAL_PROCESS_RIGHT,)

#: Midline posterior elements. Present on both sides by definition, so they are
#: excluded from any left/right comparison.
POSTERIOR_MIDLINE = (ARCUS_VERTEBRAE, SPINOSUS_PROCESS)

#: Everything that is genuinely a posterior element. Deliberately does NOT
#: include 49/50 (body), 60 (cord) or 61 (canal/CSF).
POSTERIOR_ELEMENTS = (
    POSTERIOR_MIDLINE + FACET_LEFT + FACET_RIGHT + COSTAL_LEFT + COSTAL_RIGHT
)

SEMANTIC_LABEL_NAMES = {
    VERTEBRA_FULL: "vertebra_full",
    ARCUS_VERTEBRAE: "arcus_vertebrae",
    SPINOSUS_PROCESS: "spinous_process",
    COSTAL_PROCESS_LEFT: "costal_process_left",
    COSTAL_PROCESS_RIGHT: "costal_process_right",
    SUPERIOR_ARTICULAR_LEFT: "superior_articular_left",
    SUPERIOR_ARTICULAR_RIGHT: "superior_articular_right",
    INFERIOR_ARTICULAR_LEFT: "inferior_articular_left",
    INFERIOR_ARTICULAR_RIGHT: "inferior_articular_right",
    VERTEBRA_CORPUS_BORDER: "vertebra_corpus_border",
    VERTEBRA_CORPUS: "vertebra_corpus",
    ENDPLATE_SUPERIOR: "endplate_superior",
    ENDPLATE_INFERIOR: "endplate_inferior",
    SPINAL_CORD: "spinal_cord",
    SPINAL_CANAL: "spinal_canal",
    RIB_LEFT: "rib_left",
    RIB_RIGHT: "rib_right",
    VERTEBRA_DISC: "intervertebral_disc",
}

# --------------------------------------------------------------------------
# SPINEPS instance labels: vertebra index
# --------------------------------------------------------------------------
#: instance label -> anatomical name (C1=1 … C7=7, T1=8 … T12=19, L1=20 … L5=24)
VERTEBRA_NAMES = {
    **{i: f"C{i}" for i in range(1, 8)},
    **{i: f"T{i - 7}" for i in range(8, 20)},
    **{i: f"L{i - 19}" for i in range(20, 25)},
    25: "sacrum",
}
NAME_TO_VERTEBRA = {v: k for k, v in VERTEBRA_NAMES.items()}

#: Instance labels that are actual (non-sacral) vertebrae.
VERTEBRA_LABEL_MIN = 1
VERTEBRA_LABEL_MAX = 24
THORACIC_LABELS = tuple(range(8, 20))
LUMBAR_LABELS = tuple(range(20, 25))


def vertebra_name(label: int) -> str:
    """Anatomical name for a SPINEPS instance label ('T11'), never a raw id."""
    return VERTEBRA_NAMES.get(int(label), f"id_{int(label)}")


def vertebra_label(name: str) -> int | None:
    """'T11' -> 18. Returns None for an unknown name (never guesses)."""
    return NAME_TO_VERTEBRA.get(name.strip().upper())


def is_vertebra(label: int) -> bool:
    return VERTEBRA_LABEL_MIN <= int(label) <= VERTEBRA_LABEL_MAX


# --------------------------------------------------------------------------
# TotalSpineSeg label space (tss_map.json)
# --------------------------------------------------------------------------
TSS_SPINAL_CORD = 1
TSS_CSF = 2
#: What "the canal" means in this label space: the cord plus the CSF around it.
#: TotalSpineSeg has no single canal label in tss_map.json — the `step1_canal` file
#: looks like one and is a soft probability map (8999 distinct values on the real
#: study), which is what an earlier version of the canal stage thresholded.
TSS_CANAL_LABELS = (TSS_SPINAL_CORD, TSS_CSF)
TSS_SACRUM = 50
#: In tss_map.json vertebrae occupy 11..47 and the sacrum is 50; every label
#: >= 63 is an intervertebral disc (C2-C3=63 … L5-S=100).
TSS_DISC_LABEL_MIN = 63
TSS_VERTEBRAE = {
    **{10 + i: f"C{i}" for i in range(1, 8)},
    **{20 + i: f"T{i}" for i in range(1, 13)},
    **{40 + i: f"L{i}" for i in range(1, 8)},
    TSS_SACRUM: "sacrum",
}
TSS_DISCS = {
    63: "C2-C3", 64: "C3-C4", 65: "C4-C5", 66: "C5-C6", 67: "C6-C7",
    71: "C7-T1", 72: "T1-T2", 73: "T2-T3", 74: "T3-T4", 75: "T4-T5",
    76: "T5-T6", 77: "T6-T7", 78: "T7-T8", 79: "T8-T9", 80: "T9-T10",
    81: "T10-T11", 82: "T11-T12", 91: "T12-L1", 92: "L1-L2", 93: "L2-L3",
    94: "L3-L4", 95: "L4-L5", 100: "L5-S",
}


def tss_disc_name(label: int) -> str:
    """Human-readable disc level for a TotalSpineSeg label."""
    return TSS_DISCS.get(int(label), f"disc_label_{int(label)}")


def resolve_corpus_label(semantic_data, min_voxels: int = 200) -> int | None:
    """Pick whichever corpus label this SPINEPS build actually wrote.

    TPTBox defines both 49 (Vertebra_Corpus_border, commented "actual corpus
    body") and 50 (Vertebra_Corpus). The old notebook hard-coded 49; if a model
    writes 50 instead, the body mask silently becomes empty and every wedge
    angle falls back to the full vertebra mask — which is exactly the bug that
    produced posterior heights of 2-9 mm and negative wedge angles.

    Returns the candidate with the most voxels, or None if neither is present.
    """
    best, best_count = None, 0
    for label in CORPUS_CANDIDATES:
        count = int((semantic_data == label).sum())
        if count >= min_voxels and count > best_count:
            best, best_count = label, count
    return best
