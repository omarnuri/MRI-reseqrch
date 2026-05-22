"""MRI sequence + orientation classification and selection (notebook Cell 2c)."""


def classify(meta):
    """Classify a NIfTI's sequence (T1/T2/T2_FS/STIR) and orientation.

    `meta` is the dcm2niix JSON sidecar dict (may be empty).
    """
    desc = (str(meta.get("SeriesDescription", "")) + " "
            + str(meta.get("ProtocolName", ""))).lower()
    te, tr = meta.get("EchoTime"), meta.get("RepetitionTime")
    if "stir" in desc:
        seq = "STIR"
    elif "t2" in desc and ("fs" in desc or "fat" in desc):
        seq = "T2_FS"
    elif "t2" in desc:
        seq = "T2"
    elif "t1" in desc:
        seq = "T1"
    elif te and tr and te > 60 and tr > 2000:
        seq = "T2"
    elif te and tr and te < 30 and tr < 1000:
        seq = "T1"
    else:
        seq = "unknown"

    if "sag" in desc:
        orient = "sagittal"
    elif "ax" in desc or "tra" in desc:
        orient = "axial"
    elif "cor" in desc:
        orient = "coronal"
    else:
        orient = "unknown"
    return seq, orient


def pick_sequence(sequence_info, seq, orient):
    """First entry in `sequence_info` matching sequence + orientation, else None."""
    cands = [s for s in sequence_info
             if s.get("sequence") == seq and s.get("orientation") == orient]
    return cands[0]["nifti"] if cands else None


def pick_any_orientation(sequence_info, seq, prefer=("sagittal", "coronal", "axial")):
    """Pick a sequence of type `seq` in any preferred orientation.

    Returns (nifti_path, orientation) or (None, None). Used to recover
    STIR (or T2_FS as fallback) when the gold-standard sagittal STIR is
    missing but coronal/axial STIR is present in the study — without this
    the edema-detection path stays inactive on otherwise-usable data.
    """
    for orient in prefer:
        for s in sequence_info:
            if s.get("sequence") == seq and s.get("orientation") == orient:
                return s.get("nifti"), orient
    return None, None


def pick_stir_or_fatsat(sequence_info):
    """Pick the best STIR/fat-suppressed sequence available, any orientation.

    Returns (nifti_path, orientation, source) where `source` is "STIR" or
    "T2_FS" so downstream code can label the anomaly method honestly.
    Preference order: STIR (sag>cor>ax) > T2_FS (sag>cor>ax).
    """
    nii, orient = pick_any_orientation(sequence_info, "STIR")
    if nii:
        return nii, orient, "STIR"
    nii, orient = pick_any_orientation(sequence_info, "T2_FS")
    if nii:
        return nii, orient, "T2_FS"
    return None, None, None

