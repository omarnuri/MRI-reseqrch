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
