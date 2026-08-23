"""How much thoracic-spine MRI is actually inside ROCOv2?

The claim to test: figures harvested from open-access papers could stand in for a
thoracic dataset. That is measurable — count the captions, then check how many of
those images are MRI rather than radiograph or CT.
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent / "rocov2"

SPINE = re.compile(r"\b(spine|spinal|vertebra\w*|intervertebral|disc\s+herniat\w*)\b", re.I)
THORACIC = re.compile(r"\b(thoracic|dorsal\s+spine|T\d{1,2}[-–/]T?\d{0,2}\s*(vertebra|level|disc)?)\b", re.I)
THORACIC_STRICT = re.compile(r"\b(thoracic\s+(spine|vertebra\w*|disc|cord)|dorsal\s+spine)\b", re.I)
MRI = re.compile(r"\b(MRI|MR\s+imag\w*|magnetic\s+resonance|T1[-\s]?weighted|T2[-\s]?weighted|STIR)\b", re.I)
FACET = re.compile(r"\b(facet|zygapophys\w*|costovertebral|costotransverse)\b", re.I)


def load_captions():
    rows = []
    for split in ("train", "valid", "test"):
        path = ROOT / f"{split}_captions.csv"
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                row["_split"] = split
                rows.append(row)
    return rows


rows = load_captions()
field = "caption" if "caption" in rows[0] else list(rows[0])[1]
print(f"columns: {list(rows[0])}")
print(f"captions total: {len(rows)}")

counts = Counter()
thoracic_mri_examples = []
for row in rows:
    text = row.get(field) or ""
    is_spine, is_mri = bool(SPINE.search(text)), bool(MRI.search(text))
    is_thoracic = bool(THORACIC_STRICT.search(text))
    counts["spine"] += is_spine
    counts["mri"] += is_mri
    counts["spine+mri"] += is_spine and is_mri
    counts["thoracic"] += is_thoracic
    counts["thoracic+mri"] += is_thoracic and is_mri
    counts["thoracic+spine+mri"] += is_thoracic and is_spine and is_mri
    counts["facet/costovertebral"] += bool(FACET.search(text))
    counts["facet+thoracic"] += bool(FACET.search(text)) and is_thoracic
    if is_thoracic and is_spine and is_mri and len(thoracic_mri_examples) < 8:
        thoracic_mri_examples.append((row.get("ID") or row.get("id") or "?", text[:220]))

for key, value in counts.most_common():
    print(f"{key:24s} {value:6d}   ({100.0 * value / len(rows):.2f}%)")

print("\nexamples of thoracic spine MRI captions:")
for image_id, text in thoracic_mri_examples:
    print(f"  [{image_id}] {text}")
