# CLAUDE.md

Spine MRI analysis pipeline (`spinelab`). Research tool for a single subject's own
study, run on Google Colab. **Not a diagnostic device.**

Load the `spine-pipeline` skill before changing anything under `spinelab/`,
`notebooks/`, or the report — it holds the invariants (evidence levels, canonical
orientation, label-based sides, mask→image resampling, reference-tissue
thresholds) and the reasons behind them.

## Layout

* `spinelab/` — the package. `analysis.py` is pure numpy and fully tested;
  `stages/*.py` are thin I/O wrappers; `pipeline.py` runs stages with resume.
* `notebooks/colab_pipeline.ipynb` — 4-cell launcher, the entry point for users.
* `notebooks/legacy/` — the old 57-cell notebook. Some of its stated conclusions
  are formally retracted in its own text; do not reuse them.
* `docs/` — bug ledger, stack review, clinical scope, privacy.
* `tests/` — 97 tests, no GPU, no patient data, ~1 s.

## Commands

```bash
python -m pytest -q
python -m spinelab run --dicom /path/study.zip --cache ~/spinelab-cache
python -m spinelab audit --dicom /path/dicom
```

Local venv: `.venv/` (Python 3.14, numpy + nibabel + pytest only — the heavy
segmentation stack is installed in Colab, never here).

## Hard constraints

* Patient data (DICOM, NIfTI, results) must never be committed. `.gitignore`
  covers it; the repository history already contains one archive with PHI —
  see `docs/PRIVACY.md`, unresolved and owner-action-only.
* A `HEURISTIC`-level number may not be phrased as a finding anywhere.
* Missing sequence → `SkipStage` with the reason, never a substitute number.
* Do not claim a segmentation stage works without an actual Colab GPU run.

## Environment notes

The operator's internet is slow: prefer changes that avoid re-downloading model
weights (Drive-backed caches, resumable stages, sparse clones). Local pip installs
of large wheels (scipy, torch) time out — keep `analysis.py` numpy-only.
