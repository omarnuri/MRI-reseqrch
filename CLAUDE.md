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
* `notebooks/colab_pipeline.ipynb` — the launcher, the entry point for users.
  `tests/test_notebook.py` guards it: cells must parse, carry no committed outputs
  (one used to embed the patient's name), and read results from the station the run
  actually wrote to.
* `notebooks/legacy/` — the old 57-cell notebook. Some of its stated conclusions
  are formally retracted in its own text; do not reuse them.
* `docs/` — bug ledger, stack review, clinical scope, privacy.
* `tests/` — 404 tests, no GPU, no patient data, ~70 s.

## Commands

```bash
python -m pytest -q
python -m spinelab run --dicom /path/study.zip --cache ~/spinelab-cache
python -m spinelab run --dicom /path/study.zip --station 2   # one station per run
python -m spinelab audit --dicom /path/dicom
python -m spinelab setup --sct --cache ~/spinelab-cache      # Spinal Cord Toolbox, ~3 GB
python -m spinelab normative --cache ~/spinelab-cache        # reference cohort, ~21 MB of masks
```

Two local venvs, on purpose:

* `.venv/` — Python 3.14, numpy + nibabel + pydicom + pytest. Runs the test suite
  in ~10 s. This is the one to use for everything that does not need a model.
* `.venv312/` — Python 3.12 (same minor as Colab), the full segmentation stack on
  CPU torch. Exists so the GPU stages can be debugged here instead of by guessing
  at a Colab traceback. `uv` manages the interpreter: `uv python install 3.12`.

```bash
.venv312/Scripts/python -m spinelab run --dicom … --device cpu --timeout 14400
```

CPU inference is roughly ten times slower than an A100 and produces the same masks,
so it finds code bugs but is not how a real run should be made.

## Hard constraints

* Patient data (DICOM, NIfTI, results) must never be committed. `.gitignore`
  covers it; the repository history already contains one archive with PHI —
  see `docs/PRIVACY.md`, unresolved and owner-action-only.
* A `HEURISTIC`-level number may not be phrased as a finding anywhere.
* `CALIBRATED` is the only level allowed to say which side of a threshold a value falls
  on, and only when the cohort the threshold came from travels with the number. Today
  that is the Spinal Cord Toolbox cervical measures and nothing else.
* A percentile against the `normative` cohort is a position, not a threshold: that
  stage is `MEASUREMENT`, and the words "normal" and "abnormal" may not appear beside
  one. The cohort spread is computed here, not published, which is the whole
  difference from `CALIBRATED`.
* Missing sequence → `SkipStage` with the reason, never a substitute number.
* Do not claim a segmentation stage works without an actual Colab GPU run.

## Environment notes

The operator's internet is slow: prefer changes that avoid re-downloading model
weights (Drive-backed caches, resumable stages, sparse clones). Keep `analysis.py`
numpy-only regardless — it is the part that must stay testable in one second.

Two upstream facts that cost several sessions, both encoded in `envsetup.py`:

* SPINEPS 2.0.0 declares `acvl-utils==0.2`; nnU-Net has needed `>=0.2.6` since
  2.7.0. No resolver can satisfy both, so SPINEPS is installed with `--no-deps` and
  its dependencies are listed by hand. Do not put SPINEPS back into
  `SEGMENTATION_PACKAGES`.
* Google Drive is a cache, never a requirement. A failed mount must degrade to a
  local cache, and nothing may be created under an unmounted `/content/drive` —
  that stub is what makes the next mount fail too.
