---
name: spine-pipeline
description: Work on the spinelab spine-MRI pipeline in this repository — add or change a stage, debug a Colab run, interpret results, or touch anything that produces a number about a patient's spine. Use when the task involves spinelab/, notebooks/colab_pipeline.ipynb, DICOM/NIfTI handling, SPINEPS / TotalSpineSeg / TotalSegmentator, vertebra or facet-joint measurements, or the HTML report.
---

# spinelab

Inference-only spine MRI pipeline. Research tool, **not** a diagnostic device.
The whole design exists to stop one specific failure: numbers that are real but
support no conclusion being presented as findings.

## The rule that overrides convenience

Every value that reaches a human carries an evidence level
(`spinelab/evidence.py`):

| level | means | may a conclusion be phrased from it? |
|---|---|---|
| `MODEL` | output of a pretrained, published segmentation model with real weights | yes, as an observation |
| `MEASUREMENT` | deterministic geometry on such a mask | yes, as a number; clinical meaning is the radiologist's |
| `HEURISTIC` | intensity thresholds, percentiles, robust z-scores | **no** — only "look here first" |
| `NOT_DIAGNOSTIC` | wired up but not usable on this data | **no**, and it must not feed other stages |

If a new stage cannot honestly claim `MODEL` or `MEASUREMENT`, it reports the
measurement plus what would be needed to interpret it, and nothing else.
`tests/test_pipeline_synthetic.py::test_report_does_not_invent_a_diagnosis`
enforces the wording side of this.

## Non-negotiables when changing code

1. **Canonical orientation.** Load images with `utils.load_canonical()`. After
   that axis 0 = L→R, 1 = P→A, 2 = I→S. Never assume orientation from `nib.load`.
2. **Sides come from labels, not geometry.** SPINEPS labels articular processes
   per side (45/47 left, 46/48 right; costal 43/44) — see `labels.py`. Never split
   an array at `shape[0] // 2`: it assumes the patient's midline sits at the centre
   of the field of view and can mirror left and right outright.
3. **Resample masks onto images, never images onto masks.** Interpolating
   intensities (order=3) plus zero-padding outside the source FOV manufactures
   outliers. Masks move with `order=0`.
4. **Thresholds come from reference tissue.** `analysis.robust_threshold()` on a
   tissue that is *not* under test. A percentile inside the measured region is
   self-referential and saturates when one side is uniformly bright.
5. **No fat suppression, no oedema.** If `picks["FATSAT_BEST"]` is None, the stage
   raises `SkipStage` with that reason. A bright-voxel count on plain T2 cannot
   separate oedema from normal fatty marrow — on T2 fatty marrow is the *brighter*
   of the two.
6. **Names, not raw ids.** `labels.vertebra_name(18)` → `"T11"`,
   `labels.tss_disc_name(81)` → `"T10-T11"`. An unknown id becomes `id_NN`, never
   a guess.
7. **No magic numbers in stages.** Upstream label values live in `labels.py`;
   thresholds live in `Config`.
8. **Never mutate the environment mid-run.** No pip install inside a stage, no
   writing to `sitecustomize.py`, no monkeypatching `torch.load`, no
   `os.kill(os.getpid(), 9)`. All of these were in the old notebook; see the bug
   ledger.
9. **PHI stays out of the repo and out of the report.** Data lives on Drive. The
   report prints `subject_id` only.

## This study's actual data (checked with `spinelab inspect`)

Sagittal T2 and T1 (17 slices, 3.5 mm), coronal STIR (23 slices, 4 mm, TI 100 ms —
the only fat-suppressed series), axial T2 (66 slices, 4 mm, **0.49 mm in-plane**),
plus a scout. 1.5 T.

Consequences that shape every design decision here:
* the axial series is the only plane where thoracic facet joints are resolvable, and
  it carries the finest resolution in the study — `facets_axial` exists for it;
* oedema can only come from the coronal STIR, and only if `fatsat_qc` confirms the
  suppression works;
* masks are computed on the sagittal T2, whose left-right voxel size (3.5 mm) is the
  size of a facet joint — so mask-derived left/right precision is coarse, and that
  caveat belongs in any output that compares sides.

## Alignment and reliability stages

`register` moves the SPINEPS masks (computed on sagittal T2) onto the
fat-suppressed series with a rigid, mutual-information refinement of the header
alignment. Rationale: a facet joint is 2-4 mm across, smaller than normal
inter-series patient motion. A correction beyond 15 mm / 10 deg is rejected as a
failed optimisation rather than trusted. Downstream stages check
`ctx.masks_in_fatsat_space()` and record `masks_motion_corrected`; when it is
False they must state that the masks are header-placed and that small left/right
differences are within alignment noise.

`crosscheck` (quality profile) runs TotalSegmentator's `vertebrae_mr` and reports
per-level Dice against SPINEPS. Registration cannot add coverage: structures
outside the fat-suppressed slab stay unmeasurable, and that is reported, not
interpolated over.

Mirror TTA (`Config.tta_mirror`) re-runs the segmenter on an L-R mirrored copy,
mirrors the result back and swaps side-specific label ids
(`analysis.mirror_side_labels`: 43↔44, 45↔46, 47↔48, 63↔64). Note the subtlety a
test already caught: a model names sides by *appearance*, so on a mirrored study
the anatomically-left structure gets the right-side label — which is why the swap
is needed and why simulating the model as identity-on-labels is wrong. Low
agreement here invalidates every left/right number in the run, and the report says
so.

## Adding a stage

```python
# spinelab/stages/mystage.py
def run(ctx: Context) -> StageResult:
    image = ctx.require_sequence("T2_SAG")        # raises SkipStage when absent
    masks = ctx.stage_data("spineps")["instance_masks"]
    ...
    return StageResult(name="mystage", status=Status.OK,
                       evidence=Evidence.MEASUREMENT, data=payload)
```

Then register it in `pipeline._registry()` and add it to `config.DEFAULT_STAGES` in
dependency order. A stage reads previous results through `ctx.stage_data(name)`,
which falls back to `results/stages/<name>.json` — so stages stay runnable in
isolation and a run resumes after a Colab disconnect.

Distinguish the two failure kinds:
* inputs legitimately absent → `raise SkipStage("why")` → status `skipped`;
* something broke → let it raise, or return `Status.FAILED` with a reason.
Conflating them is what made the old status table unreadable.

Put the maths in `analysis.py` (pure numpy, no I/O) and test it there. Stage
modules should be thin I/O wrappers.

## Testing

```bash
python -m pytest -q          # 97 tests, ~1 s, no GPU and no patient data
```

`tests/test_pipeline_synthetic.py` builds a phantom study (block vertebrae, facet
labels per side, a planted bright focus) and runs the analysis stages end to end
with the segmentation stages pre-seeded. Any new stage that produces a
side-comparison or a per-level number needs a case there — in particular a case
where the answer must be a **refusal** (missing sequence, unequal coverage).

The segmentation stages themselves need Colab with a GPU. Do not claim they work
without a real run; say "requires a Colab run" instead.

## Colab notes

`notebooks/colab_pipeline.ipynb` is 4 cells and stays that way. Weight caches are
env-var driven (`Config.export_env`): `SPINEPS_SEGMENTOR_MODELS`,
`TOTALSEG_HOME_DIR`, `TOTALSPINESEG_DATA`, `nnUNet_results`, `HF_HOME`,
`TORCH_HOME` — all under a Drive-backed directory, because the operator's
connection is slow and re-downloading ~7 GB per session is the dominant cost.

## Read before proposing changes to method

* `docs/bug-ledger-2026-07-27.md` — the 26 defects and why each fix looks the way
  it does. Most "obvious simplifications" here reintroduce one of them.
* `docs/clinical-context.md` — what this data cannot answer regardless of code.
* `docs/stack-review-2026-07.md` — which tools are in, which are out, and why
  (U²AD targets spinal-cord hyperintensity, not vertebral marrow; SpineNet is
  lumbar-anchored; zero-shot VLM/CLIP output is not evidence).
