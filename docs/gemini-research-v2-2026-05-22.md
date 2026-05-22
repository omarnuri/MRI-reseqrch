# Gemini Deep Research v2 — поиск инструментов для precision boost

**Дата**: 2026-05-22  
**Контекст**: после ~5 часов итераций мы имеем working end-to-end pipeline (~92-95% от исходной цели). Эта итерация ищет конкретные инструменты которые могут поднять precision до diagnostic-grade.

## Как использовать

1. Открыть **Google AI Studio** → новый чат
2. Выбрать модель: **Deep Research Max Preview (Apr-2026)**
3. Включить: Code execution, Grounding with Google Search, URL context, Thinking summaries
4. Скопировать **весь блок ниже** (от `# ROLE` до `Begin the report now.`) в поле ввода
5. Запустить. Ожидаемое время: 20-40 мин deep research
6. Сохранить отчёт в `docs/gemini-research-v2-report-2026-05-22.md`

---

## Промпт (копировать целиком от `# ROLE` до `Begin the report now.`)

````
# ROLE
You are a senior medical-imaging AI research scientist specialized in thoracic spine MRI analysis. You have already produced one research report for this team 1 day ago (gemini-research-2026-05-21.md). They have now built a working end-to-end Colab pipeline based on that report and need a SECOND, more focused research pass to push the system from ~92% accuracy to "diagnostic-grade".

# WHAT THE TEAM ALREADY HAS WORKING

Do NOT recommend these — they are integrated and producing results:

| Tool | Source / Version | Output |
|------|------------------|--------|
| dcm2niix | apt | DICOM → NIfTI |
| SPINEPS | pip 1.x (Hendrik-code) | 14 thoracic vertebrae body + posterior elements, centroids (corpus label 49 isolation working) |
| TotalSpineSeg | pip 20260429 (neuropoly) | Spinal cord, canal, 14 IVDs with label ≥63 filtering, step1/step2 outputs |
| TotalSegmentator MR (v2.13) | pip (wasserth) | autochthon L/R, scapula, ribs, intervertebral_discs combined, 30 organs/muscles |
| MedGemma-4b-it / 27b-it | HuggingFace (google) | VLM second opinion via HF_TOKEN — DON'T find replacements for this |
| BiomedCLIP (PubMedBERT + ViT-B/16) | open_clip_torch | Image-text similarity ranking against 6 differential queries |
| Robust MAD anomaly per-vertebra | custom Python | Outlier detection on STIR/T2 with FOV coverage gating |
| Goutallier-style fatty fraction | T1 p75-within-muscle threshold | Found 24.9% right autochthon fat — important biomarker |
| nibabel canonical resample | standard | Cross-grid mask comparison |
| pyradiomics → skimage GLCM fallback | pip | First-order + texture features per vertebra |

# WHAT IS BROKEN / GATED / SUBOPTIMAL — these need replacements

| Tool we tried | Why it failed | What we need instead |
|---------------|---------------|----------------------|
| **SpineNetV2** (Jamaludin et al. 2017) | Trained ONLY on lumbar discs — produces incorrect Pfirrmann grades on thoracic | Open-weight thoracic-validated disc degeneration grader (Pfirrmann-equivalent or DDD score) |
| **bowang-lab/SpineNet** fork | Inference API undocumented, signature unstable | Same need as above |
| **MedSAM2** (wanglab) | HF processor class unrecognized; prompt format unclear when called from `transformers` | Open prompt-based 3D medical segmenter with documented Python API |
| **U2AD** (zhibaishouheilab) | Requires manual weight setup, no Python entry point | Open-weight unsupervised anomaly detector for thoracic STIR/T2 hyperintensity |

# CLINICAL CONTEXT — your search is for THIS specific case

- Adult male, former contact-sport athlete (judo / wrestling)
- Chronic mid-thoracic pain between scapulae
- KEY clinical sign: contralateral side-bending pattern (bend left → right-side vertebral pain, bend right → left-side) — anatomically points to costovertebral / costotransverse joints or facet syndrome
- Asymmetric loading history: heavy shoulder bag on left + heavy backpack
- Previous radiologist read: NO significant findings reported
- We have multi-sequence MRI: T2 sagittal, T1 sagittal, T2 axial, **T2 coronal STIR**
- Current AI findings (working pipeline):
  - T1 wedge +5.2°, **T2 wedge +10.4°**, T10 +4.7° (Scheuermann threshold ≥5°, formal criterion needs 3 contiguous — we have 2 + borderline)
  - STIR/T2 anomaly hotspots in T3-T9 region (matches inter-scapular pain zone)
  - autochthon R 9.34% larger CSA + **24.9% T1 fatty infiltration** (vs 0% L)
  - Costovertebral subtle bright signal 1.74-1.8× R > L
  - Canal stenosis EXCLUDED (max narrowing 16.91%)
  - Cross-tool SPINEPS vs TSS cord Dice 0.765

# COMPUTE ENVIRONMENT

- Google Colab Pro+ with NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM) OR A100 80GB / H100 80GB
- Python 3.12, CUDA 13.0+, torch 2.10-2.12 (volatile across Colab sessions)
- numpy 2.x ABI, scikit-image 0.26+
- **Real measured pipeline peak VRAM: ~14-15 GB** (each tool releases before next; far below theoretical sum)
- We track per-tool VRAM + wall-clock in `performance_log.json` for impact-per-cost analysis

# WHAT I NEED FROM YOU

## Section 1 — Open-weight replacements for the 4 broken/gated tools

For each of (SpineNetV2 thoracic alternative), (MedSAM2 alternative), (U2AD alternative), give 2-4 candidates with:
- HuggingFace OR GitHub URL (verify it exists)
- Weights URL (must be public, non-gated, downloadable without login OR with documented free token flow)
- Last commit date (must be < 18 months ago to be considered current)
- Python API entrypoint OR working CLI command
- Validated on thoracic spine in literature (YES/NO/PARTIAL with citation)
- Benchmark numbers if available (Dice, AUC, sensitivity vs radiologist)
- VRAM footprint, inference time per study
- License (MUST be open — MIT / Apache / BSD / non-commercial OK if research)
- Integration effort estimate (HOURS to wire into our pipeline)

## Section 2 — Patho-specific models for the 6 identified gaps

For each of:
1. **Modic Type I/II/III classifier** (current need: per-vertebra absolute Modic grade, not just "anomaly score")
2. **Scheuermann disease automatic grading** (need: probability + formal Cobb angle + endplate irregularity score)
3. **Costovertebral / costotransverse joint segmentation or scoring** (most critical for this patient — no current tool segments these specifically)
4. **Schmorl node detection** (endplate herniation)
5. **Annular fissure / HIZ (high-intensity zone) detector** for discs
6. **Paraspinal muscle Goutallier 0-IV grading** (formal classification, not our percentile proxy)

For each pathology, list 1-3 candidate tools with same matrix as Section 1.

## Section 3 — Foundation models (2026 vintage, post-MedGemma-27b)

We already use MedGemma-4b/27b through HF_TOKEN. Look for newer (2025 Q4 — 2026 Q2):
- **3D medical foundation models** that can be prompted for thoracic spine tasks (RadFM successors, Merlin v2 if exists, MAIRA-3, MONAI Bundles 2026 catalog)
- **Open vision-language radiology models** that could give differential reports on cropped vertebra patches
- **Specialized spine foundation models** trained on large vertebral datasets

For each: same matrix + specific note on whether thoracic data was in training set.

## Section 4 — Ensemble and uncertainty methods

The team needs to combine multiple independent detectors into a single "high-confidence vertebra" ranking. Find:
- **Calibration libraries** for medical AI ensembles (Platt scaling, isotonic regression, temperature scaling for our multiple classifiers)
- **Uncertainty quantification** approaches (MC dropout, deep ensembles, conformal prediction, evidential learning) usable on top of existing inference models without retraining
- **Disagreement-based human-in-loop systems** that surface ONLY the cases where ensemble votes split

Each with code repository + Python API + integration effort.

## Section 5 — Integration recipes

For your top-3 recommended tools across Sections 1-3, write a concrete integration snippet (10-30 lines Python) showing:
1. How to download/load weights
2. How to run inference on T2_SAG / T1_SAG / STIR (specify which)
3. How to post-process output into our existing per_vertebra / per_disc dict format
4. How to compute Dice or agreement against existing SPINEPS/TSS masks (where applicable)

# OUTPUT REQUIREMENTS

- **Language: Russian** for narrative sections, but keep tool names, code, paper titles, URLs, parameter names in English
- Structure: use the 5 section headings above translated to Russian
- **Length: be exhaustive** — better to have 4 candidates per slot with comparison than 1 candidate
- Cite papers inline (DOI, arXiv, MICCAI/RSNA/SPIE proceedings)
- For every claim about a benchmark number — provide source URL or paper citation
- If a tool you'd otherwise recommend has last_commit > 18 months, flag it as STALE and propose newer alternative
- End with a **prioritized "Top-5 to integrate first"** list ranked by (clinical impact × ease of integration) / VRAM cost

# ANTI-HALLUCINATION GUARDRAILS

- Every URL must be verifiable — if you're unsure, write "URL needs verification" not a guessed link
- Do not invent benchmark numbers — if not findable, write "benchmark not published, would need own evaluation"
- Distinguish clearly between "validated on thoracic spine in published literature" vs "could plausibly generalize but untested on thoracic"
- If a tool exists only as a research code dump with no released weights, mark it as "research-only, weights need request from authors"
- Do not recommend commercial / proprietary tools (Aidoc, Zebra Medical, RAD-AI, FDA-approved closed-source products)
- Do not recommend tools requiring custom training from scratch — we are inference-only
- Do not give clinical diagnoses or treatment advice; this is a tooling assessment

Begin the report now.
````

**Конец промпта.**

---

## Что делать после получения отчёта от Gemini

1. Сохранить полный отчёт в `docs/gemini-research-v2-report-2026-05-22.md`
2. Из Section 5 "Top-5 to integrate first" взять top-3 и проверить **каждый URL** через WebFetch — реальный live test, не trust Gemini blindly
3. Из проверенных — выбрать 2-3 которые:
   - Не gated
   - Last commit < 18 мес
   - Имеют documented Python API
   - Validated на thoracic (хоть в каком-то bench)
4. Интегрировать в notebook (Cells 21-25) **по одному с измерением performance** через `measure_tool`
5. После каждого нового tool — пересчитать confluence (Cell 19) и accuracy audit (Cell 18); добавить новые baselines в `accuracy_audit.json`

## Acceptance criteria

После v19 integration ожидаем:
- Минимум 3 новых "method" в confluence (помимо robust MAD + BiomedCLIP)
- T5/T3 (наши топ-аномалии) flagged ≥4 independent methods
- L1/C7 (FOV-edge) flagged ≤1 method
- Один из: реальный Modic grade ИЛИ Scheuermann probability ИЛИ Goutallier 0-IV в findings.json
- `performance_log.json` показывает peak <30 GB, total wall-clock <45 мин
