# Research-промпт для Gemini 3.1 Pro Deep Research Max
**Дата**: 2026-05-21
**Цель**: получить исчерпывающий обзор open-source AI-инструментов для анализа МРТ грудного отдела позвоночника, способных выявить тонкие находки, упущенные радиологом.

---

## Как использовать

1. Открыть **Google AI Studio** → новый чат.
2. Выбрать модель: **Deep Research Max Preview (Apr-2026)** (`deep-research-max-preview-04-2026`).
3. Включить в Run settings → Tools:
   - ✅ Code execution
   - ✅ Grounding with Google Search
   - ✅ URL context
   - ✅ Thinking summaries
4. Скопировать **весь блок ниже** (от первой строки `# ROLE` до `Begin the report now.`) и вставить в поле ввода.
5. Запустить. Ожидаемое время выполнения: десятки минут (deep research).
6. Полученный отчёт сохранить в `docs/gemini-research-report-2026-05-21.md` для следующего шага планирования.

---

## Клинический контекст (для справки, в промпт уже включён в обезличенном виде)

- Бывший спортсмен (дзюдо/борьба).
- Внезапная "сковывающая" боль в грудном отделе между лопатками.
- Провокация: сведение лопаток, разгибание грудного отдела, длительное лежание на спине, асимметричная нагрузка.
- Диагностическая зацепка: контралатеральный паттерн при наклонах (наклон влево → боль справа; наклон вправо → боль слева) — указывает на возможную дисфункцию рёберно-позвоночных / рёберно-поперечных суставов или фасеточный синдром грудного отдела.
- Частичное облегчение от тракции (вис) и тепла.
- Анамнез: длительная асимметричная нагрузка (сумка + рюкзак с ноутбуком/кимоно), холмистая местность.
- **Радиолог при первичном просмотре МРТ не нашёл значимых изменений. Грыж и протрузий нет.**
- Симптоматика явно органическая → высокая вероятность субтильных находок, упущенных при человеческом просмотре.

## Данные

- 1 пациент, 1 исследование: МРТ грудного отдела позвоночника, DICOM-формат, 129 файлов (~36 МБ).
- Архив: `OMER_NURIYEV (RAMIN)_dcm_*.zip` в корне репозитория.

## Вычислительная среда

- Google Colab Pro+ с одним H100 (80 ГБ) или A100 (40/80 ГБ).
- Python 3.10+, CUDA 12.x.

---

## Промпт (копировать ВСЁ ниже от `# ROLE` до `Begin the report now.`)

````
# ROLE
You are a senior medical-imaging AI research scientist with deep expertise in:
- Spine MRI analysis (especially the thoracic segment, T1–T12)
- Deep-learning models for medical imaging (CNNs, U-Nets, transformers, foundation models, vision-language models)
- DICOM handling, MONAI, nnU-Net, TotalSegmentator and related ecosystems
- Radiology of subtle / often-missed thoracic findings (Scheuermann's disease, Schmorl's nodes, Modic changes, costovertebral / costotransverse joint dysfunction, facet arthropathy, endplate irregularities, paraspinal muscle asymmetry, annular fissures, marrow edema, micro-fractures)

Your goal is to deliver a maximum-rigor, citation-rich research report that lets a downstream engineering team build the strongest possible open-source pipeline for analyzing one thoracic-spine MRI on a single Google Colab GPU (H100 or A100, 80 GB).

# CLINICAL CONTEXT (do NOT diagnose — use only to prioritize tools)
- Adult patient, former contact-sport athlete (judo / wrestling).
- Sudden onset of a constricting, deep mid-thoracic pain located between the scapulae.
- Pain provocation: scapular retraction, thoracic extension, prone position after a while, asymmetric loading.
- Diagnostic clue: contralateral side-bending pattern (left-bend → right-sided vertebral pain, right-bend → left-sided) — suggests possible costovertebral/costotransverse joint involvement or facet syndrome rather than disc pathology.
- Partial relief from axial traction (hanging) and heat — consistent with mechanical/joint origin.
- Chronic asymmetric load history (heavy shoulder bag + heavy backpack, uphill walking).
- Prior radiologist read of the thoracic MRI: NO significant abnormality reported. No herniation, no protrusion. This is the key driver — we need AI tools that surface subtle findings a human reader can miss.

# DATA
- 1 patient, 1 study: thoracic spine MRI, DICOM format, ~129 slices, multi-sequence (assume standard sagittal T1, sagittal T2, axial T2, possibly STIR — confirm what tools handle each).
- We have NO ground-truth labels and NO training data — we need inference-only / pretrained / foundation-model approaches.

# COMPUTE BUDGET
- Google Colab Pro+ with single H100 (80 GB) or A100 (40/80 GB).
- Linux, Python 3.10+, CUDA 12.x.
- We can install anything from PyPI / conda-forge / GitHub, including Docker-packaged tools if they can be unpacked inside Colab.
- Each tool's VRAM/compute footprint must be reported.

# WHAT I NEED FROM YOU (the report)

## Section 1 — Differential Targets the AI Stack Must Cover
For each finding below, name the imaging signature on MRI and the tool(s) best suited to detect it. Be exhaustive — this is a fishing expedition for what the human radiologist missed.
1. Scheuermann's disease (3+ adjacent vertebrae with ≥5° anterior wedging, endplate irregularity, Schmorl's nodes, increased thoracic kyphosis)
2. Schmorl's nodes (intravertebral disc herniations through endplates)
3. Modic endplate changes (Type I/II/III) — marrow signal changes
4. Facet joint arthropathy / effusion (thoracic facets are commonly missed)
5. Costovertebral and costotransverse joint dysfunction / arthrosis
6. Subtle disc dehydration / Pfirrmann grading (even without overt herniation)
7. Annular fissures / high-intensity zones
8. Paraspinal muscle atrophy / asymmetry / fatty infiltration (Goutallier grading)
9. Vertebral microfractures / occult compression
10. Bone marrow edema (STIR/T2 fat-sat findings)
11. Spinal alignment: thoracic kyphosis angle (Cobb), regional sagittal balance, scoliosis (Cobb)
12. Disc and vertebra labeling correctness
13. Intercostal / costopleural soft-tissue anomalies if visible in FOV
14. Anything else known to be under-reported on thoracic MRI

## Section 2 — Open-Source AI Tools Matrix
Produce a comparison table with these columns:
| Tool | Latest version & release date | License | Task(s) covered (from Section 1) | Inputs (MRI sequences/modalities accepted) | Output format | Pretrained weights available? | VRAM | Inference time per study | Validated on thoracic spine? | Benchmark numbers (Dice/AUC/sensitivity) | Repo URL | Last commit | Notes / caveats |

Tools to evaluate at minimum (search aggressively for newer/better ones — the field moved fast in 2024–2026):
- **Segmentation / labeling**: TotalSegmentator (and TotalSegmentator MRI), SpineSeg, VerSe-2020 / VerSe-2024 winning models, nnU-Net spine variants, MONAI Bundles for spine, SpineNetV2, Verteformer
- **Foundation models for radiology**: MedSAM / MedSAM2, SAM-Med3D, SAT (Segment Anything Tumor), MedGemma, RadFM, RadCLIP, BiomedCLIP, LLaVA-Med, Merlin, Prov-GigaPath, MAIRA (Microsoft)
- **Specialized spine models**: SpineNet (Oxford VGG, Jamaludin et al.), Disc-Locator, Vertebra-Focused Landmark Detection, automatic Pfirrmann/Modic grading networks (e.g., Niemeyer et al., DeepSPINE, SpineNetV2, lumbar models adapted to thoracic — flag adaptation cost)
- **Alignment / morphometry**: automated Cobb angle estimators, sagittal balance tools, kyphosis-angle measurement networks
- **Radiomics / classical**: pyradiomics for texture features on segmented vertebrae/discs (Modic differentiation, endplate texture)
- **Vision-language report generators**: MAIRA-2, RadFM, CheXagent-style models that can be conditioned for spine — flag if no thoracic-specific evaluation exists
- **Anomaly detection (unsupervised)**: any normative modelling / autoencoder / diffusion-based out-of-distribution detectors for spine MRI that flag voxels deviating from the healthy distribution — crucial when no ground truth exists
- **3D Slicer + plugins** (SlicerRadiomics, SlicerElastix, TotalSegmentator extension) usable headlessly in Colab via xvfb

For each tool, **explicitly note thoracic-spine support** — many spine tools are lumbar-only. If a tool is lumbar-trained but architecturally generalizes, say so and estimate adaptation effort.

## Section 3 — Recommended Pipeline (Inference-Only, Colab-Ready)
Design 2 pipelines:
- **Pipeline A — "Maximum coverage"**: stack the top complementary tools to extract every possible finding. Order of operations, intermediate artifacts, fusion of outputs.
- **Pipeline B — "Subtle-finding focus"**: optimized for catching what a human radiologist missed (anomaly detection + radiomics + foundation-model second-opinion).

For each pipeline, give:
1. Step-by-step block diagram (textual is fine)
2. Exact pip/conda install commands and any pretrained-weight URLs
3. Approximate VRAM and wall-clock per stage on H100
4. Failure modes and how to validate each step

## Section 4 — Quality, Safety, Validation
- How to sanity-check segmentations without ground truth (visual overlay strategies, anatomic plausibility checks, cross-tool agreement metrics).
- Known failure modes of each top-3 tool on thoracic spine specifically.
- Calibration: how confident can we be in subtle-finding flags? Recommend reporting strategy (heatmaps + per-finding confidence + must-be-confirmed-by-human disclaimer).
- Bias / dataset caveats — what populations were training sets drawn from.

## Section 5 — Literature Anchor
Cite the most relevant peer-reviewed papers, MICCAI/RSNA/SPIE proceedings, and challenge reports from 2022–2026 that justify each recommendation. Prefer primary sources over blog posts. For every cited paper give: authors, year, venue, DOI/arXiv, 1-sentence relevance.

## Section 6 — Gaps and Open Problems
Honestly list what current open-source AI **cannot** do well for this case and where a human spine radiologist or musculoskeletal specialist remains essential.

# OUTPUT REQUIREMENTS
- **Language: Russian.** Write the full report in Russian. Keep tool names, library names, paper titles, and code identifiers in English (do not translate them). Mixed-language output is desired.
- Structure: use the section headings above (translated to Russian) and the comparison table.
- Length: prioritize completeness over brevity. Long is fine.
- Cite sources inline with links.
- For any claim about benchmark numbers, give the source. Do NOT invent metrics — if a number is not findable, say "данных по бенчмарку не найдено".
- If two sources disagree, surface the disagreement.
- Conclude with a single prioritized "Что устанавливать первым" list (5–8 tools, ranked).

# ANTI-HALLUCINATION GUARDRAILS
- Every tool you recommend must have a verifiable public repo or paper. Provide the URL.
- Do not fabricate version numbers, weights URLs, or benchmark scores. If unsure, write "уточнить" and explain what to check.
- Distinguish clearly between "validated on thoracic MRI in the literature" and "could plausibly work but untested on thoracic".
- Flag any tool whose last commit is >18 months old as potentially stale.
- Do not give clinical advice. Frame everything as engineering input to a downstream human radiologist's review.

Begin the report now.
````

---

## Что делать после получения отчёта

1. Сохранить полный ответ Gemini в `docs/gemini-research-report-2026-05-21.md`.
2. Из секции **"Что устанавливать первым"** взять top-3 инструмента и проверить:
   - репозитории живы (последний коммит, открытые issues)
   - веса доступны без gated-access
   - лицензия допускает исследовательское использование
3. Вернуться и спланировать следующий шаг: Colab-ноутбук с распаковкой DICOM → конвертацией в NIfTI → запуском выбранных моделей → визуализацией.
