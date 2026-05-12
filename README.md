---
title: Aviation Intelligence System
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: app/app.py
pinned: false
license: mit
short_description: Identify aircraft, check route, explain in plain English.
---

# AI Applications Project — Aviation Intelligence System

> ZHAW *AI Applications* — FS26 Semester Project · **Tim Dubath**

A multimodal AI assistant for aviation enthusiasts. Upload a photo of an aircraft, pick a
route, and the system identifies the aircraft, predicts whether the aircraft can realistically
fly that route, and explains the verdict in natural language with Wikipedia citations.

Detailed methodology, metrics, and reflections live in
[`DOCUMENTATION.md`](DOCUMENTATION.md). This README follows the course template and points
into the codebase for evidence.

## Project Metadata

- **Project title:** Aviation Intelligence System
- **Student:** Tim Dubath
- **GitHub repository URL:** https://github.com/TimDubath-dev/aviation-intelligence-system
- **Deployment URL:** https://huggingface.co/spaces/dubattim/aviation-intelligence-system
- **Submission date:** 2026-05-12

### Mandatory Setup Checks

- [x] At least 2 blocks selected (all 3 implemented)
- [x] Multiple and different data sources used (6 sources, see [§2.1 of `DOCUMENTATION.md`](DOCUMENTATION.md#21-data-sources))
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution:
- **Primary block 1:** Computer Vision — aircraft variant identification
  ([`src/cv/`](src/cv))
- **Primary block 2:** NLP / RAG — grounded natural-language explanation
  ([`src/nlp/`](src/nlp))
- **Third block (graded as extra work):** ML on Numeric Data — route-feasibility
  classification ([`src/numeric/`](src/numeric))

All three blocks are chained end-to-end: CV decides *which plane*, the spec lookup pulls the
matching row, the numeric model predicts feasibility, and the LLM produces a grounded
explanation given all upstream outputs.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- **Problem statement:** Determining whether a specific aircraft can realistically fly a
  specific route currently requires manually chaining aircraft identification, spec lookup,
  distance computation, feasibility reasoning, and natural-language explanation — across
  multiple tools.
- **Goal:** Compress this entire chain into one click: photo + route → identified aircraft +
  feasibility verdict + grounded explanation.
- **Success criteria:**
  1. CV top-5 ≥ 95% on the FGVC-Aircraft test split.
  2. Numeric model ROC-AUC ≥ 0.90 on the held-out test split (and on the difficult segment).
  3. RAG explanation faithfulness ≥ 4 / 5 with ≥ 80% source-citing rate.
  4. End-to-end inference < 10 s on HF Spaces CPU basic.

All four targets are met — see [§4 of `DOCUMENTATION.md`](DOCUMENTATION.md#4-evaluation--analysis).

### 1.2 Integration Logic
- **How the selected blocks interact:** Chained pipeline — every block consumes the previous
  block's output. Implementation: [`src/pipeline.py`, lines 71-150](src/pipeline.py#L71-L150)
  (the `run()` orchestrator: CV at [L74-99](src/pipeline.py#L74-L99), spec lookup at
  [L101-102](src/pipeline.py#L101-L102), numeric at [L108-116](src/pipeline.py#L108-L116),
  RAG + LLM at [L118-138](src/pipeline.py#L118-L138)).

  ```
  Photo ─► [CV: DINOv2] ─► variant ─► [Spec lookup] ◄─ Route (origin, dest)
                                            │
                                            ▼
                                  [Numeric ML: XGBoost]
                                            │
                                            ▼
                                  [RAG: FAISS + MiniLM]
                                            │
                                            ▼
                                  [LLM: GPT-4o-mini / Haiku]
                                            │
                                            ▼
                                  Explanation + citations
  ```
- **Data and output flow between blocks:**
  - CV → spec lookup: predicted variant string (e.g. `A320`).
  - Spec lookup → numeric: dict of structured fields (range, MTOW, ETOPS, engines).
  - Numeric → NLP: feasibility probability (float) + verdict (bool).
  - RAG → LLM: top-4 Wikipedia chunks injected into the prompt.

![Architecture Diagram](docs/architecture.png)

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Curated aircraft specs CSV ([`data/raw/curated_aircraft_specs.csv`](data/raw/curated_aircraft_specs.csv)) | Tabular | 100 rows × 12 columns | Source of the per-aircraft features (range, MTOW, ETOPS, engines) |
| 2 | [OpenFlights airports](https://openflights.org) ([`data/raw/openflights/airports.dat`](data/raw/openflights/airports.dat); loader at [`src/numeric/build_dataset.py`, lines 53-64](src/numeric/build_dataset.py#L53-L64)) | Tabular | 7,698 airports | Great-circle distance between origin and destination |
| 3 | Synthesized route-feasibility dataset ([`src/numeric/build_dataset.py`, lines 96-134](src/numeric/build_dataset.py#L96-L134)) | Tabular | 50,000 `(aircraft, route)` rows | Training & evaluation labels |

#### 2A.2 Preprocessing and Features
- **Cleaning steps:** Specs hand-curated from Wikipedia infoboxes + manufacturer datasheets;
  units standardized to metric; OpenFlights filtered to entries with valid IATA + ICAO
  (drops helipads/seaplane bases). See
  [§2.2 of `DOCUMENTATION.md`](DOCUMENTATION.md#22-data-cleaning--preprocessing).
- **Preprocessing steps:** 50k synthetic `(aircraft, origin, destination)` triples with
  per-flight headwind perturbation `N(20, 25) km/h` and payload factor `Beta(2, 2)`
  ([`src/numeric/build_dataset.py`, lines 107-108](src/numeric/build_dataset.py#L107-L108)),
  3% label noise ([`src/numeric/build_dataset.py`, lines 111-113](src/numeric/build_dataset.py#L111-L113)),
  fixed RNG seed `42` at [`src/numeric/build_dataset.py`, line 39](src/numeric/build_dataset.py#L39).
  The physics-based label rule is implemented in
  [`src/numeric/build_dataset.py`, lines 80-93](src/numeric/build_dataset.py#L80-L93).
- **Feature engineering and selection:** 9 numeric + manufacturer one-hot. Key engineered
  features: `range_margin_ratio = distance/range`, `payload_proxy`, `long_haul`,
  `transoceanic`, `twin_engine`, `etops_capable`
  ([`src/numeric/features.py`, lines 11-16](src/numeric/features.py#L11-L16); feature matrix
  assembly at [`src/numeric/features.py`, lines 19-29](src/numeric/features.py#L19-L29)).

#### 2A.3 Model Selection
- **Models tested:** Logistic Regression, MLP (64, 32), XGBoost.
- **Why these models were chosen:** Linear baseline (LogReg) to confirm a strong signal
  exists; MLP to capture nonlinear interactions in normalized feature space; XGBoost for
  calibrated probabilities on tabular data with mixed numeric/boolean features.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline | Trivial labels (rule on range only), no noise | LogReg ([`src/numeric/train.py`, lines 48-50](src/numeric/train.py#L48-L50)) | ROC-AUC 0.998 (overall) | — |
| 2 | Make the task non-trivial | Added headwind, payload, 3% label noise, weighted hard-segment sampling ([`src/numeric/build_dataset.py`, lines 80-113](src/numeric/build_dataset.py#L80-L113)) | LogReg, MLP, XGBoost ([`src/numeric/train.py`, lines 46-77](src/numeric/train.py#L46-L77)) | ROC-AUC 0.95 overall / 0.91 hard | -0.05 (intentional, exposes model differences) |
| 3 | Pick winner + calibration | Final hyperparameter sweep + Brier-score evaluation ([`src/numeric/train.py`, lines 110-123](src/numeric/train.py#L110-L123)) | LogReg, MLP, **XGBoost** ([`src/numeric/train.py`, lines 65-77](src/numeric/train.py#L65-L77)) | **ROC-AUC 0.956**, **F1 0.927**, **Brier 0.032** | +0.005 vs MLP, +0.003 vs LogReg |

Full per-model metrics: [`models/numeric/metrics.json`](models/numeric/metrics.json).

#### 2A.5 Evaluation and Error Analysis
- **Metrics used:** Accuracy, F1, ROC-AUC, Brier score
  ([`src/numeric/train.py`, lines 80-88](src/numeric/train.py#L80-L88)); per-segment metrics
  on the difficult band `distance/range ∈ [0.7, 1.1]`
  ([`src/numeric/train.py`, lines 104-107](src/numeric/train.py#L104-L107));
  5-fold stratified CV ([`src/numeric/train.py`, lines 91-101](src/numeric/train.py#L91-L101));
  permutation importance ([`src/numeric/train.py`, lines 157-171](src/numeric/train.py#L157-L171)).
- **Final results (XGBoost, 20% test split):** Accuracy 96.4%, F1 0.927, ROC-AUC 0.956,
  Brier 0.032. Hard segment: F1 0.862, ROC-AUC 0.949.
- **Error patterns and likely causes:** Residual errors concentrate in the hard segment where
  unobserved headwind and payload shift the effective range across the feasibility threshold.
  Permutation importance shows the model correctly relies on `range_margin_ratio` (0.331) and
  `payload_proxy` (0.004); manufacturer dummies are noise — exactly the ground-truth structure.

#### 2A.6 Integration with Other Block(s)
- **Inputs received from other block(s):** Variant name from CV → spec lookup → feature vector.
- **Outputs provided to other block(s):** Feasibility probability + verdict, injected into
  the NLP prompt so the LLM can reference the numeric verdict in its explanation.

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Wikipedia aircraft & airport articles (fetched via REST API: [`src/nlp/build_index.py`, lines 31-46](src/nlp/build_index.py#L31-L46)) | Unstructured text | ~120 articles → 1,236 chunks (~500 words each, [`src/nlp/build_index.py`, lines 49-51](src/nlp/build_index.py#L49-L51)) | RAG grounding corpus |
| 2 | Numeric pipeline output (variant, specs, route, feasibility probability) | Structured context | per request | Injected into the LLM prompt at [`src/pipeline.py`, lines 119-138](src/pipeline.py#L119-L138) |
| 3 | Hand-crafted few-shot examples ([`src/nlp/prompts.py`, lines 37-50](src/nlp/prompts.py#L37-L50)) | Text | 2 examples (Cessna 172 ZRH→JFK, A350 ZRH→NRT) | Few-shot prompt strategy |

#### 2B.2 Preprocessing and Prompt Design
- **Text preprocessing:** Wikipedia `action=query&prop=extracts&explaintext=1`
  ([`src/nlp/build_index.py`, lines 31-46](src/nlp/build_index.py#L31-L46));
  chunked at ~500 words, no overlap
  ([`src/nlp/build_index.py`, lines 49-51](src/nlp/build_index.py#L49-L51));
  embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized) and indexed
  in FAISS `IndexFlatIP`
  ([`src/nlp/build_index.py`, lines 91-103](src/nlp/build_index.py#L91-L103)).
- **Prompt design or retrieval setup:** Top-4 cosine retrieval via
  [`src/nlp/retriever.py`, lines 24-33](src/nlp/retriever.py#L24-L33); system prompt at
  [`src/nlp/prompts.py`, lines 9-14](src/nlp/prompts.py#L9-L14); zero-shot template at
  [L16-23](src/nlp/prompts.py#L16-L23); RAG template at
  [L25-35](src/nlp/prompts.py#L25-L35). LLM calls in
  [`src/nlp/generate.py`, lines 12-21](src/nlp/generate.py#L12-L21) (OpenAI) /
  [L24-33](src/nlp/generate.py#L24-L33) (Anthropic).

#### 2B.3 Approach Selection
- **Approach used:** Retrieval-Augmented Generation (RAG) with FAISS + MiniLM and a hosted
  LLM (GPT-4o-mini primary, Claude Haiku secondary).
- **Alternatives considered:** Pure prompt engineering (no retrieval), classical extractive
  QA on Wikipedia, and fine-tuning a smaller LM. RAG was chosen because (a) the explanation
  must cite specific aircraft specs that change over time, (b) fine-tuning is overkill for
  120 articles, and (c) retrieval guards against hallucinated specs for niche aircraft.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline (parametric memory only) | No retrieval, no examples ([`src/nlp/prompts.py`, lines 16-23](src/nlp/prompts.py#L16-L23)) | `gpt-4o-mini`, zero-shot | Faithfulness 3.4 / 5, 0% citations | — |
| 2 | Add retrieval | FAISS top-4 ([`src/nlp/retriever.py`, lines 24-33](src/nlp/retriever.py#L24-L33)) + "cite source titles" instruction ([`src/nlp/prompts.py`, lines 25-35](src/nlp/prompts.py#L25-L35)) | `gpt-4o-mini`, RAG | **Faithfulness 4.6 / 5, 85% citations** | +1.2 faithfulness |
| 3 | Add worked examples | 2 few-shot examples prepended ([`src/nlp/prompts.py`, lines 37-50](src/nlp/prompts.py#L37-L50)) | `gpt-4o-mini`, RAG + few-shot | Faithfulness 4.5 / 5, 90% citations | +5pp citation rate, no helpfulness gain |

Full responses for all 120 evaluation runs: [`models/nlp/eval_results.json`](models/nlp/eval_results.json).

#### 2B.5 Evaluation and Error Analysis
- **Evaluation strategy:** 20 hand-crafted questions (easy / medium / hard / edge cases)
  scored on faithfulness, helpfulness, and source-citing rate; cross-model comparison
  (GPT-4o-mini vs Claude Haiku); hallucination probe on 5 non-existent aircraft.
- **Results:** RAG (4.6 faithfulness) clearly beats zero-shot (3.4); few-shot is roughly
  equivalent to plain RAG. Both LLMs correctly refuse to fabricate specs for non-existent
  aircraft when RAG returns no relevant chunks.
- **Error patterns and likely causes:** Zero-shot occasionally hallucinates plausible-but-wrong
  specs (e.g. ATR-72 range stated as 3,000 km vs the correct 1,528 km); without retrieval the
  LLM falls back to potentially outdated parametric memory. Claude Haiku is slightly more
  prone to drifting away from the provided specs to add narrative context.

#### 2B.6 Integration with Other Block(s)
- **Inputs received from other block(s):** Variant name (from CV), structured specs (spec
  lookup), feasibility probability + verdict (from numeric model).
- **Outputs provided to other block(s):** Final user-facing string — the natural-language
  explanation surfaced in the Gradio UI.

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) (Oxford VGG, `torchvision.datasets.FGVCAircraft`) | Images (JPEG) | 10,000 images / 100 variant classes | CV training + evaluation (train/val/test split as supplied) |
| 2 | Wikimedia Commons scrape ([`src/cv/scrape_extra_images.py`, lines 32-46](src/cv/scrape_extra_images.py#L32-L46); category overrides at [L54-60](src/cv/scrape_extra_images.py#L54-L60)) | Images (JPEG) | 2,001 additional images across 100 classes | Training-set augmentation (variable per-class coverage) |
| 3 | [OpenSky aircraft database](https://opensky-network.org) (Oct 2024 snapshot, loaded via [`src/cv/build_registration_lookup.py`](src/cv/build_registration_lookup.py)) | Tabular | 601,270 records → 52,044 mapped to FGVC variants | OCR registration → variant lookup table |

#### 2C.2 Preprocessing and Augmentation
- **Image preprocessing:** Wikimedia images filtered to ≥256 px short side
  ([`src/cv/scrape_extra_images.py`, line 33](src/cv/scrape_extra_images.py#L33)); validation/test
  uses deterministic `Resize` + `CenterCrop`
  ([`src/cv/train_vit.py`, lines 42-45](src/cv/train_vit.py#L42-L45)).
- **Augmentation strategy:** local pipeline (`RandomResizedCrop`, `RandomHorizontalFlip`,
  `ColorJitter`) at [`src/cv/train_vit.py`, lines 35-41](src/cv/train_vit.py#L35-L41); the
  stronger `RandAugment` + `RandomErasing` schedule used for the final DINOv2 run lives in
  [`notebooks/train_vit_colab.ipynb`](notebooks/train_vit_colab.ipynb). Training arguments
  (LR, schedule, FP16, batch size): [`src/cv/train_vit.py`, lines 81-98](src/cv/train_vit.py#L81-L98).

#### 2C.3 Model Selection
- **Vision model(s) used:** Primary — fine-tuned `facebook/dinov2-base` (86M params; final
  schedule in [`notebooks/train_vit_colab.ipynb`](notebooks/train_vit_colab.ipynb), local
  reproduction stub at [`src/cv/train_vit.py`, lines 67-109](src/cv/train_vit.py#L67-L109)).
  Baseline — `openai/clip-vit-large-patch14` zero-shot with prompt template at
  [`src/cv/clip_baseline.py`, line 22](src/cv/clip_baseline.py#L22) and eval loop at
  [L31-41](src/cv/clip_baseline.py#L31-L41).
  Tiebreaker — EasyOCR + OpenSky registration lookup, regex patterns at
  [`src/cv/ocr.py`, lines 28-32](src/cv/ocr.py#L28-L32), one-shot detection at
  [`src/cv/ocr.py`, lines 103-112](src/cv/ocr.py#L103-L112), promotion logic at
  [`src/pipeline.py`, lines 82-99](src/pipeline.py#L82-L99).
- **Why these model(s) were chosen:** DINOv2 self-supervised features transfer exceptionally
  well to fine-grained recognition; CLIP zero-shot quantifies how far transfer goes without
  fine-tuning; the OCR tiebreaker specifically targets within-family confusion (737-300 vs
  737-400) which is the model's dominant error mode.

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Zero-shot floor | No training, prompt template `"a photo of a {variant} aircraft"` ([`src/cv/clip_baseline.py`, line 22](src/cv/clip_baseline.py#L22)) | CLIP-L/14 zero-shot ([`src/cv/clip_baseline.py`, lines 24-41](src/cv/clip_baseline.py#L24-L41)) | Top-1 32.8%, Top-5 77.4% | — |
| 2 | Supervised fine-tune | DINOv2-base, FGVC-only, 10 epochs (training args at [`src/cv/train_vit.py`, lines 81-98](src/cv/train_vit.py#L81-L98)) | DINOv2-base (FGVC) | Top-1 ~80%, Top-5 ~95% | +47pp Top-1 |
| 3 | Add data + longer schedule | Add 2k Wikimedia images ([`src/cv/scrape_extra_images.py`](src/cv/scrape_extra_images.py)), 20 epochs, RandAugment, RandomErasing ([`notebooks/train_vit_colab.ipynb`](notebooks/train_vit_colab.ipynb)) | **DINOv2-base (FGVC + extras)** | **Top-1 84.5%, Top-5 97.0%** | +4.5pp Top-1, +2pp Top-5 |
| 4 (orthogonal) | Resolve within-family confusion | EasyOCR + OpenSky registration lookup ([`src/cv/ocr.py`, lines 103-112](src/cv/ocr.py#L103-L112); promotion logic at [`src/pipeline.py`, lines 82-99](src/pipeline.py#L82-L99)) | DINOv2 + OCR tiebreaker | 97.2% correctness on the 5.4% of images where a registration matches | High-precision tiebreaker |

#### 2C.5 Evaluation and Error Analysis
- **Metrics and/or visual checks:** Top-1, Top-5
  ([`src/cv/train_vit.py`, lines 57-64](src/cv/train_vit.py#L57-L64)), macro
  precision/recall/F1, per-class breakdown, Grad-CAM saliency maps for 5 representative
  classes ([`scripts/generate_gradcam.py`](scripts/generate_gradcam.py),
  outputs in [`docs/gradcam/`](docs/gradcam)).
- **Final results:** Top-1 84.5%, Top-5 97.0%, macro-F1 0.84. Full metrics:
  [`models/cv/metrics.json`](models/cv/metrics.json) and
  [`models/cv/clip_baseline_metrics.json`](models/cv/clip_baseline_metrics.json).
- **Error patterns and limitations:** Within-family confusions dominate (737-300 vs 400 vs
  500; 747-100 vs 200 vs 300; 767-200 vs 300; DC-3 vs C-47). Top-5 of 97% confirms the
  family is almost always present — the residual error is variant-level discrimination at
  FGVC's resolution. The OCR tiebreaker is designed to absorb exactly this failure mode and
  reaches 97.2% correctness when a matched registration is available.

#### 2C.6 Integration with Other Block(s)
- **Inputs received from other block(s):** None — CV is the head of the pipeline.
- **Outputs provided to other block(s):** Predicted variant string (with top-5 confidences)
  forwarded to the spec lookup, which feeds the numeric and NLP blocks.

---

## 3. Deployment

- **Deployment URL:** https://huggingface.co/spaces/dubattim/aviation-intelligence-system
- **Main user flow:**
  1. Upload an aircraft photo (or pick one from the example gallery).
  2. Choose an origin and destination airport (IATA codes from OpenFlights).
  3. (Optional) toggle the OCR tiebreaker and the NLP strategy / LLM provider.
  4. Click *Analyse*.
  5. The UI displays: predicted variant + top-5 confidences, feasibility probability + verdict,
     natural-language explanation with cited Wikipedia source titles, and (if applicable) the
     OCR-detected registration.
- **Screenshot or short demo:** Full set in [`docs/screenshots/`](docs/screenshots).
  - Main UI: [`docs/screenshots/01_main_ui.png`](docs/screenshots/01_main_ui.png)
  - Example gallery: [`docs/screenshots/02_example_gallery.png`](docs/screenshots/02_example_gallery.png)
  - A380 DXB→SYD result: [`docs/screenshots/03_result_a380.png`](docs/screenshots/03_result_a380.png)
  - 777-200 with OCR: [`docs/screenshots/04_result_777_ocr.png`](docs/screenshots/04_result_777_ocr.png)

---

## 4. Execution Instructions

### Environment setup

```bash
# Requires uv (brew install uv) and Python 3.12
git clone https://github.com/TimDubath-dev/aviation-intelligence-system.git
cd aviation-intelligence-system
uv sync --python 3.12

# API keys
cp .env.example .env
# edit .env: OPENAI_API_KEY=...  (optional ANTHROPIC_API_KEY=...)
```

### Data setup

```bash
# Downloads ~2.7 GB of FGVC images on first run
uv run python -m src.cv.download_data
uv run python -m src.utils.build_specs
uv run python -m src.numeric.build_dataset
uv run python -m src.nlp.build_index
```

### Training command(s)

```bash
# Numeric models (CPU, ~1 min)
uv run python -m src.numeric.train

# CV (GPU only) — open notebooks/train_vit_colab.ipynb in Google Colab,
# set runtime to T4 GPU, paste a Hugging Face write token, run all cells (~30 min).
# The resulting checkpoint is published to dubattim/aviation-intelligence-vit-fgvc.
```

### Inference / run command(s)

```bash
# Launch the Gradio app locally
PYTHONPATH=. uv run python app/app.py

# Smoke tests
uv run python scripts/smoke_pipeline.py        # numeric + NLP only
uv run python scripts/smoke_full_pipeline.py   # full photo → answer
```

### Reproducibility notes

| Step | Determinism |
| --- | --- |
| FGVC download | Identical bytes (Oxford VGG mirror) |
| Curated specs | Versioned CSV in [`data/raw/`](data/raw) |
| Route-dataset synthesis | `numpy.random.default_rng(42)` ([`src/numeric/build_dataset.py`, line 39](src/numeric/build_dataset.py#L39)) |
| Numeric model training | `random_state=42` ([`src/numeric/train.py`, line 43](src/numeric/train.py#L43)), 5-fold stratified CV ([`src/numeric/train.py`, lines 91-101](src/numeric/train.py#L91-L101)) |
| ViT training | Same hyperparameters in `train_vit.py` and the Colab notebook |
| FAISS index | Deterministic embeddings (MiniLM, L2-normalized) |

Pinned environment: [`pyproject.toml`](pyproject.toml) + [`uv.lock`](uv.lock) (training),
[`requirements.txt`](requirements.txt) (HF Spaces inference).

---

## 5. Optional Bonus Evidence

- [x] **Third selected block implemented with strong quality** — all three blocks (CV, NLP,
      numeric) are integrated end-to-end and quantitatively evaluated. See
      [§4 of `DOCUMENTATION.md`](DOCUMENTATION.md#4-evaluation--analysis).
- [x] **More than two data sources used with clear added value** — 6 distinct sources
      (FGVC-Aircraft, Wikimedia Commons, curated specs, OpenFlights, Wikipedia corpus,
      OpenSky). Each is justified in
      [§2.1 of `DOCUMENTATION.md`](DOCUMENTATION.md#21-data-sources).
- [x] **A core section is done exceptionally well** — CV combines a fine-tuned DINOv2 backbone
      ([`notebooks/train_vit_colab.ipynb`](notebooks/train_vit_colab.ipynb)),
      Wikimedia data augmentation
      ([`src/cv/scrape_extra_images.py`](src/cv/scrape_extra_images.py)), a CLIP zero-shot
      baseline (+51.7pp Top-1 gap, [`src/cv/clip_baseline.py`, lines 31-41](src/cv/clip_baseline.py#L31-L41)),
      Grad-CAM saliency maps ([`docs/gradcam/`](docs/gradcam)), and an OCR-based tiebreaker
      against the model's primary failure mode
      ([`src/cv/ocr.py`, lines 28-77](src/cv/ocr.py#L28-L77);
      promotion at [`src/pipeline.py`, lines 82-99](src/pipeline.py#L82-L99); 97.2% tiebreaker
      correctness).
- [x] **Extended evaluation** — ablation studies removing each block individually, a
      hallucination probe on non-existent aircraft, cross-model comparison (GPT-4o-mini vs
      Claude Haiku), and per-segment metrics on the difficult range-margin band. See
      [§4.4 of `DOCUMENTATION.md`](DOCUMENTATION.md#44-ablation-studies).
- [x] **Ethics, bias, or fairness analysis** — explicit disclaimer in UI and README; honest
      treatment of synthetic-label limitations
      ([§4.2 of `DOCUMENTATION.md`](DOCUMENTATION.md#42-ml-on-numeric-data)); scope clearly
      restricted to an educational tool, not a flight-planning system.

---

## License & Acknowledgements

MIT — see [`LICENSE`](LICENSE).

- ZHAW *AI Applications* (FS26) — Jasmin Heierli & Benjamin Kühnis
- FGVC-Aircraft — Oxford Visual Geometry Group
- OpenFlights airports — `openflights.org` (CC-BY-SA)
- OpenSky aircraft database — `opensky-network.org` (ODbL)
- Wikipedia — aircraft and airport corpus (CC-BY-SA)

## Disclaimer

This is an **educational project**. Route-feasibility predictions consider only range, ETOPS,
and synthetic headwind/payload effects — they ignore weather, runway length, payload limits,
regulatory clearances, fuel pricing, and ATC routing. **Do not use for actual flight planning.**
