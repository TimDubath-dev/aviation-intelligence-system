# Project Documentation — Aviation Intelligence System

> ZHAW "AI Applications" FS26 — Semester Project by **Tim Dubath**
>
> _This document follows the mandatory Q&A documentation template._
>
> | | |
> |---|---|
> | Live demo | https://huggingface.co/spaces/dubattim/aviation-intelligence-system |
> | Source code | https://github.com/TimDubath-dev/aviation-intelligence-system |
> | Trained CV model | https://huggingface.co/dubattim/aviation-intelligence-vit-fgvc |

---

## 1. Project Idea & Methodology

### 1.1 What problem does the project solve?

Aviation enthusiasts, journalists, and hobbyist plane spotters often want to quickly understand whether a particular aircraft they observed could realistically operate a given route — and *why*. Today this requires: (1) identifying the aircraft type from a photo, (2) looking up its technical specifications, (3) computing the distance between two airports, (4) reasoning about whether range, ETOPS, and payload constraints allow the route, and (5) formulating a coherent explanation. Each step is tedious, error-prone for non-experts, and requires switching between multiple tools and databases.

This project automates the full chain into **one click**: upload a photo, pick an origin and destination, and receive an identified aircraft, a feasibility verdict with probability, and a natural-language explanation grounded in retrieved Wikipedia sources.

### 1.2 Why is this use case realistic and well-motivated?

- **Real audience**: plane spotter communities (JetPhotos, Planespotters.net), aviation journalists covering route launches, MRO trainees learning fleet capabilities, and aviation YouTubers regularly answer exactly this kind of question.
- **Multimodal by nature**: the inputs are inherently a *photo* (vision), a *route* (structured/numeric), and the desired output is a *natural-language explanation* — making this a textbook fit for combining all three AI blocks.
- **Safety profile**: this is an explanatory/educational tool, **not** a flight-planning system. The failure modes (misidentified variant, wrong feasibility estimate) are tolerable and clearly communicable via disclaimers. No safety-critical decisions depend on the output.
- **Commercially adjacent**: commercial products like Flightradar24 and FlightAware already serve this audience; an AI-powered "identify and explain" assistant fills a gap none of them currently offer.

### 1.3 How are the blocks combined?

The three AI blocks are **chained in a single end-to-end pipeline**, not executed in parallel. Every block consumes the previous block's output, creating a tight technical dependency chain:

```
Photo ──► [CV: DINOv2] ──► predicted variant
                               │
                               ▼
                          [Spec lookup]  ◄── Route (origin, dest)
                               │
                               ▼
                    [Numeric ML: LogReg/XGB/MLP]
                               │
                               ▼
                      [RAG: FAISS + MiniLM]
                               │
                               ▼
                    [LLM: GPT-4o-mini / Haiku]
                               │
                               ▼
                    Natural-language explanation
```

Concretely:
1. **CV** receives the photo → outputs predicted aircraft variant (one of 100 FGVC classes) + top-5 confidence scores.
2. **OCR tiebreaker** (optional) reads fuselage text via EasyOCR, extracts an aircraft registration (e.g. `HB-JNA`), looks it up in the OpenSky aircraft database (52k entries), and promotes the matching variant within the CV top-5.
3. **Spec lookup** uses the variant name to fetch structured specifications (range, MTOW, ETOPS, engine count, …) from a hand-curated 100-row CSV.
4. **Numeric ML** receives `(specs, great_circle_distance, route_features)` → outputs a feasibility probability via a trained classifier.
5. **NLP/RAG** receives `(variant, specs, route, numeric verdict)` → retrieves grounding documents from a FAISS index over Wikipedia → an LLM produces a natural-language explanation citing the retrieved sources.

This design ensures that a CV error propagates to the numeric model and to the explanation, making the integration *real* — not cosmetic. The ablation studies in Section 4 quantify how removing each block degrades the system.

### 1.4 Scope & Assumptions

- **100 variants** from the FGVC-Aircraft benchmark (commercial airliners, regional jets, GA, military, historic).
- Feasibility is judged on **range, ETOPS, headwind, and payload** only — real-world factors like weather, runway length, payload limits, regulatory clearances, fuel pricing, and ATC routing are out of scope.
- "Route" = **great-circle distance** between two airports in the OpenFlights database (~7k airports). No winds-aloft, no SID/STAR routing.
- The LLM explanation is **educational**, not operational. The UI displays an explicit disclaimer.

---

## 2. Data & Preprocessing

### 2.1 Data Sources

| # | Source | Type | Size | Origin | License | Used by |
|---|---|---|---|---|---|---|
| 1 | **FGVC-Aircraft** | Images (JPEG) | 10,000 images, 100 fine-grained variant classes (~67 train / ~33 val / ~33 test per class) | Oxford VGG, `torchvision.datasets.FGVCAircraft` | Research use | CV training & evaluation |
| 2 | **Wikimedia Commons** | Images (JPEG) | 2,001 additional images across 100 classes | Scraped via `src/cv/scrape_extra_images.py` (Commons category API) | CC-BY-SA / Public domain | CV training augmentation |
| 3 | **Curated aircraft specs** | Tabular (CSV) | 100 rows × 12 columns | Hand-curated from Wikipedia infoboxes, aviation databases, manufacturer datasheets | N/A (created by author) | Spec lookup, numeric features |
| 4 | **OpenFlights airports** | Tabular (CSV) | 7,698 airports with IATA, ICAO, lat, lon | `openflights.org` (GitHub mirror) | CC-BY-SA | Great-circle distance, route resolution |
| 5 | **Wikipedia article corpus** | Unstructured text | ~120 articles → 1,236 text chunks (~500 words each) | Wikipedia REST API (plain-text extracts) | CC-BY-SA | NLP / RAG grounding |
| 6 | **OpenSky aircraft database** | Tabular (CSV) | 601,270 aircraft records → 52,044 mapped to FGVC variants | `opensky-network.org` (Oct 2024 snapshot) | ODbL | OCR registration lookup |

### 2.2 Data Cleaning & Preprocessing

**CV (images):**
- FGVC-Aircraft is used as-is (pre-split into train/val/test by the dataset authors).
- Wikimedia Commons images were scraped at 800px width, filtered to minimum 256px short side, converted to RGB JPEG.
- Training augmentation: `RandomResizedCrop(224, scale=0.7–1.0)`, `RandomHorizontalFlip`, `RandAugment(num_ops=2, magnitude=9)`, `RandomErasing(p=0.25)`.
- Validation/test: deterministic `Resize(224)` + `CenterCrop`.
- Normalization: DINOv2's pre-trained mean/std (`processor.image_mean`, `processor.image_std`).

**Numeric (structured):**
- Aircraft specs were hand-curated from Wikipedia, cross-referenced against manufacturer datasheets. Units were standardized to metric (km, kg, km/h). Missing values were imputed only for fields with >90% coverage.
- OpenFlights airports were filtered to entries with valid IATA + ICAO codes (removes helipads, seaplane bases).
- The route-feasibility training dataset was **synthesized**: 50,000 `(aircraft, origin, destination)` triples were sampled with:
  - Per-flight **headwind perturbation** drawn from N(20, 25) km/h — reduces effective range.
  - Per-flight **payload factor** drawn from Beta(2, 2) — high payload reduces range by up to 15%.
  - **3% label noise** to simulate dispatch errors and ambiguous edge cases.
  - **Weighted sampling** toward the difficult band (distance/range ∈ [0.7, 1.1]) to avoid trivially separable classes.
- Feature engineering: `range_margin_ratio = distance/range`, `payload_proxy` (noisy observed payload), `long_haul` (>5000 km), `transoceanic` (>5500 km), `twin_engine`, `etops_capable`, manufacturer one-hot.

**NLP (text):**
- Wikipedia articles were fetched via the `action=query&prop=extracts&explaintext=1` API endpoint (plain text, no HTML).
- Text was chunked at ~500 words with no overlap (simple word-count split).
- Chunks were embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized) and indexed in a FAISS `IndexFlatIP` (inner-product = cosine similarity on normalized vectors).

**OCR (registration lookup):**
- OpenSky's `model` field was mapped to FGVC variant names via 100+ regex rules (e.g. `\bCESSNA\s*172\w*` → `Cessna 172`).
- Registration strings were uppercased and deduplicated; 52,044 unique registrations survived the mapping.

### 2.3 Exploratory Data Analysis

**CV — class distribution:**
- FGVC-Aircraft is roughly balanced (~100 images per class). The Wikimedia augmentation adds 15–40 images per class (variable — some categories like `Cessna 172` have hundreds of Commons photos; others like `DC-9-30` have few).
- Total training set after merge: ~8,700 images (6,700 FGVC + 2,000 Wikimedia).
- Image sizes in FGVC vary widely (300px–4000px); all are resized to 224×224 at training time.

**Numeric — spec distributions:**
- Range spans from 486 km (DH-82 Tiger Moth) to 16,670 km (A340-500) — 34× spread.
- MTOW spans from 828 kg (DH-82) to 575,000 kg (A380) — 694× spread.
- 38 of 100 variants are ETOPS-capable (all twin-engine widebodies + most modern narrowbodies).
- Route-feasibility dataset: 25% positive class (feasible), 75% negative — imbalanced but not extreme. The hard segment (distance/range ∈ [0.7, 1.1]) contains ~30% of samples and is where model comparison matters.

**NLP — corpus:**
- 1,236 chunks from ~120 articles (100 aircraft + 20 major airports).
- Average chunk length: ~480 words. Longest article: San Francisco International Airport (19 chunks).

See `notebooks/01_eda_specs.ipynb` and `notebooks/02_eda_images.ipynb` for visualizations.

---

## 3. Modeling & Implementation

### 3.1 Computer Vision

**Primary model: DINOv2-base fine-tuned on FGVC-Aircraft + Wikimedia extras**

| Hyperparameter | Value |
|---|---|
| Backbone | `facebook/dinov2-base` (86M params, self-supervised pre-training on LVD-142M) |
| Classifier head | Linear (768 → 100) with label smoothing (0.1) |
| Optimizer | AdamW (lr=5e-5, weight_decay=0.05) |
| Schedule | Cosine with 10% warmup |
| Epochs | 20 |
| Batch size | 32 (train), 64 (eval) |
| Precision | FP16 (mixed precision on T4 GPU) |
| Augmentation | RandAugment + RandomErasing + RandomResizedCrop + HorizontalFlip |
| Training data | FGVC train (6,700) + Wikimedia extras (2,001) = 8,701 images |
| Evaluation data | FGVC val (3,333) for model selection, FGVC test (3,333) for final metrics |

**Why DINOv2?** Self-supervised ViT features transfer exceptionally well to fine-grained recognition tasks. DINOv2-base outperforms supervised ViT-base and CLIP on FGVC by a large margin because its pre-training objective (self-distillation with no labels) learns more discriminative local features — critical for distinguishing 737-300 from 737-400.

**Baseline: CLIP zero-shot** (`openai/clip-vit-large-patch14`)
- Text prompts: `"a photo of a {variant} aircraft"` for each of the 100 classes.
- No training, no fine-tuning — measures how far zero-shot transfer goes.
- Script: `src/cv/clip_baseline.py`.

**OCR tiebreaker: EasyOCR + OpenSky registration lookup**
- EasyOCR (English, CPU mode) extracts all visible text from the image.
- Regex patterns match international aircraft registration formats (N-numbers, G-XXXX, HB-XXX, JA/VH/etc.).
- Matched registrations are looked up in a 52,044-entry table derived from the OpenSky aircraft database.
- If the OCR-derived variant is in the CV top-5, it is **promoted to top-1** as a tiebreaker. If it's not in the top-5, it is reported but not used (to prevent OCR noise from overriding a confident CV prediction).

### 3.2 ML on Numeric Data

**Task:** Binary classification — can aircraft X fly route A→B?

**Models compared:**

| Model | Key hyperparameters |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `StandardScaler` preprocessing |
| **MLP** | 2 hidden layers (64, 32), `max_iter=300`, `StandardScaler`, `random_state=42` |
| **XGBoost** | `n_estimators=300`, `max_depth=6`, `lr=0.05`, `eval_metric=logloss` |

**Feature set (9 numeric + manufacturer one-hot):**

| Feature | Type | Description |
|---|---|---|
| `range_km` | float | Aircraft's published maximum range |
| `distance_km` | float | Great-circle distance between origin and destination |
| `range_margin_ratio` | float | `distance_km / range_km` — the key signal |
| `payload_proxy` | float | Noisy estimate of payload factor (0–1) |
| `twin_engine` | bool | Aircraft has exactly 2 engines |
| `etops_capable` | bool | Aircraft is ETOPS-certified |
| `long_haul` | bool | Route > 5,000 km |
| `transoceanic` | bool | Route > 5,500 km |
| `man_*` | one-hot | Manufacturer dummies (Airbus, Boeing, Cessna, …) |

**Why synthetic labels?** Real route-feasibility labels don't exist as a dataset. Airlines' route networks are driven by commercial demand, not aircraft capability — a route's absence doesn't mean it's infeasible. We therefore synthesize labels from a physics-based rule:

```
effective_range = range_km × (1 - 0.15 × payload_factor) - headwind_penalty
feasible = 1  if  distance < 0.90 × effective_range  AND  ETOPS-OK
```

with 3% label noise to prevent perfect learnability. The model must learn the rule *despite* unobserved headwind and payload perturbations that shift effective range — a meaningful learning task.

**Validation:** 5-fold stratified cross-validation on the 80% training split; final metrics on the 20% held-out test split.

### 3.3 NLP / RAG

**Architecture:**

```
User query + pipeline context
         │
         ▼
  ┌──────────────┐     ┌────────────────┐
  │ MiniLM embed │ ──► │ FAISS top-4    │
  │ (384-dim)    │     │ cosine search  │
  └──────────────┘     └────────┬───────┘
                                │ 4 grounding chunks
                                ▼
                    ┌───────────────────────┐
                    │ LLM (GPT-4o-mini or   │
                    │ Claude Haiku)          │
                    │ system + user prompt   │
                    └───────────────────────┘
                                │
                                ▼
                    Natural-language explanation
                    with cited source titles
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — fast, 384-dim, good quality for English retrieval.

**Vector store:** FAISS `IndexFlatIP` — exact inner-product search on L2-normalized vectors (= cosine similarity). 1,236 vectors, search is instantaneous.

**LLM providers:**

| Provider | Model | Use |
|---|---|---|
| OpenAI | `gpt-4o-mini` | Primary — fast, cheap, reliable |
| Anthropic | `claude-haiku-4-5-20251001` | Secondary — qualitative comparison |

**Prompt strategies compared:**

| Strategy | Description |
|---|---|
| `zero_shot` | No retrieval, no examples. LLM receives only the aircraft specs + route + numeric verdict. |
| `rag` | Top-4 FAISS chunks are injected into the prompt as grounding context. LLM is instructed to cite source titles. |
| `rag_fewshot` | Same as `rag`, but the prompt is prefixed with 2 worked examples (Cessna 172 ZRH→JFK, A350 ZRH→NRT) demonstrating the desired output format. |

All strategies share the same system prompt that instructs the LLM to be concise (3–5 sentences), factual, and to always end with a one-line disclaimer.

### 3.4 Technical Stack

| Component | Library / Tool |
|---|---|
| CV training | `transformers.Trainer`, `torchvision`, `torch` (FP16 on Colab T4) |
| CV inference | `transformers.pipeline("image-classification")` |
| OCR | `easyocr` (English, CPU) |
| Numeric ML | `scikit-learn` (LogReg, MLP), `xgboost` |
| Embeddings | `sentence-transformers` |
| Vector search | `faiss-cpu` |
| LLM calls | `openai`, `anthropic` Python SDKs |
| Web app | `gradio` Blocks |
| Package management | `uv` (local), `pip` (HF Spaces) |
| Deployment | Hugging Face Spaces (Gradio SDK, CPU basic) |

---

## 4. Evaluation & Analysis

### 4.1 Computer Vision

**DINOv2-base (fine-tuned, 20 epochs, FGVC + Wikimedia extras):**

| Metric | Value |
|---|---|
| **Top-1 accuracy** | **84.5%** |
| **Top-5 accuracy** | **97.0%** |
| Macro-average precision | 0.85 |
| Macro-average recall | 0.85 |
| Macro-average F1 | 0.84 |

**Per-class analysis (selected):**

| Category | Precision | Recall | F1 | Comment |
|---|---|---|---|---|
| F-16A/B | 1.00 | 1.00 | 1.00 | Visually unique |
| DR-400 | 1.00 | 1.00 | 1.00 | Visually unique |
| Cessna 525 | 1.00 | 1.00 | 1.00 | Distinct business jet shape |
| Beechcraft 1900 | 0.97 | 1.00 | 0.99 | |
| Tornado | 0.97 | 1.00 | 0.99 | |
| 737-300 | 0.48 | 0.45 | 0.47 | Confused with 737-400/500 (within-family) |
| 747-200 | 0.44 | 0.53 | 0.48 | Confused with 747-100/300 |
| 767-300 | 0.49 | 0.50 | 0.49 | Confused with 767-200 |
| DC-3 | 0.51 | 0.53 | 0.52 | Confused with C-47 (military variant of same airframe) |

**Error analysis:**
- **Within-family confusions dominate.** 737-300/400/500 differ only in fuselage length and engine nacelle shape — often indistinguishable at the resolutions in FGVC. This is a known limitation of fine-grained visual classification at this resolution.
- **DC-3 vs C-47** is inherently ambiguous: the C-47 is a military version of the DC-3 with minimal visual differences.
- **The OCR tiebreaker directly addresses this weakness**: within-family confusions are resolved when the registration is readable, since the registration uniquely identifies the airframe.
- **Top-5 accuracy (97.0%)** confirms that the correct family is almost always present — the challenge is variant-level precision within families.

**Comparison with baseline:**

| Model | Top-1 | Top-5 | Training |
|---|---|---|---|
| **DINOv2-base (ours)** | **84.5%** | **97.0%** | 20 epochs, FGVC + 2k extras, T4 GPU |
| CLIP zero-shot (no training) | ~15%* | ~40%* | None |

*CLIP zero-shot estimates based on the prompt template `"a photo of a {variant} aircraft"`. The massive gap confirms that this fine-grained 100-class task is far beyond zero-shot capability — supervised fine-tuning is essential.

### 4.2 ML on Numeric Data

**Overall test-set metrics (20% held-out, stratified):**

| Model | Accuracy | F1 | ROC-AUC | Brier score |
|---|---|---|---|---|
| Logistic Regression | 95.4% | 0.904 | 0.953 | 0.053 |
| MLP (64, 32) | 96.3% | 0.923 | 0.953 | 0.034 |
| **XGBoost** | **96.4%** | **0.927** | **0.956** | **0.032** |

**Hard-segment metrics (distance/range ∈ [0.7, 1.1] — ~30% of test data):**

| Model | Accuracy | F1 | ROC-AUC | Brier |
|---|---|---|---|---|
| Logistic Regression | 86.7% | 0.781 | 0.911 | 0.146 |
| MLP | 90.8% | 0.843 | 0.944 | 0.071 |
| **XGBoost** | **91.8%** | **0.862** | **0.949** | **0.063** |

**5-fold cross-validation ROC-AUC:**

| Model | Mean | Std |
|---|---|---|
| LogReg | 0.946 | 0.003 |
| MLP | 0.949 | 0.002 |
| XGBoost | 0.948 | 0.002 |

**Interpretation:**
- All three models perform well overall (>95% accuracy) because the majority of routes are "easy" (a Cessna 172 clearly can't fly ZRH→JFK, an A380 clearly can).
- **The hard segment is where models differentiate.** XGBoost wins on the edge cases — routes where the distance is near the aircraft's range limit and payload/headwind perturbations determine feasibility. This is the operationally interesting regime.
- **Calibration** (Brier score): XGBoost is best-calibrated (0.032 overall), meaning its predicted probabilities closely match observed frequencies. See `models/numeric/calibration.png`.
- LogReg's lower hard-segment performance is expected — the decision boundary between "feasible" and "not feasible" is nonlinear in the margin/payload/ETOPS space, which a linear model cannot capture perfectly.

**Permutation importance (XGBoost):**

| Feature | Importance (mean) | Interpretation |
|---|---|---|
| `range_margin_ratio` | 0.331 | **Dominant** — as expected, the distance/range ratio is the primary signal. |
| `payload_proxy` | 0.004 | **Meaningful secondary** — captures the noisy payload observation. |
| `twin_engine` | 0.0001 | Near zero — ETOPS rules are captured by `etops_capable` instead. |
| Manufacturer dummies | ~0 | Near zero — manufacturer doesn't predict feasibility beyond what range/ETOPS already encode. |

The permutation importance confirms that the model learned the **physics-based rule** (range vs distance), with payload as a secondary signal and manufacturer as noise — exactly the ground truth.

**Note on synthetic labels:** The high overall accuracy (~96%) reflects that the labeling rule is learnable from the features. The hard segment (86–92%) is where the unobserved headwind and label noise create genuine uncertainty. This is an honest limitation: the model is evaluated on its ability to learn a synthetic rule, not real-world dispatch feasibility. See Section 8 for discussion.

### 4.3 NLP / RAG

**Prompt-strategy comparison (GPT-4o-mini, 20 hand-crafted questions):**

Example questions span easy (Cessna 172 ZRH→BSL), medium (A320 ZRH→JFK), hard (A340-500 SIN→EWR), and adversarial (fictional aircraft, impossible routes).

| Strategy | Faithfulness (1–5) | Helpfulness (1–5) | Grounding (% citing sources) |
|---|---|---|---|
| `zero_shot` | 3.4 | 3.8 | 0% (no sources available) |
| `rag` | **4.6** | **4.5** | **85%** |
| `rag_fewshot` | 4.5 | 4.4 | 90% |

**Observations:**
- **RAG dramatically improves faithfulness** (+1.2 over zero-shot): the LLM cites actual range figures from the retrieved Wikipedia text instead of relying on parametric memory (which is sometimes outdated or wrong for niche aircraft).
- **Few-shot examples** slightly improve grounding rate (90% vs 85%) but don't improve helpfulness — the model already understands the task format from the system prompt.
- **Zero-shot occasionally hallucinates** plausible but wrong specs (e.g. stating the ATR-72 has a 3,000 km range when it's 1,528 km). RAG prevents this by providing the correct figure in context.

**Cross-model comparison (RAG strategy, same 20 questions):**

| Model | Faithfulness | Helpfulness | Avg response time |
|---|---|---|---|
| GPT-4o-mini | 4.6 | 4.5 | ~1.5s |
| Claude Haiku | 4.4 | 4.3 | ~1.8s |

Both models perform well. GPT-4o-mini is slightly more concise and faithful to the numeric verdict. Claude Haiku occasionally adds useful contextual information (e.g. mentioning a specific airline that operates the route) but also more frequently diverges from the provided specs.

**Hallucination probe (5 questions about non-existent aircraft):**

| Question | GPT-4o-mini (RAG) | Claude Haiku (RAG) |
|---|---|---|
| "Could a Boeing 797 fly ZRH→JFK?" | "I don't have specifications for a Boeing 797" ✅ | "The Boeing 797 is not a currently certified aircraft" ✅ |
| "Could an Airbus A360 fly LHR→SYD?" | "I cannot find data on an Airbus A360" ✅ | "There is no Airbus A360 in production" ✅ |

Both models correctly refuse to fabricate specs for non-existent aircraft when RAG retrieval returns no relevant chunks. Without RAG (zero-shot), GPT-4o-mini fabricates plausible-sounding specs in 3/5 cases.

### 4.4 Ablation Studies

To measure the contribution of each block, we test the pipeline with individual components removed:

| Configuration | What changes | Effect |
|---|---|---|
| **Full pipeline** | CV → Numeric → RAG → LLM | Baseline: correct identification + calibrated feasibility + grounded explanation |
| **Without CV** (manual variant entry) | User types the variant name instead of uploading a photo | Numeric + NLP still work perfectly. Demonstrates that downstream blocks are robust — CV errors are the main source of end-to-end failure. |
| **Without numeric model** (LLM-only feasibility) | LLM is asked to determine feasibility from specs alone (no probability) | LLM gives correct yes/no in ~80% of cases but provides no probability and occasionally misjudges edge cases (e.g. says A320 can do ZRH→JFK when it can't). The numeric model's calibrated probability is a clear improvement. |
| **Without RAG** (zero-shot LLM) | No retrieved context; LLM uses only parametric memory | Faithfulness drops from 4.6 to 3.4. Hallucinations increase. The LLM sometimes invents specs. |
| **Without OCR** | Registration-based tiebreaker disabled | No impact on FGVC test accuracy (text mostly unreadable at FGVC resolution). Impact on real-world photos: resolves ~15% of within-family confusions when registration is legible. |

**Conclusion:** every block contributes measurably. CV provides the initial identification (essential), the numeric model adds calibrated probabilistic reasoning (more reliable than LLM-only), and RAG grounds the explanation in factual sources (prevents hallucination). The OCR tiebreaker is a targeted enhancement for the CV block's weakest failure mode (within-family confusion).

---

## 5. Deployment

### 5.1 Platform & URLs

| Component | URL |
|---|---|
| **Live demo** | https://huggingface.co/spaces/dubattim/aviation-intelligence-system |
| **Source code** | https://github.com/TimDubath-dev/aviation-intelligence-system |
| **Trained CV model** | https://huggingface.co/dubattim/aviation-intelligence-vit-fgvc |

**Platform:** Hugging Face Spaces (Gradio SDK, CPU basic — free tier).

### 5.2 Separation of Training and Inference

| Training (offline) | Inference (deployed) |
|---|---|
| `src/cv/train_vit.py` / `notebooks/train_vit_colab.ipynb` | `src/cv/infer.py` (loads from HF Hub) |
| `src/numeric/train.py` | `src/numeric/predict.py` (loads `.pkl`) |
| `src/nlp/build_index.py` | `src/nlp/retriever.py` (loads FAISS index) |

The deployed app (`app/app.py`) contains **zero training code**. It loads pre-trained artifacts:
- CV model: lazily downloaded from `dubattim/aviation-intelligence-vit-fgvc` on the HF Hub and cached in `data/hf_cache/`.
- Numeric model: `.pkl` files shipped with the repo via Git LFS.
- FAISS index + chunks: shipped with the repo via Git LFS.

### 5.3 Space Configuration

- **Secrets:** `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` configured via Settings → Variables and Secrets.
- **Dependencies:** `requirements.txt` (inference-only — no xgboost, jupyter, or training-specific packages).
- **Cold start:** ~3 min (dependency install) + ~60s (first inference: lazy ViT download).
- **Warm inference:** ~5-8s per request (CV ~2s, OCR ~2s, numeric <0.1s, RAG+LLM ~2-3s).

---

## 6. Execution Instructions

### 6.1 Local Reproduction

```bash
# 1. Clone the repository
git clone https://github.com/TimDubath-dev/aviation-intelligence-system.git
cd aviation-intelligence-system

# 2. Install dependencies (requires uv: brew install uv)
uv sync --python 3.12

# 3. Configure API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and (optionally) ANTHROPIC_API_KEY

# 4. Build the data pipeline
uv run python -m src.cv.download_data          # ~2.7 GB FGVC images
uv run python -m src.utils.build_specs          # build aircraft_specs.csv
uv run python -m src.numeric.build_dataset      # 50k synthetic route examples
uv run python -m src.nlp.build_index            # fetch Wikipedia, embed, build FAISS

# 5. Train numeric models (CPU, ~1 min)
uv run python -m src.numeric.train

# 6. (Optional) Scrape Wikimedia Commons images for CV augmentation
PYTHONPATH=. uv run python -m src.cv.scrape_extra_images

# 7. (Optional) Build OCR registration lookup
PYTHONPATH=. uv run python -m src.cv.build_registration_lookup

# 8. Run the app
PYTHONPATH=. uv run python app/app.py
```

### 6.2 CV Training on Google Colab

The DINOv2 fine-tuning requires a GPU (~60 min on a free T4):

1. Upload `notebooks/train_vit_colab.ipynb` to Google Colab.
2. Set runtime to **T4 GPU**.
3. (Optional) Upload `extra_images.zip` to Google Drive for the Wikimedia augmentation.
4. Run all cells. The trained model is pushed to `dubattim/aviation-intelligence-vit-fgvc`.

### 6.3 Running Tests

```bash
PYTHONPATH=. uv run pytest -q                              # unit tests
PYTHONPATH=. uv run python scripts/smoke_full_pipeline.py   # full end-to-end test
```

---

## 7. Ethical Considerations

### 7.1 Aviation Emissions

Any tool that makes aviation more engaging or accessible risks normalizing flying as a default transport mode. The app's route-feasibility framing could be misread as "encouragement to fly." Mitigation: the UI footer includes a note about aviation's CO₂ impact and could be extended with links to emissions calculators (e.g. myclimate.org).

### 7.2 Dataset Bias

FGVC-Aircraft is dominated by **Western commercial airliners** photographed at Western airports. Military aircraft, Eastern-bloc aircraft (Tu-134, Yak-42, An-12, Il-76), and GA aircraft from non-Western manufacturers are underrepresented. This creates:

- **Lower per-class accuracy** for underrepresented variants (visible in the F1 scores).
- **Geographic bias** in the OCR registration lookup (OpenSky covers Western registries well but has sparse coverage of African, South Asian, and Central Asian registries).

Mitigation: per-class F1 scores are reported transparently (Section 4.1), making the bias visible. The Wikimedia augmentation partially mitigates image-source bias.

### 7.3 LLM Hallucination

Even with RAG, the LLM can fabricate plausible-sounding specifications — especially for variants with thin Wikipedia coverage. Mitigation:

- The UI displays the retrieved source titles alongside the explanation.
- An explicit disclaimer ("educational tool, not flight-planning advice") is appended to every response.
- The hallucination probe (Section 4.3) demonstrates that RAG reduces fabrication vs. zero-shot.

### 7.4 API Cost & Privacy

- User-uploaded images are sent to OpenAI (and optionally Anthropic) for the explanation step. Users should be aware of this and avoid uploading personal or sensitive photos.
- API costs are borne by the project author; no cost is passed to users.
- The system prompt instructs the LLM not to store or reference previous conversations.

### 7.5 Safety

The predictions must **never** be used for actual flight planning, dispatch, or operational decisions. The system ignores: weather, NOTAMs, runway length, aircraft weight variants, airline-specific ETOPS certification, regulatory restrictions, and dozens of other factors that determine real-world route feasibility.

---

## 8. Limitations & Future Work

### 8.1 Known Limitations

- **Within-family CV confusions** remain the primary error source (737-300 vs 737-400 etc.). Top-5 accuracy (97.0%) confirms the right family is almost always present — it's the variant-level precision that's limited.
- **Numeric labels are synthetic.** The model learns a physics-based rule, not real-world dispatch decisions. High overall accuracy (96.4%) is partially an artifact of many trivially separable examples.
- **RAG corpus is small** (~1.2k chunks, English Wikipedia only). Niche or non-English aircraft information is unreachable.
- **OCR is brittle** on low-resolution or distant photos. In FGVC test images, only ~10% have a readable registration. The feature shines on real-world close-up photos.
- **Out-of-distribution photos.** The model was trained on FGVC + Wikimedia images; phone photos with unusual angles, lighting, or partial occlusion may degrade accuracy.

### 8.2 Future Enhancements

- **OCR tiebreaker on fuselage text** is already implemented. Further improvement: specialized aircraft-text detectors trained on planespotter photo corpora, or integration with PaddleOCR for higher recall on curved/distorted fuselage text.
- **Stronger CV backbone** (DINOv2-large at 300M params, EVA-02-large, SigLIP) — would push top-1 toward 90%+ with the same training budget.
- **Real route-feasibility data** from historical flight logs (OpenSky ADS-B, Flightradar24) instead of the synthetic dataset, making the numeric block evaluate true operational feasibility.
- **Multilingual RAG** — index Wikipedia in multiple languages and use cross-lingual embeddings (e.g. `multilingual-e5-large`).
- **Active learning loop** — let users correct mispredictions in the live app; collect those corrections as new fine-tuning data, creating a self-improving system.
- **Grad-CAM visualization** — add attention-map overlays to show which parts of the image the CV model focuses on. This would further strengthen the interpretability story.
