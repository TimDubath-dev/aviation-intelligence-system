# Project Documentation — Aviation Intelligence System

> ZHAW "AI Applications" FS26 — Semester Project by Tim Dubath
> _This document follows the mandatory Q&A documentation template._

---

## 1. Project Idea & Methodology

### What problem does the project solve?
Aviation enthusiasts, journalists, and hobbyist plane spotters often want to quickly understand whether a particular aircraft they observed could realistically operate a given route — and *why*. Looking up specifications, computing distances, and reasoning about feasibility is tedious and error-prone for non-experts. This project automates the full chain: from a single photo and a route, the system identifies the aircraft, decides whether it could fly the route, and explains the answer in plain language with citations.

### Why is this use case realistic and well-motivated?
- **Real audience**: plane spotter communities (e.g. JetPhotos, Planespotters.net), aviation journalists, MRO trainees, and aviation YouTubers regularly answer exactly this kind of question.
- **Multimodal by nature**: the inputs are inherently a *photo* (vision), a *route* (structured/numeric), and the output users want is a *natural-language explanation* — making it a textbook fit for combining all three AI blocks.
- **Safe**: this is an explanatory/educational tool, **not** a flight-planning system, so the failure modes are tolerable and clearly communicable.

### How are the blocks combined?
The blocks are chained in a single end-to-end pipeline. Every block consumes the previous block's output:

1. **CV** receives the photo → outputs predicted aircraft variant + confidence.
2. **Spec lookup** uses the variant to fetch structured specs (range, MTOW, ETOPS, …).
3. **Numeric ML** receives `(specs, great_circle_distance, route_features)` → outputs feasibility probability.
4. **NLP/RAG** receives `(variant, specs, route, numeric verdict)` → retrieves grounding documents from Wikipedia and produces a natural-language explanation citing sources.

### Scope & Assumptions
- Limited to the **100 variants** present in FGVC-Aircraft.
- Feasibility is judged on **range and ETOPS** only — payload, weather, runway length, and regulatory factors are out of scope (and explicitly stated in the UI as a disclaimer).
- "Route" = great-circle distance between two airports in OpenFlights; no winds aloft, no SID/STAR routing.

---

## 2. Data & Preprocessing

_To be filled during M1–M3._

### Data Sources
| # | Source | Type | Size | Used by |
|---|---|---|---|---|
| 1 | FGVC-Aircraft | Images | ~10k images, 100 classes | CV |
| 2 | Wikipedia infoboxes (scraped) | Tabular text → CSV | 100 rows × ~10 cols | Numeric, NLP |
| 3 | OpenFlights airports | Tabular | ~7k airports | Numeric |
| 4 | Wikipedia article text | Unstructured text | ~100 articles → ~2k chunks | NLP / RAG |

### Cleaning & Preprocessing
_(filled during implementation)_

### EDA
_(filled during implementation — see `notebooks/01_eda_specs.ipynb` and `notebooks/02_eda_images.ipynb`)_

---

## 3. Modeling & Implementation

### CV
- **Fine-tuned ViT-base** on FGVC-Aircraft (HuggingFace `Trainer`, AdamW, cosine schedule, augmentation).
- **Baseline:** CLIP zero-shot with variant names as text prompts.

### Numeric ML
- **Models compared:** Logistic Regression, XGBoost, small MLP.
- **Features:** great-circle distance, range margin ratio, ETOPS-required flag, twin-engine flag, manufacturer one-hot.
- **CV:** 5-fold stratified.

### NLP / RAG
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector store:** FAISS (local).
- **Generator:** OpenAI GPT-4o-mini (primary), Claude Haiku (qualitative comparison).
- **Prompt strategies compared:** zero-shot, RAG, RAG + few-shot.

---

## 4. Evaluation & Analysis

_To be filled during M2–M4._

### CV
- Top-1, Top-5 accuracy on FGVC-Aircraft test split
- Per-manufacturer F1
- Confusion matrix
- Grad-CAM qualitative inspection

### Numeric
- Accuracy, F1, ROC-AUC
- Calibration plot
- SHAP feature importance
- Error analysis: edge cases (long-haul twin-engines near ETOPS limit)

### NLP / RAG
- Qualitative rubric (faithfulness, helpfulness, grounding) on 20 hand-crafted questions
- Hallucination probe: 5 questions about non-existent aircraft
- Cross-model comparison: GPT-4o-mini vs Claude Haiku

### Ablations
- Pipeline without RAG (zero-shot LLM)
- Pipeline without CV (manual variant entry)
- Pipeline without numeric model (LLM-only feasibility)

---

## 5. Deployment

- **Platform:** Hugging Face Spaces (Gradio SDK)
- **URL:** https://huggingface.co/spaces/dubattim/aviation-intelligence-system
- **Source:** https://github.com/TimDubath-dev/aviation-intelligence-system
- **Separation of training and inference:** training scripts in `src/*/train*.py` produce artifacts that are pushed to the HF Hub; the deployed app `app/app.py` only loads and runs inference.
- **Screenshots:** see `docs/screenshots/`

---

## 6. Execution Instructions

See [README.md → Quickstart](README.md#quickstart) for full reproduction steps.

---

## 7. Ethical Considerations

- **Aviation emissions**: any tool that makes aviation more engaging risks normalising flying. The app includes a footer note about CO₂ impact and links to emissions calculators.
- **Dataset bias**: FGVC-Aircraft is dominated by Western commercial airliners; military and Eastern-bloc aircraft are underrepresented. Reported per-manufacturer F1 makes this visible.
- **LLM hallucination**: even with RAG, the LLM can fabricate plausible specs. The UI displays retrieved sources and an explicit "not flight-planning advice" disclaimer.
- **API cost & privacy**: user images are sent to OpenAI; the app warns users not to upload personal photos.

---

## 8. Limitations & Future Work

**Known limitations** of the current pipeline:

- **CV is fine-grained-hard.** With 100 visually similar variants and only ~67 training images per class in FGVC, top-1 accuracy is bounded; many top-5 confusions are within the same family (e.g. A330-200 vs A330-300).
- **Numeric labels are synthetic.** Route feasibility is generated from a deterministic rule (range, ETOPS, headwind, payload) plus 3% noise. The model is therefore evaluating the model's ability to learn the rule, not real-world dispatch feasibility.
- **RAG corpus is small** (~1.2k chunks, English Wikipedia only). Niche or non-English aircraft information is unreachable.
- **Out-of-distribution photos.** Web/phone photos differ in angle and lighting from FGVC and degrade CV accuracy. Wikimedia Commons augmentation partially mitigates this.

**Future enhancements** that would meaningfully push the system further:

- **OCR tiebreaker on fuselage text.** Modern airliners carry visible registration codes (e.g. `HB-JNA`) and airline livery branding on the fuselage. An OCR step (e.g. **PaddleOCR**) could extract this text and use it as a tiebreaker when the CV top-5 is ambiguous — the registration uniquely identifies the airframe and can be cross-referenced against an aircraft register database to recover the exact variant. This is especially valuable for distinguishing within-family variants (737-800 vs 737-900) where the visual difference is small.
- **Stronger CV backbone** (DINOv2-large, EVA-02-large, SigLIP) — would push top-1 toward 90% with the same training budget.
- **Real route-feasibility data** scraped from historical flight logs (Flightradar24, OpenSky) instead of the synthetic dataset, making the numeric block evaluate true real-world feasibility.
- **Multilingual RAG** — index Wikipedia in multiple languages and use cross-lingual embeddings.
- **Active learning loop** — let users correct mispredictions in the live app; collect those corrections as new training data.
