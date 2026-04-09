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

# ✈️ Aviation Intelligence System

> ZHAW *AI Applications* — FS26 Semester Project · **Tim Dubath**

A multimodal AI assistant for aviation enthusiasts, plane spotters, and journalists. Upload a photo of an aircraft and pick a route — the system will identify the aircraft, decide whether it could realistically fly that route, and explain the answer in natural language with citations from Wikipedia.

| | |
|---|---|
| 🚀 **Live demo**  | https://huggingface.co/spaces/dubattim/aviation-intelligence-system |
| 💻 **Source code** | https://github.com/TimDubath-dev/aviation-intelligence-system |
| 🤖 **Trained ViT** | https://huggingface.co/dubattim/aviation-intelligence-vit-fgvc |
| 📄 **Documentation** | [`DOCUMENTATION.md`](DOCUMENTATION.md) |

---

## What it does

```
        ┌──────────────────┐                       ┌──────────────────┐
        │  Aircraft photo  │                       │  Route ZRH→JFK   │
        └────────┬─────────┘                       └────────┬─────────┘
                 ▼                                          │
        ┌──────────────────┐                                │
        │  CV: ViT-base    │                                │
        │  100 variants    │                                │
        └────────┬─────────┘                                │
                 │ predicted variant                        │
                 ▼                                          │
        ┌──────────────────┐                                │
        │  Spec lookup     │ ◄──────────────────────────────┘
        │  (curated CSV)   │
        └────────┬─────────┘
                 │ specs + great-circle distance + features
                 ▼
        ┌──────────────────┐
        │  Numeric ML      │   Logistic Regression │ MLP │ XGBoost
        │  feasibility     │
        └────────┬─────────┘
                 │ verdict + probability
                 ▼
        ┌──────────────────┐    ┌──────────────────────────┐
        │  RAG retriever   │ ──►│  LLM explainer           │ ──►  natural-language answer
        │  FAISS · MiniLM  │    │  GPT-4o-mini / Haiku     │      with cited sources
        └──────────────────┘    └──────────────────────────┘
```

The pipeline is **chained**, not parallel — every block consumes the previous block's output. CV decides *which plane*, the spec lookup pulls the matching row, the numeric model predicts feasibility from the resulting features, and the LLM produces a grounded explanation given all upstream outputs.

## All three AI blocks

| Block | Role | Models / techniques |
|---|---|---|
| **🖼️ Computer Vision** | Identify the aircraft variant from a photo | Fine-tuned **ViT-base** (10 epochs on FGVC-Aircraft, 100 classes) + **CLIP zero-shot** baseline for comparison |
| **📊 ML on Numeric Data** | Predict whether the aircraft can fly the route | **Logistic Regression**, **MLP**, **XGBoost** trained on 50k synthetic (plane, route) examples with realistic headwind/payload perturbations and 3% label noise |
| **💬 NLP / RAG** | Explain the verdict in natural language with citations | **FAISS** index over Wikipedia (~1.2k chunks) + **MiniLM** embeddings; **GPT-4o-mini** primary, **Claude Haiku** secondary; three prompt strategies compared (zero-shot / RAG / RAG + few-shot) |

## Data sources (4)

| # | Source | Type | Use |
|---|---|---|---|
| 1 | **FGVC-Aircraft** (Oxford VGG) | 10k images, 100 fine-grained variants | CV training & evaluation |
| 2 | **Curated aircraft specs** | 100 rows × 10 fields, hand-curated CSV | Spec lookup, numeric features |
| 3 | **OpenFlights airports** | ~7k IATA airports with lat/lon | Great-circle distance, route resolution |
| 4 | **Wikipedia article corpus** | ~120 articles → 1236 text chunks | RAG grounding for the LLM explainer |

## Quickstart (local)

```bash
cd semester-project
uv sync                                  # install pinned deps
cp .env.example .env && $EDITOR .env     # add OPENAI_API_KEY (+ ANTHROPIC_API_KEY)

# build the data pipeline (downloads ~2.7 GB of FGVC images)
uv run python -m src.cv.download_data
uv run python -m src.utils.build_specs
uv run python -m src.numeric.build_dataset
uv run python -m src.nlp.build_index

# train models
uv run python -m src.numeric.train       # CPU, ~1 min
# ViT training is GPU-only — use the Colab notebook below instead

# launch the Gradio app
PYTHONPATH=. uv run python app/app.py
```

### Train the ViT on Colab

The fine-tuned 100-class ViT lives at `dubattim/aviation-intelligence-vit-fgvc`. To reproduce it, open [`notebooks/train_vit_colab.ipynb`](notebooks/train_vit_colab.ipynb) in Google Colab, set runtime to **T4 GPU**, paste a Hugging Face write token, and run all cells (~30 min).

## Project structure

```
semester-project/
├── README.md                ← you are here
├── DOCUMENTATION.md         ← mandatory Q&A documentation
├── pyproject.toml           ← uv-managed deps (training)
├── requirements.txt         ← pip deps (HF Spaces inference)
├── .env.example
│
├── data/
│   ├── raw/
│   │   ├── fgvc_aircraft/             ← (gitignored, 5.2 GB) FGVC images
│   │   ├── openflights/airports.dat   ← ~7k airport coordinates
│   │   ├── curated_aircraft_specs.csv ← 100 hand-curated aircraft
│   │   └── variant_wiki_mapping.csv   ← FGVC variant → Wikipedia title
│   ├── processed/
│   │   ├── aircraft_specs.csv         ← canonical specs used at runtime
│   │   └── route_dataset.csv          ← (gitignored) 50k labeled training rows
│   ├── rag_corpus/
│   │   ├── chunks.parquet             ← 1236 Wikipedia text chunks
│   │   └── index.faiss                ← FAISS index
│   └── hf_cache/                      ← (gitignored) HF model cache
│
├── models/
│   ├── numeric/
│   │   ├── logreg.pkl  mlp.pkl  xgboost.pkl
│   │   ├── metrics.json               ← per-model evaluation
│   │   ├── permutation_importance.json
│   │   └── calibration.png
│   └── cv/                            ← (gitignored) local ViT checkpoint
│
├── notebooks/
│   ├── 01_eda_specs.ipynb
│   ├── 02_eda_images.ipynb
│   └── train_vit_colab.ipynb          ← Colab GPU training notebook
│
├── src/
│   ├── cv/         ← download_data, train_vit, clip_baseline, infer
│   ├── numeric/    ← build_dataset, features, train, predict
│   ├── nlp/        ← build_index, retriever, prompts, generate
│   ├── utils/      ← geo (haversine), scraping, build_specs
│   └── pipeline.py ← end-to-end orchestrator (image+route → answer)
│
├── app/
│   ├── app.py                         ← Gradio Blocks UI (HF Space entrypoint)
│   └── examples/                      ← 5 example aircraft photos
│
├── scripts/
│   ├── smoke_pipeline.py              ← numeric+NLP smoke test
│   └── smoke_full_pipeline.py         ← full image→answer test
│
└── tests/test_geo.py
```

**Separation of training and inference:** every `src/*/train*.py` is training-only and produces an artifact (`.pkl`, HF Hub repo). The deployed app `app/app.py` only loads those artifacts — it does no training.

## Reproducibility

| Step | Determinism |
|---|---|
| FGVC download | Identical bytes (Oxford VGG mirror) |
| Curated specs | Versioned CSV in `data/raw/` |
| Route dataset synthesis | `numpy.random.default_rng(42)` |
| Numeric model training | `random_state=42` for all 3 models, 5-fold stratified CV |
| ViT training | Same hyperparams in `train_vit.py` and the Colab notebook |
| FAISS index | Deterministic embeddings (MiniLM, normalized) |

## Deploy your own copy to Hugging Face Spaces

```bash
# 1. Create an empty Gradio Space at https://huggingface.co/new-space
# 2. Set OPENAI_API_KEY (+ ANTHROPIC_API_KEY) under Settings → Secrets
# 3. From the project root:
git init && git lfs install
git lfs track "*.faiss" "*.pkl" "*.parquet"
git add . && git commit -m "Initial deployment"
git remote add hf https://huggingface.co/spaces/<you>/aviation-intelligence-system
git push hf main
```

The Space ships only the inference assets (~7 MB after LFS). The fine-tuned ViT is lazily fetched from `dubattim/aviation-intelligence-vit-fgvc` on first request and cached in `data/hf_cache/`.

## Disclaimer

This is an **educational project**. Route-feasibility predictions consider only range, ETOPS, and synthetic headwind/payload effects — they ignore real-world factors like weather, runway length, payload limits, regulatory clearances, fuel pricing, and ATC routing. **Do not use for actual flight planning.**

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

- ZHAW *AI Applications* (FS26) — Jasmin Heierli & Benjamin Kühnis
- FGVC-Aircraft dataset — Oxford Visual Geometry Group
- OpenFlights airports — `openflights.org` (CC-BY-SA)
- Wikipedia — for the aircraft and airport text corpus
