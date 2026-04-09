---
title: Aviation Intelligence System
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.7.0
app_file: app/app.py
pinned: false
license: mit
short_description: Identify aircraft, check route, explain in plain English.
---

# Aviation Intelligence System

ZHAW "AI Applications" FS26 — Semester Project by **Tim Dubath**.

A multimodal AI application that, from a single aircraft photo and a route (origin → destination airport), identifies the aircraft variant, predicts whether it can realistically fly the requested route, and explains the verdict in natural language grounded in retrieved documents.

> **Live demo:** https://huggingface.co/spaces/dubattim/aviation-intelligence-system
> **Source:** https://github.com/TimDubath-dev/aviation-intelligence-system

## Overview

This project combines **all three** AI blocks from the course into one tightly integrated pipeline:

| Block | Role | Models |
|---|---|---|
| **Computer Vision** | Identify aircraft variant from photo | Fine-tuned ViT-base (FGVC-Aircraft, 100 variants) + CLIP zero-shot baseline |
| **ML on Numeric Data** | Predict route feasibility from specs + great-circle distance | Logistic Regression + XGBoost + MLP |
| **NLP / RAG** | Explain the verdict in natural language with grounded citations | GPT-4o-mini with FAISS retrieval over Wikipedia (+ Claude Haiku comparison) |

The blocks are **chained**: CV output → spec lookup → numeric model → LLM explanation. Each block consumes the previous block's output.

## Architecture

```
 [photo]                        [origin / destination]
    │                                    │
    ▼                                    │
 ┌──────────────────┐                    │
 │  CV: ViT-base    │                    │
 │  100 variants    │                    │
 └────────┬─────────┘                    │
          │ predicted variant            │
          ▼                              │
 ┌──────────────────┐                    │
 │  Spec lookup     │  ◄─────────────────┘
 │  (Wikipedia      │
 │   infoboxes)     │
 └────────┬─────────┘
          │ specs + distance + features
          ▼
 ┌──────────────────┐
 │  Numeric ML      │
 │  feasibility     │
 │  classifier      │
 └────────┬─────────┘
          │ verdict + probability
          ▼
 ┌──────────────────┐     ┌──────────────────┐
 │  RAG retriever   │ ──► │  LLM explainer   │ ──► natural-language answer
 │  (FAISS)         │     │  (GPT-4o-mini)   │
 └──────────────────┘     └──────────────────┘
```

## Data Sources

1. **FGVC-Aircraft** — 10k images, 100 fine-grained variants (CV training).
2. **Aircraft specs** — scraped from Wikipedia infoboxes (range, MTOW, ETOPS, …).
3. **OpenFlights airports** — IATA/ICAO codes + lat/lon for great-circle distance.
4. **Wikipedia article corpus** — chunked & embedded into FAISS for RAG.

## Quickstart

```bash
# 1. Install (uses uv)
cd semester-project
uv sync

# 2. Configure secrets
cp .env.example .env
# fill in OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY)

# 3. Build data
python -m src.cv.download_data
python -m src.utils.scraping        # scrape aircraft specs
python -m src.numeric.build_dataset
python -m src.nlp.build_index

# 4. Train models
python -m src.cv.train_vit
python -m src.numeric.train

# 5. Run the app
python app/app.py
```

## Project Structure

```
semester-project/
├── README.md              ← this file
├── DOCUMENTATION.md       ← mandatory Q&A documentation
├── pyproject.toml
├── .env.example
├── data/                  ← raw + processed datasets (gitignored)
├── notebooks/             ← EDA + evaluation notebooks
├── src/
│   ├── cv/                ← Computer Vision block
│   ├── numeric/           ← ML Numeric block
│   ├── nlp/               ← NLP / RAG block
│   ├── utils/             ← scraping, geo helpers
│   └── pipeline.py        ← end-to-end orchestration
├── app/                   ← Gradio inference app (deployed entrypoint)
├── tests/
└── docs/                  ← architecture diagram + screenshots
```

**Training and inference are separated:** training scripts live in `src/*/train*.py`, while `app/app.py` only loads pre-trained artifacts.

## Deploy to Hugging Face Spaces

The project is one push away from a public Space:

```bash
# 1. Create a new (empty) Space at https://huggingface.co/new-space
#    SDK: Gradio  ·  Hardware: CPU basic (free)
#    Name: aviation-intelligence-system  (or whatever you prefer)

# 2. Add the OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY) to the
#    Space's "Settings → Variables and secrets" page.

# 3. From the project root, push everything to the Space:
git init
git lfs install                        # for the FAISS index + model pkls
git lfs track "*.faiss" "*.pkl" "*.parquet"
git add .
git commit -m "Initial deployment"
git remote add hf https://huggingface.co/spaces/<your-username>/aviation-intelligence-system
git push hf main
```

The Space build will:
1. Read `requirements.txt` and install the inference deps (~3 min)
2. Launch `app/app.py` (the Gradio entrypoint declared in the README frontmatter)
3. On first request, lazy-download the fine-tuned ViT from
   `dubattim/aviation-intelligence-vit-fgvc` into `data/hf_cache/`

Total Space repo size after `git lfs` is ~7 MB — well within the free tier.

## Documentation

Full project documentation following the mandatory Q&A template is in [DOCUMENTATION.md](DOCUMENTATION.md).

## Disclaimer

This project is an academic exercise. The route-feasibility predictions are illustrative only and **must not be used for actual flight planning**.
