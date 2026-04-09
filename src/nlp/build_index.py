"""Build a FAISS index from Wikipedia article text for the 100 aircraft variants
plus the top ~200 airports by traffic.

Output:
    data/rag_corpus/chunks.parquet   (chunk_id, source, title, text)
    data/rag_corpus/index.faiss
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
CORPUS_DIR = REPO_ROOT / "data" / "rag_corpus"
HF_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ZHAW-AviationIntelligence/0.1"}
CHUNK_TOKENS = 500  # approx — we use word count as proxy


def fetch_extract(title: str) -> str | None:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for _, p in pages.items():
        if "extract" in p:
            return p["extract"]
    return None


def chunk(text: str, n_words: int = CHUNK_TOKENS) -> list[str]:
    words = re.split(r"\s+", text.strip())
    return [" ".join(words[i:i + n_words]) for i in range(0, len(words), n_words)]


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    specs = pd.read_csv(PROCESSED / "aircraft_specs.csv")

    titles: list[tuple[str, str]] = []  # (source, title)
    for _, row in specs.iterrows():
        if pd.notna(row.get("wiki_title")):
            titles.append(("aircraft", row["wiki_title"]))

    # also a handful of major airports
    airports = pd.read_csv(REPO_ROOT / "data" / "raw" / "openflights" / "airports.dat",
                           header=None, na_values=["\\N"])
    # column 1 is name, column 4 is IATA
    big = ["JFK", "LHR", "ZRH", "DXB", "HND", "SIN", "FRA", "CDG", "LAX", "ORD",
           "ATL", "AMS", "IST", "PEK", "PVG", "SYD", "GRU", "DOH", "SFO", "MEX"]
    for iata in big:
        match = airports[airports[4] == iata]
        if len(match):
            titles.append(("airport", f"{match.iloc[0][1]}"))

    rows = []
    for source, title in titles:
        try:
            txt = fetch_extract(title)
        except Exception as e:
            print(f"  ! {title}: {e}")
            continue
        if not txt:
            continue
        for i, c in enumerate(chunk(txt)):
            rows.append({"source": source, "title": title, "chunk_id": i, "text": c})
        print(f"  ✓ {title}: {len(chunk(txt))} chunks")

    df = pd.DataFrame(rows)
    df.to_parquet(CORPUS_DIR / "chunks.parquet")
    print(f"Total chunks: {len(df)}")

    print("Embedding ...")
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    enc = SentenceTransformer(EMBED_MODEL, cache_folder=str(HF_CACHE_DIR))
    emb = enc.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True,
                     normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(CORPUS_DIR / "index.faiss"))
    with open(CORPUS_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"embed_model": EMBED_MODEL, "n": len(df)}, f)
    print(f"Wrote FAISS index ({len(df)} vectors) → {CORPUS_DIR / 'index.faiss'}")


if __name__ == "__main__":
    main()
