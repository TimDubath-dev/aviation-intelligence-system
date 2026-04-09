"""FAISS retriever wrapping the corpus built by build_index.py."""

from __future__ import annotations

from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "data" / "rag_corpus"
HF_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Retriever:
    def __init__(self):
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.index = faiss.read_index(str(CORPUS_DIR / "index.faiss"))
        self.chunks = pd.read_parquet(CORPUS_DIR / "chunks.parquet")
        self.encoder = SentenceTransformer(EMBED_MODEL, cache_folder=str(HF_CACHE_DIR))

    def search(self, query: str, k: int = 5) -> list[dict]:
        emb = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(emb, k)
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            row = self.chunks.iloc[int(i)]
            out.append({"score": float(s), "title": row["title"], "text": row["text"]})
        return out
