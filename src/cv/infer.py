"""Inference helper for the fine-tuned 100-class FGVC-Aircraft ViT classifier."""

from __future__ import annotations

from pathlib import Path

from transformers import pipeline

# 1. Local fine-tuned model produced by `src/cv/train_vit.py`
#    or downloaded once from the HF Hub primary repo (cached under
#    data/hf_cache to keep all artifacts inside this project folder).
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "cv" / "aircraft-vit"
HF_PRIMARY = "dubattim/aviation-intelligence-vit-fgvc"
HF_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "hf_cache"

_pipe = None


def get_pipeline():
    """Lazy-load the classifier. Order: local checkpoint → HF Hub primary repo.

    The HF cache is pinned to <project>/data/hf_cache so the project remains
    self-contained — nothing is read from the user's home directory.
    """
    global _pipe
    if _pipe is None:
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model_path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_PRIMARY
        _pipe = pipeline(
            "image-classification",
            model=model_path,
            model_kwargs={"cache_dir": str(HF_CACHE_DIR)},
        )
    return _pipe


def predict(image_path: str, top_k: int = 5) -> list[dict]:
    p = get_pipeline()
    return p(image_path, top_k=top_k)
