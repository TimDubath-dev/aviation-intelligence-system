"""CLIP zero-shot baseline on FGVC-Aircraft test split."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.datasets import FGVCAircraft
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "fgvc_aircraft"
OUT = REPO_ROOT / "models" / "cv" / "clip_baseline_metrics.json"
MODEL = "openai/clip-vit-large-patch14"


def main() -> None:
    ds = FGVCAircraft(root=str(DATA_ROOT), split="test", annotation_level="variant", download=True)
    classes = ds.classes
    prompts = [f"a photo of a {c} aircraft" for c in classes]

    model = CLIPModel.from_pretrained(MODEL).eval()
    processor = CLIPProcessor.from_pretrained(MODEL)
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    correct1, correct5, total = 0, 0, 0
    for img, label in tqdm(ds, desc="CLIP zero-shot"):
        inputs = processor(images=img.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ text_emb.T).squeeze(0)
        top5 = sims.topk(5).indices.tolist()
        correct1 += int(top5[0] == label)
        correct5 += int(label in top5)
        total += 1

    metrics = {
        "model": MODEL,
        "n": total,
        "top1": correct1 / total,
        "top5": correct5 / total,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
