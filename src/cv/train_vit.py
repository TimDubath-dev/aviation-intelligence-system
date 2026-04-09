"""Fine-tune ViT-base on FGVC-Aircraft (variant level, 100 classes).

Uses HuggingFace Trainer + torchvision FGVCAircraft dataset.
Saves best model to models/cv/aircraft-vit/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FGVCAircraft
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "fgvc_aircraft"
OUT_DIR = REPO_ROOT / "models" / "cv" / "aircraft-vit"
MODEL_ID = "google/vit-base-patch16-224"


class FGVCWrapper(Dataset):
    def __init__(self, split: str, processor: ViTImageProcessor, train: bool):
        self.ds = FGVCAircraft(
            root=str(DATA_ROOT), split=split, annotation_level="variant", download=True
        )
        self.processor = processor
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((224, 224)),
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = self.tx(img.convert("RGB"))
        pixel = self.processor(img, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel, "labels": int(label)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    top5 = np.argsort(logits, axis=-1)[:, -5:]
    return {
        "accuracy": float((preds == labels).mean()),
        "top5_accuracy": float(np.mean([l in t for l, t in zip(labels, top5)])),
    }


def main() -> None:
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    train_ds = FGVCWrapper("train", processor, train=True)
    val_ds = FGVCWrapper("val", processor, train=False)

    n_classes = len(train_ds.ds.classes)
    model = ViTForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=n_classes,
        id2label={i: c for i, c in enumerate(train_ds.ds.classes)},
        label2id={c: i for i, c in enumerate(train_ds.ds.classes)},
        ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    processor.save_pretrained(str(OUT_DIR))
    print(f"Saved best ViT to {OUT_DIR}")


if __name__ == "__main__":
    main()
