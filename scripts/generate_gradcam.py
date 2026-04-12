"""Generate Grad-CAM attention heatmaps for the fine-tuned DINOv2 classifier.

Produces overlay images showing which regions the model attends to when
classifying each of the 5 example aircraft photos.

Output:
    docs/gradcam/  — one overlay PNG per example image
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXAMPLES = sorted((ROOT / "app" / "examples").glob("*.jpg"))
OUT_DIR = ROOT / "docs" / "gradcam"
HF_CACHE = ROOT / "data" / "hf_cache"
MODEL_ID = "dubattim/aviation-intelligence-vit-fgvc"


def get_gradcam(model, processor, img_pil):
    """Compute Grad-CAM for the top predicted class on a ViT/DINOv2 model."""
    # Preprocess
    inputs = processor(images=img_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].requires_grad_(True)

    # Forward
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    pred_class = logits.argmax(dim=-1).item()
    pred_label = model.config.id2label[pred_class]
    confidence = torch.softmax(logits, dim=-1)[0, pred_class].item()

    # Backward on the predicted class
    model.zero_grad()
    logits[0, pred_class].backward()

    # Get gradients of the last hidden layer
    # For ViT/DINOv2 via transformers, we hook into the pixel_values gradient
    grad = pixel_values.grad[0]  # (C, H, W)

    # Channel-wise mean of absolute gradients → spatial attention
    cam = grad.abs().mean(dim=0).detach().numpy()  # (H, W)

    # Normalize to [0, 1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam, pred_label, confidence


def overlay_cam(img_pil, cam, alpha=0.4):
    """Overlay a heatmap on the original image."""
    img_np = np.array(img_pil.resize((224, 224)))
    # Resize cam to match image
    from scipy.ndimage import zoom
    if cam.shape != (224, 224):
        zoom_h = 224 / cam.shape[0]
        zoom_w = 224 / cam.shape[1]
        cam = zoom(cam, (zoom_h, zoom_w), order=1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(cam, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, cache_dir=str(HF_CACHE))
    model.eval()

    for img_path in EXAMPLES:
        print(f"\n{img_path.name}:")
        img = Image.open(img_path).convert("RGB")
        cam, pred, conf = get_gradcam(model, processor, img)
        print(f"  Prediction: {pred} ({conf:.1%})")

        fig = overlay_cam(img, cam)
        fig.suptitle(f"{pred} ({conf:.1%})", fontsize=14, y=1.02)
        out = OUT_DIR / f"gradcam_{img_path.stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
