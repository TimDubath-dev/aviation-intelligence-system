"""Download FGVC-Aircraft dataset (100 variants) into data/raw/fgvc_aircraft/.

Uses torchvision's built-in FGVCAircraft dataset, which downloads from the
Oxford VGG mirror and lays out train/val/test splits automatically.
"""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import FGVCAircraft

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "fgvc_aircraft"


def download() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        print(f"Downloading FGVC-Aircraft split={split} ...")
        FGVCAircraft(
            root=str(DATA_ROOT),
            split=split,
            annotation_level="variant",
            download=True,
        )
    print(f"Done. Data lives under: {DATA_ROOT}")


if __name__ == "__main__":
    download()
