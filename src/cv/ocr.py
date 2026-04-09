"""OCR-based fuselage-registration tiebreaker.

Reads visible text on an aircraft photo (e.g. 'HB-JNA') and looks the
registration up in the OpenSky aircraft database. If the registration maps
to one of the 100 FGVC variants AND that variant is in the CV top-5, the
pipeline promotes it to top-1 — otherwise the OCR result is reported but
ignored.

Lazy imports keep EasyOCR's ~64 MB model out of cold-start latency unless
the user actually toggles OCR on.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOOKUP_PATH = REPO_ROOT / "data" / "processed" / "registration_to_variant.parquet"
HF_CACHE_DIR = REPO_ROOT / "data" / "hf_cache"

# International civil aircraft registration prefixes (subset). Most are
# 1-2 letters + dash + 2-5 alphanumerics. US (N + 1-5 alphanumerics) is
# the major exception — it has no dash.
REG_REGEXES = [
    re.compile(r"\bN[0-9][0-9A-Z]{1,4}\b"),                           # USA: N12345 / N1AB
    re.compile(r"\b[A-Z]{1,2}-[A-Z0-9]{2,5}\b"),                       # G-ABCD, HB-JNA, D-AABC
    re.compile(r"\b(JA|HL|VH|VT|RA|UR|9V|9M)[A-Z0-9]{3,5}\b"),         # no-dash: JA, HL, VH, VT, ...
]


@lru_cache(maxsize=1)
def _lookup() -> dict[str, str]:
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"{LOOKUP_PATH} missing. Run "
            "`PYTHONPATH=. uv run python -m src.cv.build_registration_lookup` first."
        )
    df = pd.read_parquet(LOOKUP_PATH)
    return dict(zip(df["registration"], df["variant"]))


@lru_cache(maxsize=1)
def _reader():
    """Lazy-load EasyOCR. Models are cached under data/hf_cache/easyocr."""
    import easyocr  # noqa: PLC0415

    cache = HF_CACHE_DIR / "easyocr"
    cache.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(["en"], gpu=False, model_storage_directory=str(cache),
                          download_enabled=True, verbose=False)


def extract_text(image_path: str) -> list[str]:
    """Run OCR and return all extracted text snippets, upper-cased."""
    try:
        reader = _reader()
    except Exception as e:
        print(f"[ocr] EasyOCR unavailable ({e})")
        return []
    out = reader.readtext(image_path, detail=0, paragraph=False)
    return [s.upper().strip() for s in out if s and s.strip()]


def find_registration(snippets: list[str]) -> str | None:
    """Return the first plausible aircraft registration found in OCR snippets."""
    blob = " ".join(snippets)
    blob = re.sub(r"[^A-Z0-9 \-]", " ", blob)  # strip punctuation
    for rx in REG_REGEXES:
        m = rx.search(blob)
        if m:
            reg = m.group(0).replace(" ", "")
            return reg
    return None


def lookup_variant(registration: str) -> str | None:
    return _lookup().get(registration.upper().replace("-", "").replace("-", ""))


def lookup_variant_loose(registration: str) -> str | None:
    """Try with and without dashes (OCR may eat the dash)."""
    reg = registration.upper().replace(" ", "")
    table = _lookup()
    if reg in table:
        return table[reg]
    if "-" not in reg and len(reg) >= 4:
        # try inserting a dash after the first 1 or 2 chars
        for cut in (1, 2):
            cand = reg[:cut] + "-" + reg[cut:]
            if cand in table:
                return table[cand]
    if "-" in reg:
        no_dash = reg.replace("-", "")
        if no_dash in table:
            return table[no_dash]
    return None


def detect(image_path: str) -> dict:
    """One-shot: image → {registration, variant, ocr_text}."""
    snippets = extract_text(image_path)
    reg = find_registration(snippets)
    variant = lookup_variant_loose(reg) if reg else None
    return {
        "registration": reg,
        "variant": variant,
        "ocr_text": snippets,
    }
