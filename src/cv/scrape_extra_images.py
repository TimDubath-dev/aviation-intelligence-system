"""Scrape extra training images for the 100 FGVC variants from Wikimedia Commons.

For each variant we resolve a Wikimedia Commons category (e.g.
'Category:Airbus A320') and download up to MAX_PER_CLASS images.
All images on Commons are freely licensed (CC-BY-SA / public domain).

Output:
    data/raw/extra_images/<variant>/img_001.jpg
    ...
    data/raw/extra_images/_manifest.csv   (variant, file, license, source_url)

Run from the repo root:
    PYTHONPATH=. uv run python -m src.cv.scrape_extra_images
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPING_CSV = REPO_ROOT / "data" / "raw" / "variant_wiki_mapping.csv"
OUT_DIR = REPO_ROOT / "data" / "raw" / "extra_images"
MANIFEST = OUT_DIR / "_manifest.csv"

MAX_PER_CLASS = 40
MIN_SIZE = 256  # px on the short side
HEADERS = {"User-Agent": "ZHAW-AviationIntelligence/0.1 (academic project)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


# --- variant -> Commons category --------------------------------------------

# Many FGVC variants resolve to a Commons category named after the same
# Wikipedia article we already mapped. We add explicit overrides where the
# Wikipedia title doesn't quite match the Commons category convention.
COMMONS_OVERRIDES: dict[str, str] = {
    "707-320": "Category:Boeing 707",
    "727-200": "Category:Boeing 727",
    "737-200": "Category:Boeing 737-200",
    "737-300": "Category:Boeing 737-300",
    "737-400": "Category:Boeing 737-400",
    "737-500": "Category:Boeing 737-500",
    "737-600": "Category:Boeing 737-600",
    "737-700": "Category:Boeing 737-700",
    "737-800": "Category:Boeing 737-800",
    "737-900": "Category:Boeing 737-900",
    "747-100": "Category:Boeing 747-100",
    "747-200": "Category:Boeing 747-200",
    "747-300": "Category:Boeing 747-300",
    "747-400": "Category:Boeing 747-400",
    "757-200": "Category:Boeing 757-200",
    "757-300": "Category:Boeing 757-300",
    "767-200": "Category:Boeing 767-200",
    "767-300": "Category:Boeing 767-300",
    "767-400": "Category:Boeing 767-400ER",
    "777-200": "Category:Boeing 777-200",
    "777-300": "Category:Boeing 777-300",
    "A300B4": "Category:Airbus A300",
    "A310": "Category:Airbus A310",
    "A318": "Category:Airbus A318",
    "A319": "Category:Airbus A319",
    "A320": "Category:Airbus A320",
    "A321": "Category:Airbus A321",
    "A330-200": "Category:Airbus A330-200",
    "A330-300": "Category:Airbus A330-300",
    "A340-200": "Category:Airbus A340-200",
    "A340-300": "Category:Airbus A340-300",
    "A340-500": "Category:Airbus A340-500",
    "A340-600": "Category:Airbus A340-600",
    "A380": "Category:Airbus A380",
    "ATR-42": "Category:ATR 42",
    "ATR-72": "Category:ATR 72",
    "An-12": "Category:Antonov An-12",
    "BAE 146-200": "Category:British Aerospace 146-200",
    "BAE 146-300": "Category:British Aerospace 146-300",
    "BAE-125": "Category:British Aerospace 125",
    "Beechcraft 1900": "Category:Beechcraft 1900",
    "Boeing 717": "Category:Boeing 717",
    "C-130": "Category:Lockheed C-130 Hercules",
    "C-47": "Category:Douglas C-47 Skytrain",
    "CRJ-200": "Category:Bombardier CRJ200",
    "CRJ-700": "Category:Bombardier CRJ700",
    "CRJ-900": "Category:Bombardier CRJ900",
    "Cessna 172": "Category:Cessna 172",
    "Cessna 208": "Category:Cessna 208 Caravan",
    "Cessna 525": "Category:Cessna CitationJet",
    "Cessna 560": "Category:Cessna Citation V",
    "Challenger 600": "Category:Bombardier Challenger 600",
    "DC-10": "Category:McDonnell Douglas DC-10",
    "DC-3": "Category:Douglas DC-3",
    "DC-6": "Category:Douglas DC-6",
    "DC-8": "Category:Douglas DC-8",
    "DC-9-30": "Category:McDonnell Douglas DC-9-30",
    "DH-82": "Category:De Havilland DH.82 Tiger Moth",
    "DHC-1": "Category:De Havilland Canada DHC-1 Chipmunk",
    "DHC-6": "Category:De Havilland Canada DHC-6 Twin Otter",
    "DHC-8-100": "Category:De Havilland Canada Dash 8-100",
    "DHC-8-300": "Category:De Havilland Canada Dash 8-300",
    "DR-400": "Category:Robin DR400",
    "Dornier 328": "Category:Dornier 328",
    "E-170": "Category:Embraer 170",
    "E-190": "Category:Embraer 190",
    "E-195": "Category:Embraer 195",
    "EMB-120": "Category:Embraer EMB 120 Brasília",
    "ERJ 135": "Category:Embraer ERJ 135",
    "ERJ 145": "Category:Embraer ERJ 145",
    "Embraer Legacy 600": "Category:Embraer Legacy 600",
    "Eurofighter Typhoon": "Category:Eurofighter Typhoon",
    "F-16A/B": "Category:General Dynamics F-16 Fighting Falcon",
    "F/A-18": "Category:McDonnell Douglas F/A-18 Hornet",
    "Falcon 2000": "Category:Dassault Falcon 2000",
    "Falcon 900": "Category:Dassault Falcon 900",
    "Fokker 100": "Category:Fokker 100",
    "Fokker 50": "Category:Fokker 50",
    "Fokker 70": "Category:Fokker 70",
    "Global Express": "Category:Bombardier Global Express",
    "Gulfstream IV": "Category:Gulfstream IV",
    "Gulfstream V": "Category:Gulfstream V",
    "Hawk T1": "Category:BAE Systems Hawk T1",
    "Il-76": "Category:Ilyushin Il-76",
    "L-1011": "Category:Lockheed L-1011 TriStar",
    "MD-11": "Category:McDonnell Douglas MD-11",
    "MD-80": "Category:McDonnell Douglas MD-80",
    "MD-87": "Category:McDonnell Douglas MD-87",
    "MD-90": "Category:McDonnell Douglas MD-90",
    "Metroliner": "Category:Fairchild Swearingen Metroliner",
    "Model B200": "Category:Beechcraft Super King Air",
    "PA-28": "Category:Piper PA-28 Cherokee",
    "SR-20": "Category:Cirrus SR20",
    "Saab 2000": "Category:Saab 2000",
    "Saab 340": "Category:Saab 340",
    "Spitfire": "Category:Supermarine Spitfire",
    "Tornado": "Category:Panavia Tornado",
    "Tu-134": "Category:Tupolev Tu-134",
    "Tu-154": "Category:Tupolev Tu-154",
    "Yak-42": "Category:Yakovlev Yak-42",
}


def list_category_files(category: str, limit: int = 100) -> list[str]:
    """Return up to `limit` File: titles inside a Commons category."""
    files: list[str] = []
    cont = None
    while len(files) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": min(50, limit - len(files)),
            "format": "json",
        }
        if cont:
            params["cmcontinue"] = cont
        r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        j = r.json()
        for m in j.get("query", {}).get("categorymembers", []):
            if m["title"].lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                files.append(m["title"])
        cont = j.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return files[:limit]


def file_url(file_title: str, width: int = 800) -> str | None:
    """Resolve a File:foo.jpg → direct URL of a width-px-thumbnail."""
    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": width,
        "format": "json",
    }
    r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for _, p in pages.items():
        info = (p.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        meta = info.get("extmetadata", {})
        license_short = (meta.get("LicenseShortName") or {}).get("value", "")
        return url, license_short
    return None, ""


def download(url: str, out: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < MIN_SIZE:
            return False
        img.save(out, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"      ! download failed: {e}")
        return False


def scrape_variant(variant: str, category: str) -> int:
    cls_dir = OUT_DIR / variant.replace("/", "_").replace(" ", "_")
    cls_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(cls_dir.glob("*.jpg")))
    if existing >= MAX_PER_CLASS:
        return existing
    print(f"  {variant} → {category}")
    try:
        files = list_category_files(category, limit=MAX_PER_CLASS * 2)
    except Exception as e:
        print(f"    ! list failed: {e}")
        return existing
    saved = existing
    rows: list[tuple[str, str, str, str]] = []
    for f in files:
        if saved >= MAX_PER_CLASS:
            break
        url, lic = file_url(f)
        if not url:
            continue
        out = cls_dir / f"img_{saved + 1:03d}.jpg"
        if download(url, out):
            saved += 1
            rows.append((variant, str(out.relative_to(REPO_ROOT)), lic, url))
        time.sleep(0.2)  # be polite
    print(f"    saved {saved}/{MAX_PER_CLASS}")
    if rows:
        write_header = not MANIFEST.exists()
        with MANIFEST.open("a", newline="") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(["variant", "file", "license", "source_url"])
            w.writerows(rows)
    return saved


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING_CSV)
    total = 0
    for variant in mapping["variant"]:
        cat = COMMONS_OVERRIDES.get(variant)
        if not cat:
            print(f"  {variant}: no Commons category mapping, skipping")
            continue
        total += scrape_variant(variant, cat)
    print(f"\nTotal images on disk: {total}")


if __name__ == "__main__":
    main()
