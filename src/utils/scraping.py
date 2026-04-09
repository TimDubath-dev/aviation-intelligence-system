"""Scrape Wikipedia infoboxes for the 100 FGVC-Aircraft variants.

Output: data/processed/aircraft_specs.csv with columns
    variant, manufacturer, range_km, mtow_kg, cruise_speed_kmh, max_pax,
    engine_type, engine_count, first_flight_year, etops_capable, wiki_url

The mapping FGVC variant -> Wikipedia article title is maintained in
data/raw/variant_wiki_mapping.csv (curated by hand for ambiguous variants).
A fallback heuristic ("variant name" + " aircraft") is used for the rest.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ZHAW-AviationIntelligence/0.1 (academic project)"}


# ---------- numeric extraction helpers ----------

_NUM = r"[\d,]+(?:\.\d+)?"


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_range_km(text: str) -> float | None:
    """Find a 'range' figure and convert to km."""
    text = text.replace("\xa0", " ")
    # km direct
    m = re.search(rf"({_NUM})\s*km", text)
    if m:
        return _to_float(m.group(1))
    # nautical miles → km
    m = re.search(rf"({_NUM})\s*nmi", text)
    if m:
        v = _to_float(m.group(1))
        return v * 1.852 if v else None
    # statute miles → km
    m = re.search(rf"({_NUM})\s*mi\b", text)
    if m:
        v = _to_float(m.group(1))
        return v * 1.609 if v else None
    return None


def parse_mass_kg(text: str) -> float | None:
    text = text.replace("\xa0", " ")
    m = re.search(rf"({_NUM})\s*kg", text)
    if m:
        return _to_float(m.group(1))
    m = re.search(rf"({_NUM})\s*lb", text)
    if m:
        v = _to_float(m.group(1))
        return v * 0.453592 if v else None
    return None


def parse_speed_kmh(text: str) -> float | None:
    text = text.replace("\xa0", " ")
    m = re.search(rf"({_NUM})\s*km/h", text)
    if m:
        return _to_float(m.group(1))
    m = re.search(rf"({_NUM})\s*mph", text)
    if m:
        v = _to_float(m.group(1))
        return v * 1.609 if v else None
    m = re.search(rf"Mach\s*({_NUM})", text)
    if m:
        v = _to_float(m.group(1))
        return v * 1234.8 if v else None  # Mach 1 ≈ 1234.8 km/h at sea level
    return None


def parse_int(text: str) -> int | None:
    m = re.search(rf"({_NUM})", text.replace(",", ""))
    return int(float(m.group(1))) if m else None


# ---------- Wikipedia fetching ----------


def fetch_html(title: str) -> str | None:
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": 1,
    }
    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        return None
    return data["parse"]["text"]["*"]


def parse_infobox(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    box = soup.find("table", class_=re.compile("infobox"))
    out: dict[str, str] = {}
    if box is None:
        return out
    for row in box.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if th and td:
            key = th.get_text(" ", strip=True).lower()
            val = td.get_text(" ", strip=True)
            out[key] = val
    return out


def extract_specs(infobox: dict[str, str]) -> dict[str, float | int | None]:
    blob = " ".join(infobox.values())
    return {
        "range_km": parse_range_km(blob),
        "mtow_kg": parse_mass_kg(
            " ".join(v for k, v in infobox.items() if "max" in k and "weight" in k)
            or blob
        ),
        "cruise_speed_kmh": parse_speed_kmh(
            " ".join(v for k, v in infobox.items() if "cruise" in k) or blob
        ),
        "max_pax": parse_int(
            " ".join(v for k, v in infobox.items() if "capacity" in k or "passeng" in k)
        ),
    }


# ---------- driver ----------


def load_variant_list() -> list[str]:
    """Load the 100 FGVC-Aircraft variant names from the dataset metadata."""
    variants_file = RAW_DIR / "fgvc_aircraft" / "fgvc-aircraft-2013b" / "data" / "variants.txt"
    if not variants_file.exists():
        raise FileNotFoundError(
            f"{variants_file} not found. Run `python -m src.cv.download_data` first."
        )
    return [line.strip() for line in variants_file.read_text().splitlines() if line.strip()]


def variant_to_wiki_title(variant: str) -> str:
    """Heuristic mapping. Override via data/raw/variant_wiki_mapping.csv if present."""
    # very loose default — many will need manual mapping
    return variant.replace("/", " ").strip()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    variants = load_variant_list()

    # optional manual override file
    override_path = RAW_DIR / "variant_wiki_mapping.csv"
    overrides: dict[str, str] = {}
    if override_path.exists():
        df = pd.read_csv(override_path)
        overrides = dict(zip(df["variant"], df["wiki_title"]))

    rows = []
    for v in variants:
        title = overrides.get(v) or variant_to_wiki_title(v)
        try:
            html = fetch_html(title)
        except Exception as e:
            print(f"  ! fetch failed for {v} ({title}): {e}")
            html = None
        if not html:
            rows.append({"variant": v, "wiki_title": title, "wiki_url": None})
            continue
        infobox = parse_infobox(html)
        specs = extract_specs(infobox)
        rows.append(
            {
                "variant": v,
                "wiki_title": title,
                "wiki_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "manufacturer": infobox.get("manufacturer"),
                **specs,
                "engine_type": infobox.get("powerplant") or infobox.get("engines"),
            }
        )
        print(f"  ✓ {v} -> {title}: range={specs['range_km']} km")
        time.sleep(0.3)  # be polite

    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "aircraft_specs.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
