"""Build the canonical aircraft_specs.csv used by the numeric and NLP blocks.

Strategy:
  - Primary source: data/raw/curated_aircraft_specs.csv (hand-curated for
    accuracy — Wikipedia infoboxes for aircraft do NOT contain performance
    specs; those live in a free-form 'Specifications' section that is hard
    to parse reliably).
  - Wikipedia title (for the RAG corpus + UI links) is taken from
    data/raw/variant_wiki_mapping.csv.
  - Output: data/processed/aircraft_specs.csv with one row per FGVC variant.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    specs = pd.read_csv(RAW / "curated_aircraft_specs.csv")
    mapping = pd.read_csv(RAW / "variant_wiki_mapping.csv")

    df = specs.merge(mapping, on="variant", how="left")
    df["wiki_url"] = df["wiki_title"].apply(
        lambda t: f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}" if pd.notna(t) else None
    )

    # derived flags used downstream
    df["twin_engine"] = df["engine_count"] == 2

    out = PROCESSED / "aircraft_specs.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {out}")
    print(df[["variant", "manufacturer", "range_km", "etops_capable"]].head(10).to_string())


if __name__ == "__main__":
    main()
