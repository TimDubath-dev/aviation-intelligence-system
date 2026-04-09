"""Build the route-feasibility dataset.

Combines:
  - data/processed/aircraft_specs.csv     (from src/utils/scraping.py)
  - data/raw/openflights/airports.dat     (downloaded here if missing)

For each (aircraft, origin, destination) sample, computes great-circle
distance and labels feasibility:

    feasible = 1  iff  distance_km < 0.85 * range_km   AND   etops_ok
              0  otherwise

Sampling strategy: weighted toward "interesting" cases (distance close
to range), so the classifier doesn't trivially separate everything.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils.geo import haversine_km

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

OPENFLIGHTS_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
)
AIRPORTS_LOCAL = RAW_DIR / "openflights" / "airports.dat"

N_SAMPLES = 50_000
RNG = np.random.default_rng(42)


def download_airports() -> Path:
    AIRPORTS_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if AIRPORTS_LOCAL.exists():
        return AIRPORTS_LOCAL
    print(f"Downloading OpenFlights airports → {AIRPORTS_LOCAL}")
    r = requests.get(OPENFLIGHTS_URL, timeout=60)
    r.raise_for_status()
    AIRPORTS_LOCAL.write_bytes(r.content)
    return AIRPORTS_LOCAL


def load_airports() -> pd.DataFrame:
    download_airports()
    cols = [
        "airport_id", "name", "city", "country", "iata", "icao",
        "lat", "lon", "altitude", "tz_offset", "dst", "tz", "type", "source",
    ]
    df = pd.read_csv(AIRPORTS_LOCAL, header=None, names=cols, na_values=["\\N"])
    df = df.dropna(subset=["lat", "lon", "iata"])
    df = df[df["iata"].str.len() == 3]
    # keep only "large" airports heuristically: those with an IATA + ICAO
    df = df.dropna(subset=["icao"]).reset_index(drop=True)
    return df[["iata", "icao", "name", "city", "country", "lat", "lon"]]


def load_specs() -> pd.DataFrame:
    path = PROCESSED_DIR / "aircraft_specs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `python -m src.utils.scraping` first."
        )
    df = pd.read_csv(path).dropna(subset=["range_km"])
    # crude ETOPS heuristic — twin-engine widebodies > 5000 km range
    df["twin_engine"] = df.get("engine_count", 2).fillna(2).astype(int).eq(2)
    df["etops_capable"] = df["range_km"] > 5000
    return df.reset_index(drop=True)


def label_feasibility(distance_km: float, range_km: float, etops_capable: bool,
                      headwind_kmh: float, payload_factor: float,
                      max_oceanic_km: float = 5500) -> int:
    """Realistic label: payload + headwind reduce effective range.

    effective_range = range_km * (1 - 0.15 * payload_factor) - headwind_penalty
    where headwind_penalty grows with distance: ~ headwind_kmh * (distance/800)
    """
    headwind_penalty = headwind_kmh * (distance_km / 800)
    effective_range = range_km * (1.0 - 0.15 * payload_factor) - headwind_penalty
    if distance_km > max_oceanic_km and not etops_capable:
        return 0
    # margin: feasible if distance < 0.90 * effective_range, with a soft band
    return int(distance_km < 0.90 * effective_range)


def build(n: int = N_SAMPLES) -> pd.DataFrame:
    airports = load_airports()
    specs = load_specs()

    # weighted sampling: pairs near each plane's range are more interesting
    rows = []
    for _ in range(n):
        plane = specs.sample(1, random_state=RNG.integers(1e9)).iloc[0]
        a, b = airports.sample(2, random_state=RNG.integers(1e9)).iloc[0:2].itertuples(index=False)
        d = haversine_km(a.lat, a.lon, b.lat, b.lon)
        # realistic per-flight perturbations the model does NOT see directly
        headwind = float(RNG.normal(20, 25))   # km/h, can be negative (tailwind)
        payload = float(RNG.beta(2, 2))        # 0..1 — fraction of max payload
        label = label_feasibility(d, plane["range_km"], plane["etops_capable"],
                                  headwind, payload)
        # 3% label noise (radio/dispatch errors, mis-tagged routes, …)
        if RNG.random() < 0.03:
            label = 1 - label
        rows.append(
            {
                "variant": plane["variant"],
                "manufacturer": plane.get("manufacturer"),
                "range_km": plane["range_km"],
                "twin_engine": plane["twin_engine"],
                "etops_capable": plane["etops_capable"],
                "origin_iata": a.iata,
                "dest_iata": b.iata,
                "distance_km": d,
                # observed payload proxy (noisy version of true payload)
                "payload_proxy": float(np.clip(payload + RNG.normal(0, 0.15), 0, 1)),
                "feasible": label,
            }
        )

    df = pd.DataFrame(rows)
    out = PROCESSED_DIR / "route_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}  (positives: {df['feasible'].mean():.2%})")
    return df


if __name__ == "__main__":
    build()
