"""End-to-end pipeline: image + route → variant → feasibility → explanation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.cv import infer as cv_infer
from src.nlp import prompts
from src.nlp.generate import generate
from src.nlp.retriever import Retriever
from src.numeric.predict import predict_one
from src.utils.geo import haversine_km

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = REPO_ROOT / "data" / "processed" / "aircraft_specs.csv"
AIRPORTS = REPO_ROOT / "data" / "raw" / "openflights" / "airports.dat"


@lru_cache(maxsize=1)
def _specs() -> pd.DataFrame:
    return pd.read_csv(SPECS)


@lru_cache(maxsize=1)
def _airports() -> pd.DataFrame:
    cols = ["airport_id", "name", "city", "country", "iata", "icao",
            "lat", "lon", "altitude", "tz_offset", "dst", "tz", "type", "source"]
    return pd.read_csv(AIRPORTS, header=None, names=cols, na_values=["\\N"]).dropna(subset=["iata"])


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return Retriever()


@dataclass
class PipelineResult:
    cv_top5: list[dict]
    variant: str
    specs: dict
    distance_km: float
    feasibility: dict
    explanation: str
    sources: list[str]


def lookup_airport(iata: str) -> dict:
    a = _airports()
    row = a[a["iata"] == iata.upper()]
    if not len(row):
        raise ValueError(f"Unknown IATA code: {iata}")
    return row.iloc[0].to_dict()


def lookup_specs(variant: str) -> dict:
    s = _specs()
    row = s[s["variant"] == variant]
    if not len(row):
        # fall back: take fuzziest match by manufacturer prefix
        return {"variant": variant, "range_km": float("nan")}
    return row.iloc[0].to_dict()


def run(image_path: str, origin_iata: str, dest_iata: str,
        strategy: str = "rag", llm: str = "openai") -> PipelineResult:
    # 1) CV
    top5 = cv_infer.predict(image_path, top_k=5)
    variant = top5[0]["label"]

    # 2) Spec lookup
    specs = lookup_specs(variant)

    # 3) Distance
    o, d = lookup_airport(origin_iata), lookup_airport(dest_iata)
    dist = haversine_km(o["lat"], o["lon"], d["lat"], d["lon"])

    # 4) Numeric model
    feas = predict_one(
        variant=variant,
        manufacturer=specs.get("manufacturer", ""),
        range_km=float(specs.get("range_km") or 0),
        twin_engine=bool(specs.get("twin_engine", True)),
        etops_capable=bool(specs.get("etops_capable", False)),
        distance_km=dist,
    )

    # 5) NLP/RAG
    ctx = {
        "variant": variant,
        "manufacturer": specs.get("manufacturer", "Unknown"),
        "range_km": float(specs.get("range_km") or 0),
        "etops": "yes" if specs.get("etops_capable") else "no",
        "origin": f"{o['name']} ({origin_iata})",
        "destination": f"{d['name']} ({dest_iata})",
        "distance_km": dist,
        "verdict": "feasible" if feas["feasible"] else "not feasible",
        "prob": feas["probability"],
        "context": "",
    }
    sources: list[str] = []
    if strategy in {"rag", "rag_fewshot"}:
        hits = _retriever().search(f"{variant} range ETOPS specifications", k=4)
        ctx["context"] = "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits)
        sources = [h["title"] for h in hits]

    system, user = prompts.build(strategy, ctx)
    explanation = generate(system, user, provider=llm)

    return PipelineResult(
        cv_top5=top5,
        variant=variant,
        specs=specs,
        distance_km=dist,
        feasibility=feas,
        explanation=explanation,
        sources=sources,
    )
