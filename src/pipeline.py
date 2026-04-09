"""End-to-end pipeline: image + route → variant → feasibility → explanation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.cv import infer as cv_infer
from src.cv import ocr as cv_ocr
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
    ocr: dict | None = None  # {registration, variant, ocr_text, used}


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
        strategy: str = "rag", llm: str = "openai",
        use_ocr_tiebreaker: bool = True) -> PipelineResult:
    # 1) CV
    top5 = cv_infer.predict(image_path, top_k=5)
    variant = top5[0]["label"]

    # 1b) Optional OCR tiebreaker — read the fuselage registration and, if it
    # corresponds to a known aircraft whose variant is in the CV top-5, prefer
    # that variant. We never override with something the CV didn't see.
    ocr_info: dict | None = None
    if use_ocr_tiebreaker:
        try:
            ocr_info = cv_ocr.detect(image_path)
            ocr_variant = ocr_info.get("variant")
            top5_labels = [r["label"] for r in top5]
            if ocr_variant and ocr_variant in top5_labels and ocr_variant != variant:
                variant = ocr_variant
                # promote it to top-1 in the displayed list
                top5 = sorted(
                    top5,
                    key=lambda r: (r["label"] != ocr_variant, -r["score"]),
                )
                ocr_info["used"] = True
            else:
                ocr_info["used"] = False
        except Exception as e:
            print(f"[pipeline] OCR step failed: {e}")
            ocr_info = None

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
        ocr=ocr_info,
    )
