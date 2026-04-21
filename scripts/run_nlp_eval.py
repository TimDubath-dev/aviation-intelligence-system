"""Run the NLP qualitative evaluation: 20 questions × 3 strategies × 2 providers.

Outputs:
    models/nlp/eval_results.json  — all responses + scores
    models/nlp/eval_summary.json  — aggregated metrics per strategy/provider

Scoring rubric (applied by human after review):
    Faithfulness (1-5): Does the response accurately reflect the specs and verdict?
    Helpfulness (1-5):  Does it answer the user's question clearly?
    Grounding (%):      Does it cite retrieved source titles?
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp import prompts
from src.nlp.generate import generate
from src.nlp.retriever import Retriever

OUT_DIR = ROOT / "models" / "nlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS = [
    # Easy — clearly feasible
    {"variant": "A380", "manufacturer": "Airbus", "range_km": 15200, "etops": False,
     "origin": "DXB", "dest": "SYD", "distance_km": 12050, "feasible": True, "prob": 0.92},
    {"variant": "777-300", "manufacturer": "Boeing", "range_km": 11135, "etops": True,
     "origin": "LHR", "dest": "HKG", "distance_km": 9650, "feasible": True, "prob": 0.85},
    {"variant": "A330-300", "manufacturer": "Airbus", "range_km": 11750, "etops": True,
     "origin": "FRA", "dest": "JFK", "distance_km": 6200, "feasible": True, "prob": 0.97},
    {"variant": "747-400", "manufacturer": "Boeing", "range_km": 13450, "etops": False,
     "origin": "NRT", "dest": "LAX", "distance_km": 8800, "feasible": True, "prob": 0.95},
    {"variant": "Cessna 172", "manufacturer": "Cessna", "range_km": 1185, "etops": False,
     "origin": "ZRH", "dest": "BSL", "distance_km": 85, "feasible": True, "prob": 0.99},
    # Easy — clearly not feasible
    {"variant": "Cessna 172", "manufacturer": "Cessna", "range_km": 1185, "etops": False,
     "origin": "ZRH", "dest": "JFK", "distance_km": 6309, "feasible": False, "prob": 0.02},
    {"variant": "ATR-72", "manufacturer": "ATR", "range_km": 1528, "etops": False,
     "origin": "CDG", "dest": "IST", "distance_km": 2250, "feasible": False, "prob": 0.08},
    {"variant": "CRJ-200", "manufacturer": "Bombardier", "range_km": 3148, "etops": False,
     "origin": "LHR", "dest": "DXB", "distance_km": 5500, "feasible": False, "prob": 0.05},
    {"variant": "DHC-6", "manufacturer": "De Havilland Canada", "range_km": 1480, "etops": False,
     "origin": "ZRH", "dest": "ATH", "distance_km": 1600, "feasible": False, "prob": 0.15},
    {"variant": "DH-82", "manufacturer": "De Havilland", "range_km": 486, "etops": False,
     "origin": "LHR", "dest": "CDG", "distance_km": 340, "feasible": True, "prob": 0.72},
    # Medium — near the limit
    {"variant": "A320", "manufacturer": "Airbus", "range_km": 6150, "etops": True,
     "origin": "ZRH", "dest": "JFK", "distance_km": 6309, "feasible": False, "prob": 0.24},
    {"variant": "737-800", "manufacturer": "Boeing", "range_km": 5765, "etops": True,
     "origin": "LHR", "dest": "DXB", "distance_km": 5500, "feasible": True, "prob": 0.58},
    {"variant": "757-200", "manufacturer": "Boeing", "range_km": 7222, "etops": True,
     "origin": "KEF", "dest": "JFK", "distance_km": 4200, "feasible": True, "prob": 0.89},
    {"variant": "A340-500", "manufacturer": "Airbus", "range_km": 16670, "etops": False,
     "origin": "SIN", "dest": "EWR", "distance_km": 15350, "feasible": True, "prob": 0.61},
    {"variant": "767-300", "manufacturer": "Boeing", "range_km": 11070, "etops": True,
     "origin": "ORD", "dest": "NRT", "distance_km": 10150, "feasible": True, "prob": 0.55},
    # Hard / edge cases
    {"variant": "ERJ 145", "manufacturer": "Embraer", "range_km": 2873, "etops": False,
     "origin": "ZRH", "dest": "LIS", "distance_km": 1850, "feasible": True, "prob": 0.78},
    {"variant": "Fokker 100", "manufacturer": "Fokker", "range_km": 3170, "etops": False,
     "origin": "AMS", "dest": "ATH", "distance_km": 2170, "feasible": True, "prob": 0.81},
    {"variant": "MD-11", "manufacturer": "McDonnell Douglas", "range_km": 12455, "etops": False,
     "origin": "FRA", "dest": "GRU", "distance_km": 9850, "feasible": True, "prob": 0.76},
    {"variant": "DC-3", "manufacturer": "Douglas", "range_km": 2400, "etops": False,
     "origin": "LHR", "dest": "CDG", "distance_km": 340, "feasible": True, "prob": 0.95},
    {"variant": "Spitfire", "manufacturer": "Supermarine", "range_km": 756, "etops": False,
     "origin": "LHR", "dest": "CDG", "distance_km": 340, "feasible": True, "prob": 0.88},
]

STRATEGIES = ["zero_shot", "rag", "rag_fewshot"]
PROVIDERS = ["openai", "anthropic"]


def build_context(q: dict, strategy: str, retriever: Retriever) -> dict:
    ctx = {
        "variant": q["variant"],
        "manufacturer": q["manufacturer"],
        "range_km": q["range_km"],
        "etops": "yes" if q["etops"] else "no",
        "origin": q["origin"],
        "destination": q["dest"],
        "distance_km": q["distance_km"],
        "verdict": "feasible" if q["feasible"] else "not feasible",
        "prob": q["prob"],
        "context": "",
    }
    if strategy in ("rag", "rag_fewshot"):
        hits = retriever.search(f"{q['variant']} range ETOPS specifications", k=4)
        ctx["context"] = "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits)
    return ctx


def main() -> None:
    retriever = Retriever()
    results = []

    for qi, q in enumerate(QUESTIONS):
        for strategy in STRATEGIES:
            ctx = build_context(q, strategy, retriever)
            sys_msg, user_msg = prompts.build(strategy, ctx)
            for provider in PROVIDERS:
                print(f"  [{qi+1}/{len(QUESTIONS)}] {q['variant']} {q['origin']}→{q['dest']} "
                      f"| {strategy} | {provider}")
                try:
                    answer, _ = generate(sys_msg, user_msg, provider=provider)
                except Exception as e:
                    answer = f"ERROR: {e}"
                results.append({
                    "question_id": qi,
                    "variant": q["variant"],
                    "origin": q["origin"],
                    "dest": q["dest"],
                    "feasible": q["feasible"],
                    "strategy": strategy,
                    "provider": provider,
                    "response": answer,
                })
                time.sleep(0.5)

    (OUT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} responses → {OUT_DIR / 'eval_results.json'}")


if __name__ == "__main__":
    main()
