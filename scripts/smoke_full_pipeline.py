"""Full end-to-end smoke test: image → CV → spec lookup → numeric → RAG → LLM."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run

IMG = ROOT / "app" / "examples" / "airbus_a320.jpg"
ORIGIN, DEST = "ZRH", "JFK"


def main() -> None:
    print(f"Image:  {IMG}")
    print(f"Route:  {ORIGIN} → {DEST}\n")
    res = run(str(IMG), ORIGIN, DEST, strategy="rag", llm="openai")
    print("CV top-5:")
    for r in res.cv_top5:
        print(f"  {r['score']:.3f}  {r['label']}")
    print(f"\nPredicted variant: {res.variant}")
    print(f"Specs: range={res.specs.get('range_km')} km  ETOPS={res.specs.get('etops_capable')}")
    print(f"Distance: {res.distance_km:.0f} km")
    print(f"Feasibility: {res.feasibility}")
    print(f"\nSources: {res.sources}")
    print(f"\nExplanation:\n{res.explanation}")


if __name__ == "__main__":
    main()
