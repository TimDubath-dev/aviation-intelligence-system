"""End-to-end smoke test of the pipeline (skips CV; uses a fixed variant).

CV is skipped because the fine-tuned 100-class FGVC model is trained on Colab
and not yet available locally. This script verifies that:
    spec lookup → distance → numeric model → RAG retrieval → LLM explanation
all run together end-to-end and that the OpenAI API key works.
"""

from __future__ import annotations

from src.nlp import prompts
from src.nlp.generate import generate
from src.nlp.retriever import Retriever
from src.numeric.predict import predict_one
from src.pipeline import lookup_airport, lookup_specs
from src.utils.geo import haversine_km

VARIANT = "A320"
ORIGIN, DEST = "ZRH", "JFK"


def main() -> None:
    print("== Spec lookup ==")
    specs = lookup_specs(VARIANT)
    print(f"  variant={VARIANT}  range_km={specs.get('range_km')}  "
          f"etops={specs.get('etops_capable')}  manuf={specs.get('manufacturer')}")

    print("\n== Distance ==")
    o, d = lookup_airport(ORIGIN), lookup_airport(DEST)
    dist = haversine_km(o["lat"], o["lon"], d["lat"], d["lon"])
    print(f"  {ORIGIN} → {DEST}: {dist:.0f} km")

    print("\n== Numeric model ==")
    feas = predict_one(
        variant=VARIANT,
        manufacturer=specs["manufacturer"],
        range_km=float(specs["range_km"]),
        twin_engine=bool(specs["twin_engine"]),
        etops_capable=bool(specs["etops_capable"]),
        distance_km=dist,
        model_name="logreg",
    )
    print(f"  feasible={feas['feasible']}  prob={feas['probability']:.3f}")

    print("\n== RAG retrieval ==")
    retr = Retriever()
    hits = retr.search(f"{VARIANT} range ETOPS specifications", k=4)
    for h in hits:
        print(f"  [{h['score']:.2f}] {h['title']}")

    print("\n== LLM explanation ==")
    ctx = {
        "variant": VARIANT,
        "manufacturer": specs["manufacturer"],
        "range_km": float(specs["range_km"]),
        "etops": "yes" if specs["etops_capable"] else "no",
        "origin": f"{o['name']} ({ORIGIN})",
        "destination": f"{d['name']} ({DEST})",
        "distance_km": dist,
        "verdict": "feasible" if feas["feasible"] else "not feasible",
        "prob": feas["probability"],
        "context": "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits),
    }
    sys, user = prompts.build("rag", ctx)
    answer, _ = generate(sys, user, provider="openai")
    print(f"\n{answer}\n")
    print("== ✅ End-to-end pipeline OK ==")


if __name__ == "__main__":
    main()
