"""Prompt templates for the explainer LLM.

Three strategies are compared in the evaluation:
    1. zero_shot   — no retrieval, no examples
    2. rag         — retrieval, no examples
    3. rag_fewshot — retrieval + 2 worked examples
"""

SYSTEM = (
    "You are an aviation expert assistant. You explain whether a given aircraft "
    "could realistically fly a given route, citing specifications. You are "
    "concise (3-5 sentences), factual, and you ALWAYS finish with a one-line "
    "disclaimer that this is an educational tool, not flight-planning advice."
)

ZERO_SHOT = """\
Aircraft: {variant} ({manufacturer})
Specs: range ≈ {range_km:.0f} km, ETOPS-capable: {etops}
Route: {origin} → {destination}, great-circle distance ≈ {distance_km:.0f} km
Numeric model verdict: {verdict} (probability {prob:.2f})

Please explain whether this aircraft could fly this route and why.
"""

RAG = """\
Aircraft: {variant} ({manufacturer})
Specs: range ≈ {range_km:.0f} km, ETOPS-capable: {etops}
Route: {origin} → {destination}, great-circle distance ≈ {distance_km:.0f} km
Numeric model verdict: {verdict} (probability {prob:.2f})

Relevant background (cite the source titles in your explanation):
{context}

Please explain whether this aircraft could fly this route and why.
"""

FEW_SHOT_EXAMPLES = """\
Example 1:
Q: Could a Cessna 172 fly Zürich → New York (≈6300 km)?
A: No. The Cessna 172 has a range of roughly 1300 km — far less than the
~6300 km transatlantic distance — so it physically cannot complete the route
without multiple refuelling stops. (Educational tool, not flight-planning advice.)

Example 2:
Q: Could an Airbus A350-900 fly Zürich → Tokyo (≈9700 km)?
A: Yes. The A350-900 has a published range of about 15000 km and is ETOPS-370
certified, comfortably exceeding both the distance and any oceanic-diversion
requirements. (Educational tool, not flight-planning advice.)

"""


def build(strategy: str, ctx: dict) -> tuple[str, str]:
    """Return (system, user_message) for the chosen strategy."""
    if strategy == "zero_shot":
        return SYSTEM, ZERO_SHOT.format(**ctx)
    if strategy == "rag":
        return SYSTEM, RAG.format(**ctx)
    if strategy == "rag_fewshot":
        return SYSTEM, FEW_SHOT_EXAMPLES + RAG.format(**ctx)
    raise ValueError(f"unknown strategy: {strategy}")
