"""Gradio Blocks UI for the Aviation Intelligence System.

Inference-only entrypoint — loads pre-trained artifacts from src/cv,
src/numeric, src/nlp. Deployed to Hugging Face Spaces.
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow `from src...` when launched from the repo root or app/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.pipeline import run  # noqa: E402

DISCLAIMER = (
    "⚠️ **Educational tool only — not flight-planning advice.** "
    "Predictions consider only range and ETOPS rules; payload, weather, "
    "runway length, and regulatory factors are ignored."
)


def analyse(image, origin, destination, strategy, llm):
    if image is None or not origin or not destination:
        return "Please provide an image, origin and destination.", {}, "", ""
    try:
        res = run(image, origin.strip().upper(), destination.strip().upper(),
                  strategy=strategy, llm=llm)
    except Exception as e:
        return f"Error: {e}", {}, "", ""

    cv_label = {r["label"]: float(r["score"]) for r in res.cv_top5}
    verdict = (
        f"### Verdict: {'✅ Feasible' if res.feasibility['feasible'] else '❌ Not feasible'}\n"
        f"Probability: **{res.feasibility['probability']:.2f}**  \n"
        f"Aircraft: **{res.variant}** — range ≈ {res.specs.get('range_km', 'unknown')} km  \n"
        f"Great-circle distance: **{res.distance_km:.0f} km**"
    )
    sources = "\n".join(f"- {s}" for s in res.sources) or "_(none — strategy without RAG)_"
    return verdict, cv_label, res.explanation, sources


with gr.Blocks(title="Aviation Intelligence System") as demo:
    gr.Markdown(
        "# ✈️ Aviation Intelligence System\n"
        "Upload a photo of an aircraft and pick an origin/destination — the "
        "system will identify the aircraft, predict whether it can fly the "
        "route, and explain the answer.\n\n" + DISCLAIMER
    )
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="filepath", label="Aircraft photo")
            origin = gr.Textbox(label="Origin IATA", placeholder="ZRH", max_lines=1)
            dest = gr.Textbox(label="Destination IATA", placeholder="JFK", max_lines=1)
            strategy = gr.Radio(
                ["zero_shot", "rag", "rag_fewshot"], value="rag",
                label="NLP strategy (ablation)",
            )
            llm = gr.Radio(["openai", "anthropic"], value="openai", label="LLM provider")
            btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            verdict_md = gr.Markdown()
            cv_out = gr.Label(label="Aircraft (top-5)", num_top_classes=5)
            explanation = gr.Markdown(label="Explanation")
            sources_md = gr.Markdown(label="Retrieved sources")

    btn.click(
        analyse,
        inputs=[img, origin, dest, strategy, llm],
        outputs=[verdict_md, cv_out, explanation, sources_md],
    )

    EXAMPLES_DIR = Path(__file__).parent / "examples"
    if EXAMPLES_DIR.exists():
        gr.Examples(
            examples=[
                [str(EXAMPLES_DIR / "a320.jpg"), "ZRH", "JFK"],
                [str(EXAMPLES_DIR / "a380.jpg"), "DXB", "SYD"],
                [str(EXAMPLES_DIR / "747_400.jpg"), "LHR", "HKG"],
                [str(EXAMPLES_DIR / "777_200.jpg"), "FRA", "GRU"],
                [str(EXAMPLES_DIR / "cessna_172.jpg"), "ZRH", "BSL"],
            ],
            inputs=[img, origin, dest],
        )

    gr.Markdown("---\n_Project by Tim Dubath — ZHAW FS26 AI Applications._")


if __name__ == "__main__":
    demo.launch()
