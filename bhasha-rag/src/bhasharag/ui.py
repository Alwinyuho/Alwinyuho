"""Gradio demo. Run with: python -m bhasharag.ui"""

from __future__ import annotations

import html

import gradio as gr

from . import config
from .api import load_pipeline

pipeline = load_pipeline()
LANGUAGE_CHOICES = [(config.LANGUAGES[lang], lang) for lang in sorted(pipeline.indexes)]


def respond(question: str, lang: str) -> tuple[str, str, str]:
    question = (question or "").strip()[:1000]
    if not question:
        return "Please type a question.", "", ""
    result = pipeline.answer(question, lang)
    status = (
        f"Refused: {result.refusal_reason}" if result.refused else f"Answered using: {result.mode}"
    )
    status += f" | confidence {result.confidence:.3f}, threshold {result.threshold:.3f}"
    status += f" | {result.timings_ms.get('total', 0):.0f} ms"
    sources = "".join(
        f"<p><b>[{s.number}]</b> {html.escape(s.text[:600])}</p>" for s in result.sources
    )
    return result.answer, status, sources


with gr.Blocks(title="BhashaRAG") as demo:
    gr.Markdown(
        "# BhashaRAG\n"
        "Ask in an Indian language. The system answers from its passages and cites them, "
        "or refuses when the evidence is weak."
    )
    with gr.Row():
        question_box = gr.Textbox(label="Question", lines=2, scale=4)
        language_box = gr.Dropdown(
            LANGUAGE_CHOICES, value=LANGUAGE_CHOICES[0][1], label="Language", scale=1
        )
    ask_button = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=3)
    status_box = gr.Textbox(label="Decision")
    sources_box = gr.HTML(label="Sources")
    ask_button.click(respond, [question_box, language_box], [answer_box, status_box, sources_box])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
