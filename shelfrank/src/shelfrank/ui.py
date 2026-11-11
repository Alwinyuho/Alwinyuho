"""Gradio demo. Run with: python -m shelfrank.ui"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from .service import Recommender

recommender = Recommender()


def find(query: str):
    hits = recommender.search(query or "")
    choices = [(f"{h['title'][:90]} ({h['asin']})", h["asin"]) for h in hits]
    return gr.update(choices=choices, value=[])


def recommend(selected: list[str]):
    if not selected:
        return pd.DataFrame({"message": ["Pick at least one game you liked."]})
    rows = recommender.recommend(selected, k=10)
    return pd.DataFrame(
        [
            {
                "title": r["title"][:100],
                "new item (no history)": "yes" if r["new_item"] else "",
                "score": round(r["score"], 3),
            }
            for r in rows
        ]
    )


with gr.Blocks(title="ShelfRank") as demo:
    gr.Markdown(
        "# ShelfRank\nSearch for video games you liked, select them, and get recommendations. "
        "Items marked *new* have no purchase history: they are found through their text alone."
    )
    query = gr.Textbox(label="Search titles", placeholder="e.g. zelda, mario kart, fifa")
    picks = gr.CheckboxGroup(label="Games you liked", choices=[])
    button = gr.Button("Recommend", variant="primary")
    output = gr.Dataframe(label="Recommendations")
    query.submit(find, query, picks)
    button.click(recommend, picks, output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862)
