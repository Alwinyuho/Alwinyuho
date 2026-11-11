"""FastAPI service exposing the pipeline."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generation import generator_from_env

MAX_QUESTION_CHARS = 1000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    lang: str


def load_pipeline():
    from .pipeline import BhashaRAG

    languages = os.environ.get("BHASHARAG_LANGS", "hi,ta").split(",")
    profile = os.environ.get("BHASHARAG_PROFILE", "cpu")
    return BhashaRAG(
        [lang.strip() for lang in languages if lang.strip()],
        profile=profile,
        generator=generator_from_env(),
    )


def create_app(pipeline=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline if pipeline is not None else load_pipeline()
        yield

    app = FastAPI(title="BhashaRAG", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        loaded = app.state.pipeline
        return {
            "status": "ok",
            "languages": sorted(loaded.indexes),
            "profile": loaded.profile.name,
            "generator": loaded.generator.name if loaded.generator else "extractive",
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is blank")
        if request.lang not in app.state.pipeline.indexes:
            supported = ", ".join(sorted(app.state.pipeline.indexes))
            raise HTTPException(
                status_code=400,
                detail=f"language '{request.lang}' is not loaded. Available: {supported}",
            )
        return app.state.pipeline.answer(question, request.lang).to_dict()

    return app


app = create_app()
