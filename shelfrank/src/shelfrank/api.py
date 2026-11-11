"""FastAPI service for recommendations."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    history: list[str] = Field(min_length=1, max_length=50, description="item ids, oldest first")
    k: int = Field(default=10, ge=1, le=50)


def create_app(recommender=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if recommender is None:
            from .service import Recommender

            app.state.recommender = Recommender()
        else:
            app.state.recommender = recommender
        yield

    app = FastAPI(title="ShelfRank", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "items": len(app.state.recommender.items)}

    @app.get("/search")
    def search(q: str = Query(min_length=2, max_length=100)) -> list[dict]:
        return app.state.recommender.search(q)

    @app.post("/recommend")
    def recommend(request: RecommendRequest) -> list[dict]:
        results = app.state.recommender.recommend(request.history, request.k)
        if not results:
            raise HTTPException(status_code=404, detail="none of the history items are known")
        return results

    return app


app = create_app()
