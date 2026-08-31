from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.schemas import (
    BookMetadataRequest,
    BookResponse,
    RecommendationRequest,
    RecommendationResponse,
    ResolveTitlesRequest,
)
from bookrec.data import load_dataset
from bookrec.implicit import HistoryMLPRecommender


ROOT = Path(__file__).resolve().parents[1]
ENSEMBLE_PATH = (
    ROOT / "artifacts" / "implicit" / "history_mlp_ensemble"
)
MIN_QUERY_INTERACTIONS = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    books = load_dataset("Books.csv")

    app.state.recommender = HistoryMLPRecommender(
        ENSEMBLE_PATH,
        books=books,
        min_candidate_interactions=20,
    )

    yield

    del app.state.recommender


app = FastAPI(
    title="Book Recommender API",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    recommender = request.app.state.recommender

    return {
        "status": "ok",
        "ensemble_members": recommender.member_count,
        "candidate_count": recommender.candidate_count,
        "device": str(recommender.device),
    }


@app.post(
    "/recommendations",
    response_model=list[RecommendationResponse],
)
def recommend(
    payload: RecommendationRequest,
    request: Request,
):
    recommender = request.app.state.recommender

    try:
        return recommender.recommend_by_isbn(
            payload.isbns,
            top_k=payload.top_k,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error.args[0]),
        ) from error


@app.post("/books/resolve", response_model=list[str])
def resolve_titles(
    payload: ResolveTitlesRequest,
    request: Request,
):
    catalog = request.app.state.recommender.catalog

    try:
        return catalog.resolve_titles(
            payload.titles,
            min_training_interactions=MIN_QUERY_INTERACTIONS,
        )
    except LookupError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@app.post("/books/metadata", response_model=list[BookResponse])
def book_metadata(
    payload: BookMetadataRequest,
    request: Request,
):
    catalog = request.app.state.recommender.catalog

    try:
        return [
            book.to_dict()
            for book in catalog.get_books(payload.isbns)
        ]
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error.args[0]),
        ) from error
