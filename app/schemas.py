from pydantic import BaseModel, ConfigDict, Field


class RecommendationRequest(BaseModel):
    isbns: list[str] = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "isbns": ["0618129022"],
                    "top_k": 5,
                }
            ]
        }
    )


class ResolveTitlesRequest(BaseModel):
    titles: list[str] = Field(min_length=1)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "titles": [
                        "The Lord of the Rings",
                        "1984",
                    ]
                }
            ]
        }
    )


class BookMetadataRequest(BaseModel):
    isbns: list[str] = Field(min_length=1)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "isbns": [
                        "0618129022",
                        "0451524934",
                    ]
                }
            ]
        }
    )


class BookResponse(BaseModel):
    isbn: str
    title: str
    author: str | None
    year: str | None
    publisher: str | None
    image_url: str | None
    training_interactions: int


class RecommendationResponse(BaseModel):
    isbn: str
    title: str
    rank: int
    score: float
