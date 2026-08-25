from pydantic import BaseModel


class SearchTextRequest(BaseModel):
    query: str
    top_k: int = 10
    modalities: list[str] = ["text"]


class SearchHitOut(BaseModel):
    score: float
    payload: dict


class SearchTextResponse(BaseModel):
    text_hits: list[SearchHitOut]
    image_hits: list[SearchHitOut]


class SearchImageResponse(BaseModel):
    image_hits: list[SearchHitOut]
    related_text_hits: list[SearchHitOut]
