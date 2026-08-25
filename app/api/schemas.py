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


class UploadResponse(BaseModel):
    job_id: str


class DeleteResponse(BaseModel):
    status: str


class IndexDirectoryRequest(BaseModel):
    path: str
    recursive: bool = True
    watch: bool = False
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    separators: list[str] | None = None


class JobStatusResponse(BaseModel):
    total: int
    processed: int
    files: dict


class CollectionsStatsResponse(BaseModel):
    text_chunks_count: int
    images_count: int
