from pydantic import BaseModel


class SearchTextRequest(BaseModel):
    query: str
    top_k: int = 10
    modalities: list[str] = ["text"]


class SearchHitOut(BaseModel):
    score: float
    doc_id: str
    chunk_index: int | None
    chunk_id: str
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


class RemoveWatchRequest(BaseModel):
    path: str


class JobStatusResponse(BaseModel):
    total: int
    processed: int
    files: dict


class JobSummaryOut(BaseModel):
    job_id: str
    total: int
    processed: int
    success_count: int
    failed_count: int


class CollectionsStatsResponse(BaseModel):
    text_chunks_count: int
    images_count: int


class ImageSummaryOut(BaseModel):
    doc_id: str
    thumbnail_path: str
    width: int
    height: int


class DocumentSummaryOut(BaseModel):
    doc_id: str
    source_path: str
    filename: str
    file_type: str
    origin: str | None
    chunk_count: int
    source_types: list[str]
    created_at: str | None
    content_hash: str
    related_images: list[ImageSummaryOut]


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    page: int | None
    source_type: str
    source_path: str
    file_type: str
    heading_path: str | None = None
    has_table: bool = False
    has_code_block: bool = False


class ChunksPageOut(BaseModel):
    chunks: list[ChunkOut]
    next_cursor: str | None


class EndpointInfo(BaseModel):
    path: str
    methods: list[str]
    name: str
    summary: str | None = None
    description: str | None = None
    tags: list[str] = []
    response_fields: list[str] | None = None
