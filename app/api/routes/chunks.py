from fastapi import APIRouter, Depends

from app.api.deps import get_store_client
from app.api.schemas import ChunkOut, ChunksPageOut
from app.store.qdrant_client import scroll_chunks

router = APIRouter()


@router.get("/chunks", response_model=ChunksPageOut)
def list_chunks(cursor: str | None = None, limit: int = 50, store=Depends(get_store_client)):
    chunks, next_cursor = scroll_chunks(store, cursor, limit)
    return ChunksPageOut(
        chunks=[
            ChunkOut(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
                text=c["text"],
                page=c.get("page"),
                source_type=c["source_type"],
                source_path=c["source_path"],
                file_type=c["file_type"],
                heading_path=c.get("heading_path"),
                has_table=c.get("has_table", False),
                has_code_block=c.get("has_code_block", False),
            )
            for c in chunks
        ],
        next_cursor=next_cursor,
    )
