import io

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image

from app.api.deps import get_clip_embedder_dep, get_store_client, get_text_embedder_dep
from app.api.schemas import (
    SearchHitOut,
    SearchImageResponse,
    SearchTextRequest,
    SearchTextResponse,
)
from app.store.qdrant_client import (
    list_chunks_for_document,
    search_images,
    search_text_hybrid,
)

router = APIRouter()

# Sentinel score for related-text hits returned by search_image: these come
# from a direct doc_id lookup (list_chunks_for_document), not a similarity
# ranking, so there is no meaningful "score" to report. 1.0 signals "exact
# association" without implying a comparable relevance ranking against the
# image_hits' real similarity scores.
RELATED_TEXT_SENTINEL_SCORE = 1.0


@router.post("/search/text", response_model=SearchTextResponse)
def search_text(
    request: SearchTextRequest,
    store=Depends(get_store_client),
    text_embedder=Depends(get_text_embedder_dep),
    clip_embedder=Depends(get_clip_embedder_dep),
):
    text_hits = []
    image_hits = []

    if "text" in request.modalities:
        query_vector = text_embedder.embed([request.query])[0]
        text_hits = search_text_hybrid(store, query_vector, top_k=request.top_k)

    if "image" in request.modalities:
        clip_query_vector = clip_embedder.embed_text(request.query)
        image_hits = search_images(store, clip_query_vector, top_k=request.top_k)

    return SearchTextResponse(
        text_hits=[SearchHitOut(score=h.score, payload=h.payload) for h in text_hits],
        image_hits=[SearchHitOut(score=h.score, payload=h.payload) for h in image_hits],
    )


@router.post("/search/image", response_model=SearchImageResponse)
async def search_image(
    image: UploadFile = File(...),
    top_k: int = 10,
    store=Depends(get_store_client),
    clip_embedder=Depends(get_clip_embedder_dep),
):
    data = await image.read()
    pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    query_vector = clip_embedder.embed_image(pil_image)
    image_hits = search_images(store, query_vector, top_k=top_k)

    # For each returned image hit, look up its associated OCR text chunks
    # (written under the same doc_id for standalone images — see
    # _ingest_image_file in app/ingest/pipeline.py) and flatten them into a
    # single related_text_hits list. We query per-hit (not just the top hit)
    # so that OCR text associated with any of the returned images is
    # surfaced, not only the single best match. Known limitation: a scanned
    # PDF page's image record uses a composite doc_id
    # (f"{doc_id}:page{N}"), which doesn't match the parent PDF's plain
    # doc_id used for its text_chunks records, so list_chunks_for_document
    # correctly returns [] for those hits rather than finding the page's
    # OCR text (mirrors the same known gap in Task 14's on_deleted handler).
    related_text_hits: list[SearchHitOut] = []
    for hit in image_hits:
        doc_id = hit.payload.get("doc_id")
        if not doc_id:
            continue
        for chunk in list_chunks_for_document(store, doc_id):
            related_text_hits.append(SearchHitOut(score=RELATED_TEXT_SENTINEL_SCORE, payload=chunk))

    return SearchImageResponse(
        image_hits=[SearchHitOut(score=h.score, payload=h.payload) for h in image_hits],
        related_text_hits=related_text_hits,
    )
