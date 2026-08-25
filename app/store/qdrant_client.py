import hashlib
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import settings
from app.models.clip_model import CLIP_VECTOR_DIM
from app.models.embeddings import DenseSparseVector

TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "images"
TEXT_DENSE_DIM = 1024


def get_qdrant_client(location: str | None = None) -> QdrantClient:
    if location:
        return QdrantClient(location=location)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collections(client: QdrantClient) -> None:
    if not client.collection_exists(TEXT_COLLECTION):
        client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config={"dense": models.VectorParams(size=TEXT_DENSE_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
    if not client.collection_exists(IMAGE_COLLECTION):
        client.create_collection(
            collection_name=IMAGE_COLLECTION,
            vectors_config={"clip": models.VectorParams(size=CLIP_VECTOR_DIM, distance=models.Distance.COSINE)},
        )


@dataclass
class TextChunkRecord:
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    file_type: str
    page: int | None
    source_type: str
    content_hash: str


@dataclass
class ImageRecord:
    doc_id: str
    source_path: str
    thumbnail_path: str
    width: int
    height: int
    has_ocr_text: bool
    content_hash: str


@dataclass
class SearchHit:
    score: float
    payload: dict


def _point_id(doc_id: str, chunk_index: int) -> str:
    # ":" 分隔符是安全的：doc_id 始终来自 make_doc_id 生成的哈希值，不含冒号，
    # 因此 doc_id:chunk_index 不会出现跨字段的歧义拼接。
    return hashlib.md5(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()


def upsert_text_chunk(client: QdrantClient, record: TextChunkRecord, vector: DenseSparseVector) -> None:
    point_id = _point_id(record.doc_id, record.chunk_index)
    sparse_indices = list(vector.sparse.keys())
    sparse_values = list(vector.sparse.values())
    client.upsert(
        collection_name=TEXT_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": vector.dense,
                    "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "doc_id": record.doc_id,
                    "chunk_index": record.chunk_index,
                    "text": record.text,
                    "source_path": record.source_path,
                    "file_type": record.file_type,
                    "page": record.page,
                    "source_type": record.source_type,
                    "content_hash": record.content_hash,
                },
            )
        ],
    )


def upsert_image(client: QdrantClient, record: ImageRecord, vector: list[float]) -> None:
    point_id = _point_id(record.doc_id, 0)
    client.upsert(
        collection_name=IMAGE_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={"clip": vector},
                payload={
                    "doc_id": record.doc_id,
                    "source_path": record.source_path,
                    "thumbnail_path": record.thumbnail_path,
                    "width": record.width,
                    "height": record.height,
                    "has_ocr_text": record.has_ocr_text,
                    "content_hash": record.content_hash,
                },
            )
        ],
    )


def search_text_hybrid(client: QdrantClient, query: DenseSparseVector, top_k: int = 10) -> list[SearchHit]:
    result = client.query_points(
        collection_name=TEXT_COLLECTION,
        prefetch=[
            models.Prefetch(query=query.dense, using="dense", limit=top_k * 2),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(query.sparse.keys()), values=list(query.sparse.values())
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )
    return [SearchHit(score=p.score, payload=p.payload) for p in result.points]


def search_images(client: QdrantClient, query_vector: list[float], top_k: int = 10) -> list[SearchHit]:
    result = client.query_points(
        collection_name=IMAGE_COLLECTION, query=query_vector, using="clip", limit=top_k
    )
    return [SearchHit(score=p.score, payload=p.payload) for p in result.points]


def delete_document(client: QdrantClient, doc_id: str) -> None:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    client.delete(collection_name=TEXT_COLLECTION, points_selector=models.FilterSelector(filter=flt))
    client.delete(collection_name=IMAGE_COLLECTION, points_selector=models.FilterSelector(filter=flt))


def get_existing_content_hash(client: QdrantClient, doc_id: str) -> str | None:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    for collection in (TEXT_COLLECTION, IMAGE_COLLECTION):
        points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=1)
        if points:
            return points[0].payload["content_hash"]
    return None


def find_doc_id_by_source_path(client: QdrantClient, source_path: str) -> str | None:
    flt = models.Filter(must=[models.FieldCondition(key="source_path", match=models.MatchValue(value=source_path))])
    for collection in (TEXT_COLLECTION, IMAGE_COLLECTION):
        points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=1)
        if points:
            return points[0].payload["doc_id"]
    return None


def find_all_doc_ids_by_source_path(client: QdrantClient, source_path: str) -> set[str]:
    """Like find_doc_id_by_source_path, but returns every distinct doc_id
    found under source_path instead of just the first match. doc_id is
    content-addressed (make_doc_id(source_ref, content_hash)), so editing an
    already-indexed file's content and re-ingesting it produces a *new*
    doc_id under the same source_path -- the old doc_id's points are left
    behind unless something looks them up and deletes them. Unlike
    find_doc_id_by_source_path's `limit=1` scroll (which would only ever
    surface one of possibly several accumulated stale doc_ids), this
    paginates through every match, following the same pattern as
    list_chunks_for_document."""
    flt = models.Filter(must=[models.FieldCondition(key="source_path", match=models.MatchValue(value=source_path))])
    doc_ids: set[str] = set()
    for collection in (TEXT_COLLECTION, IMAGE_COLLECTION):
        offset = None
        while True:
            points, offset = client.scroll(collection_name=collection, scroll_filter=flt, limit=100, offset=offset)
            doc_ids.update(p.payload["doc_id"] for p in points)
            if offset is None:
                break
    return doc_ids


def list_chunks_for_document(client: QdrantClient, doc_id: str) -> list[dict]:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(collection_name=TEXT_COLLECTION, scroll_filter=flt, limit=100, offset=offset)
        payloads.extend(p.payload for p in points)
        if offset is None:
            break
    return sorted(payloads, key=lambda p: p["chunk_index"])


def get_stats(client: QdrantClient) -> dict:
    text_info = client.get_collection(TEXT_COLLECTION)
    image_info = client.get_collection(IMAGE_COLLECTION)
    return {
        "text_chunks_count": text_info.points_count,
        "images_count": image_info.points_count,
    }


@lru_cache(maxsize=1)
def get_default_client() -> QdrantClient:
    client = get_qdrant_client()
    ensure_collections(client)
    return client
