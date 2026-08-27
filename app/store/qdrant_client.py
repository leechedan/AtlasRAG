import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
    # 全文索引，用于专有名词/术语的精确子串兜底匹配（见 search_text_hybrid
    # 里的第三路 prefetch）：向量检索对生僻实体容易语义模糊，全文索引保证
    # 只要词在原文里逐字出现就能被召回。create_payload_index 对已存在的同
    # 名同 schema 索引是幂等的，可以每次 ensure_collections 都无条件调用。
    client.create_payload_index(
        collection_name=TEXT_COLLECTION,
        field_name="text",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.MULTILINGUAL,
            min_token_len=2,
            lowercase=True,
        ),
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
    origin: str | None = None
    created_at: str | None = None
    heading_path: str | None = None
    has_table: bool = False
    has_code_block: bool = False


@dataclass
class ImageRecord:
    doc_id: str
    source_path: str
    thumbnail_path: str
    width: int
    height: int
    has_ocr_text: bool
    content_hash: str
    origin: str | None = None
    created_at: str | None = None
    image_path: str | None = None


@dataclass
class SearchHit:
    score: float
    payload: dict


def compute_chunk_id(doc_id: str, chunk_index: int) -> str:
    # ":" 分隔符是安全的：doc_id 始终来自 make_doc_id 生成的哈希值，不含冒号，
    # 因此 doc_id:chunk_index 不会出现跨字段的歧义拼接。
    return hashlib.md5(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()


def upsert_text_chunk(client: QdrantClient, record: TextChunkRecord, vector: DenseSparseVector) -> None:
    point_id = compute_chunk_id(record.doc_id, record.chunk_index)
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
                    "origin": record.origin,
                    "created_at": record.created_at,
                    "heading_path": record.heading_path,
                    "has_table": record.has_table,
                    "has_code_block": record.has_code_block,
                },
            )
        ],
    )


def upsert_image(client: QdrantClient, record: ImageRecord, vector: list[float]) -> None:
    point_id = compute_chunk_id(record.doc_id, 0)
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
                    "origin": record.origin,
                    "created_at": record.created_at,
                    "image_path": record.image_path,
                },
            )
        ],
    )


def search_text_hybrid(
    client: QdrantClient, query: DenseSparseVector, query_text: str | None = None, top_k: int = 10
) -> list[SearchHit]:
    prefetch = [
        models.Prefetch(query=query.dense, using="dense", limit=top_k * 2),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(query.sparse.keys()), values=list(query.sparse.values())
            ),
            using="sparse",
            limit=top_k * 2,
        ),
    ]
    # 第三路：精确子串匹配兜底。专有名词/术语的向量语义有时会被稀释掉，
    # 但只要原文里逐字出现该词，全文索引的 MatchText 就能把它捞回来参与
    # RRF 融合，而不是完全依赖 dense/sparse 的语义相似度排序。
    if query_text and query_text.strip():
        prefetch.append(
            models.Prefetch(
                query=query.dense,
                using="dense",
                filter=models.Filter(
                    must=[models.FieldCondition(key="text", match=models.MatchText(text=query_text))]
                ),
                limit=top_k * 2,
            )
        )
    result = client.query_points(
        collection_name=TEXT_COLLECTION,
        prefetch=prefetch,
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
    _delete_related_pdf_images(client, doc_id)


def _delete_related_pdf_images(client: QdrantClient, doc_id: str) -> None:
    """删除一份 PDF 内嵌图片的 image 记录（及其磁盘文件）。

    这些记录用组合 doc_id 存储：`f"{doc_id}:img{idx}"`（见
    `pipeline._ingest_mineru_extracted_image`），doc_id 字段本身并不等于父
    文档的 doc_id，上面按精确匹配做的 delete 捞不到它们。不清理的话就会
    变成孤儿点——list_all_documents 找不到父文档（已被删掉），就把每张
    图片当成独立文档展示出来，表现为"删除 PDF 后关联图片全部散落成单独
    的文档记录"。
    """
    prefix = f"{doc_id}:"
    stray = [p for p in _scroll_all(client, IMAGE_COLLECTION) if p.payload.get("doc_id", "").startswith(prefix)]
    if not stray:
        return
    client.delete(
        collection_name=IMAGE_COLLECTION,
        points_selector=models.PointIdsList(points=[p.id for p in stray]),
    )
    for p in stray:
        for path_field in ("image_path", "thumbnail_path"):
            path_str = p.payload.get(path_field)
            if path_str:
                Path(path_str).unlink(missing_ok=True)


def get_existing_content_hash(client: QdrantClient, doc_id: str) -> str | None:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    for collection in (TEXT_COLLECTION, IMAGE_COLLECTION):
        points, _ = client.scroll(collection_name=collection, scroll_filter=flt, limit=1)
        if points:
            return points[0].payload["content_hash"]
    return None


def find_image_by_doc_id(client: QdrantClient, doc_id: str) -> dict | None:
    flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    points, _ = client.scroll(collection_name=IMAGE_COLLECTION, scroll_filter=flt, limit=1)
    return points[0].payload if points else None


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


def _scroll_all(client: QdrantClient, collection: str) -> list:
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(collection_name=collection, limit=100, offset=offset)
        points.extend(batch)
        if offset is None:
            break
    return points


def list_all_documents(client: QdrantClient) -> list[dict]:
    text_docs: dict[str, dict] = {}
    for p in _scroll_all(client, TEXT_COLLECTION):
        payload = p.payload
        doc_id = payload["doc_id"]
        if doc_id not in text_docs:
            text_docs[doc_id] = {
                "doc_id": doc_id,
                "source_path": payload["source_path"],
                "filename": Path(payload["source_path"]).name,
                "file_type": payload["file_type"],
                "origin": payload.get("origin"),
                "chunk_count": 0,
                "source_types": set(),
                "created_at": payload.get("created_at"),
                "content_hash": payload["content_hash"],
                "related_images": [],
            }
        entry = text_docs[doc_id]
        entry["chunk_count"] += 1
        entry["source_types"].add(payload["source_type"])

    image_only_docs: dict[str, dict] = {}
    for p in _scroll_all(client, IMAGE_COLLECTION):
        payload = p.payload
        img_doc_id = payload["doc_id"]
        image_summary = {
            "doc_id": img_doc_id,
            "thumbnail_path": payload["thumbnail_path"],
            "width": payload["width"],
            "height": payload["height"],
        }

        parent_id = None
        if ":" in img_doc_id:
            candidate = img_doc_id.split(":", 1)[0]
            if candidate in text_docs:
                parent_id = candidate
        elif img_doc_id in text_docs:
            parent_id = img_doc_id

        if parent_id is not None:
            text_docs[parent_id]["related_images"].append(image_summary)
        else:
            image_only_docs[img_doc_id] = {
                "doc_id": img_doc_id,
                "source_path": payload["source_path"],
                "filename": Path(payload["source_path"]).name,
                "file_type": Path(payload["source_path"]).suffix.lstrip(".").lower(),
                "origin": payload.get("origin"),
                "chunk_count": 0,
                "source_types": [],
                "created_at": payload.get("created_at"),
                "content_hash": payload["content_hash"],
                "related_images": [image_summary],
            }

    documents = []
    for entry in text_docs.values():
        entry["source_types"] = sorted(entry["source_types"])
        documents.append(entry)
    documents.extend(image_only_docs.values())
    return sorted(documents, key=lambda d: d["doc_id"])


def scroll_chunks(client: QdrantClient, cursor: str | None = None, limit: int = 50) -> tuple[list[dict], str | None]:
    points, next_offset = client.scroll(collection_name=TEXT_COLLECTION, limit=limit, offset=cursor)
    chunks = []
    for p in points:
        payload = dict(p.payload)
        payload["chunk_id"] = compute_chunk_id(payload["doc_id"], payload["chunk_index"])
        chunks.append(payload)
    next_cursor = str(next_offset) if next_offset is not None else None
    return chunks, next_cursor


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
