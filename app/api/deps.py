from app.models.clip_model import get_clip_embedder
from app.models.embeddings import get_text_embedder
from app.models.reranker import get_reranker
from app.store.qdrant_client import get_default_client


def get_store_client():
    return get_default_client()


def get_text_embedder_dep():
    return get_text_embedder()


def get_clip_embedder_dep():
    return get_clip_embedder()


def get_reranker_dep():
    return get_reranker()
