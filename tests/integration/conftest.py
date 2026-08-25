import pytest

from app.store.qdrant_client import ensure_collections, get_qdrant_client


@pytest.fixture
def qdrant_test_client():
    client = get_qdrant_client(location=":memory:")
    ensure_collections(client)
    return client
