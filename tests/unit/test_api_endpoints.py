from fastapi.testclient import TestClient

from app.api.main import app


def test_list_endpoints_returns_registered_business_routes():
    client = TestClient(app)
    response = client.get("/endpoints")

    assert response.status_code == 200
    endpoints = response.json()

    by_path = {e["path"]: e for e in endpoints}
    assert "/health" in by_path
    assert by_path["/health"]["methods"] == ["GET"]
    assert "/search/text" in by_path
    assert by_path["/search/text"]["methods"] == ["POST"]
    assert "/documents/{doc_id}" in by_path
    assert by_path["/documents/{doc_id}"]["methods"] == ["DELETE"]


def test_list_endpoints_excludes_fastapi_internal_routes():
    client = TestClient(app)
    response = client.get("/endpoints")

    paths = {e["path"] for e in response.json()}
    assert "/openapi.json" not in paths
    assert "/docs" not in paths
    assert "/redoc" not in paths


def test_list_endpoints_sorted_by_path():
    client = TestClient(app)
    response = client.get("/endpoints")

    paths = [e["path"] for e in response.json()]
    assert paths == sorted(paths)
