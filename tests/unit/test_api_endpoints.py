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


def test_list_endpoints_summary_falls_back_to_titleized_function_name():
    client = TestClient(app)
    response = client.get("/endpoints")

    by_path = {e["path"]: e for e in response.json()}
    # health() has no explicit summary=... and no docstring, so this must
    # match FastAPI's own /docs fallback (title-cased function name) rather
    # than the route object's raw (unset, None) .summary attribute.
    assert by_path["/health"]["summary"] == "Health"


def test_list_endpoints_reports_response_model_field_names():
    client = TestClient(app)
    response = client.get("/endpoints")

    by_path = {e["path"]: e for e in response.json()}
    # /collections/stats declares response_model=CollectionsStatsResponse.
    assert set(by_path["/collections/stats"]["response_fields"]) == {
        "text_chunks_count",
        "images_count",
    }
    # /documents declares response_model=list[DocumentSummaryOut] -- the
    # list[...] wrapper must be unwrapped to reach the model's field names.
    assert "doc_id" in by_path["/documents"]["response_fields"]
    # /health has no response_model at all -- nothing to introspect.
    assert by_path["/health"]["response_fields"] is None


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
