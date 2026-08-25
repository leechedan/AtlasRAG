import httpx
import respx

from admin_ui.app import do_delete_document, do_delete_job, do_list_jobs, do_select_document, do_select_job


class FakeSelectData:
    def __init__(self, row_index):
        self.index = [row_index]


@respx.mock
def test_do_list_jobs_returns_rows():
    respx.get("http://localhost:8000/jobs").mock(
        return_value=httpx.Response(
            200,
            json=[{"job_id": "job-1", "total": 2, "processed": 2, "success_count": 2, "failed_count": 0}],
        )
    )

    rows = do_list_jobs()

    assert rows == [["job-1", 2, 2, 2, 0]]


def test_do_select_job_returns_job_id_from_clicked_row():
    jobs_table = [["job-0", 1, 1, 1, 0], ["job-1", 2, 2, 1, 1]]

    job_id = do_select_job(jobs_table, FakeSelectData(1))

    assert job_id == "job-1"


@respx.mock
def test_do_delete_job_calls_api_and_refreshes_list():
    respx.delete("http://localhost:8000/jobs/job-1").mock(return_value=httpx.Response(204))
    respx.get("http://localhost:8000/jobs").mock(return_value=httpx.Response(200, json=[]))

    message, rows, cleared_state = do_delete_job("job-1")

    assert "job-1" in message
    assert rows == []
    assert cleared_state is None


@respx.mock
def test_do_delete_job_without_selection_does_not_call_api():
    message, *_ = do_delete_job(None)

    assert "选择" in message
    assert not respx.calls


@respx.mock
def test_do_select_document_returns_chunks_and_doc_id():
    respx.get("http://localhost:8000/documents/doc-1/chunks").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "chunk_index": 0,
                    "page": None,
                    "source_type": "normal",
                    "chunk_id": "c1",
                    "text": "hello",
                    "doc_id": "doc-1",
                    "source_path": "/a.txt",
                    "file_type": "txt",
                }
            ],
        )
    )
    documents_table = [["doc-1", "a.txt", "txt", "upload", 1, "normal"]]

    rows, doc_id = do_select_document(documents_table, FakeSelectData(0))

    assert rows == [[0, None, "normal", "c1", "hello"]]
    assert doc_id == "doc-1"


@respx.mock
def test_do_delete_document_calls_api_and_refreshes_list():
    respx.delete("http://localhost:8000/documents/doc-1").mock(
        return_value=httpx.Response(200, json={"status": "deleted"})
    )

    message, table, chunks, cleared_state = do_delete_document("doc-1")

    assert "doc-1" in message
    assert chunks == []
    assert cleared_state is None


@respx.mock
def test_do_delete_document_without_selection_does_not_call_api():
    message, *_ = do_delete_document(None)

    assert "选择" in message
    assert not respx.calls
