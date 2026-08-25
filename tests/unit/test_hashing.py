from pathlib import Path

from app.ingest.hashing import make_doc_id, sha256_bytes, sha256_file, upload_source_ref


def test_sha256_bytes_is_deterministic():
    assert sha256_bytes(b"hello") == sha256_bytes(b"hello")
    assert sha256_bytes(b"hello") != sha256_bytes(b"world")


def test_sha256_file_matches_bytes(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"content")
    assert sha256_file(f) == sha256_bytes(b"content")


def test_make_doc_id_deterministic_and_source_sensitive():
    h = sha256_bytes(b"same content")
    id_a = make_doc_id("/data/a.txt", h)
    id_b = make_doc_id("/data/b.txt", h)
    id_a_again = make_doc_id("/data/a.txt", h)
    assert id_a == id_a_again
    assert id_a != id_b


def test_upload_source_ref_contains_filename():
    ref = upload_source_ref("report.pdf")
    assert ref.startswith("upload:")
    assert ref.endswith(":report.pdf")
