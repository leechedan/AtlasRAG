import fakeredis

from app.ingest.hashing import sha256_file
from app.ingest.watcher import (
    list_watched_directories,
    reconcile_directory,
    register_watch,
)


def test_register_and_list_watched_directories():
    redis_conn = fakeredis.FakeStrictRedis()
    register_watch(redis_conn, "/data/docs", recursive=True, chunk_size=500, chunk_overlap=50, separators=["\n\n"])
    register_watch(redis_conn, "/data/notes", recursive=False)

    watched = list_watched_directories(redis_conn)

    assert {
        "path": "/data/docs", "recursive": True,
        "chunk_size": 500, "chunk_overlap": 50, "separators": ["\n\n"],
    } in watched
    assert {
        "path": "/data/notes", "recursive": False,
        "chunk_size": None, "chunk_overlap": None, "separators": None,
    } in watched


def test_reconcile_directory_detects_new_and_changed_files(tmp_path):
    f1 = tmp_path / "a.txt"
    f1.write_text("original")
    f2 = tmp_path / "b.txt"
    f2.write_text("unchanged")

    known_hashes = {
        str(f2): sha256_file(f2),
        str(f1): "stale-hash-not-matching",
    }

    result = reconcile_directory(str(tmp_path), recursive=False, known_hashes=known_hashes)

    assert str(f1) in result["to_ingest"]
    assert str(f2) in result["unchanged"]


def test_reconcile_directory_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "a.xyz").write_text("data")

    result = reconcile_directory(str(tmp_path), recursive=False, known_hashes={})

    assert result["to_ingest"] == []
    assert result["unchanged"] == []
