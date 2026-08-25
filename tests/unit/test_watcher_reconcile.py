import time

import fakeredis

from app.ingest.hashing import sha256_file
from app.ingest.watcher import (
    DirectoryWatchManager,
    list_watched_directories,
    reconcile_directory,
    register_watch,
    unregister_watch,
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


def test_unregister_watch_removes_matching_entry():
    redis_conn = fakeredis.FakeStrictRedis()
    register_watch(redis_conn, "/data/docs", recursive=True, chunk_size=500)
    register_watch(redis_conn, "/data/notes", recursive=False)

    removed = unregister_watch(redis_conn, "/data/docs")

    assert removed is True
    watched_paths = {entry["path"] for entry in list_watched_directories(redis_conn)}
    assert watched_paths == {"/data/notes"}


def test_unregister_watch_returns_false_when_not_registered():
    redis_conn = fakeredis.FakeStrictRedis()
    register_watch(redis_conn, "/data/notes", recursive=False)

    removed = unregister_watch(redis_conn, "/data/does-not-exist")

    assert removed is False
    assert len(list_watched_directories(redis_conn)) == 1


def test_watch_manager_start_skips_nonexistent_directory_without_crashing(tmp_path):
    """Regression test for Critical Finding 3: DirectoryWatchManager.start()
    previously called self._observer.schedule(...) for every registered
    directory and then self._observer.start() once at the end. A single
    bad/stale entry (e.g. a directory that was later deleted, or a typo'd
    path from before Critical Finding 3's POST /directories/index
    validation existed) made watchdog's Observer.start() raise
    FileNotFoundError for the *entire* observer -- including every other,
    perfectly valid, watched directory -- which propagated out of
    DirectoryWatchManager.start() uncaught and killed app.worker.main()
    before the RQ worker loop was ever reached. With docker-compose's
    `restart: unless-stopped`, this crash-looped the worker forever and
    stopped ALL ingestion, not just the bad directory.

    A bad entry must be skipped (logged, not raised) while the good
    directory's watch keeps working."""
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    bad_dir = tmp_path / "this-does-not-exist"

    redis_conn = fakeredis.FakeStrictRedis()
    register_watch(redis_conn, str(bad_dir), recursive=False)
    register_watch(redis_conn, str(good_dir), recursive=False)

    manager = DirectoryWatchManager(redis_conn)
    try:
        manager.start()  # must not raise
        time.sleep(0.2)
        assert manager._observer.is_alive()
    finally:
        manager.stop()
