"""Directory watching: persist which directories to watch (Redis), a pure
reconcile function that diffs on-disk files against known content hashes,
and a `DirectoryWatchManager` that runs both a real-time watchdog Observer
and an hourly fallback full scan.

`doc_id` for any file discovered here is always recomputed by `ingest_file`
itself from `make_doc_id(source_ref, sha256_file(path))` — this module never
fabricates a `doc_id` from a placeholder content_hash. Looking up the *real*
stored `doc_id` for an on-disk path (needed to delete a removed file, or to
find its last-known content_hash for the reconcile diff) goes through
`find_doc_id_by_source_path`, which resolves it from Qdrant's `source_path`
payload field instead.
"""

import json
import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.config import settings
from app.ingest.hashing import sha256_file
from app.ingest.pipeline import (
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_TEXT_EXTS,
    ChunkOptions,
    delete_ingested_document,
    ingest_file,
)
from app.store.qdrant_client import (
    find_doc_id_by_source_path,
    get_default_client,
    get_existing_content_hash,
)

SUPPORTED_EXTS = SUPPORTED_TEXT_EXTS | SUPPORTED_IMAGE_EXTS | {".csv", ".pdf"}
WATCHED_DIRS_KEY = "watched_directories"

logger = logging.getLogger(__name__)


def register_watch(
    redis_conn,
    dir_path: str,
    recursive: bool,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> None:
    entry = {
        "path": dir_path,
        "recursive": recursive,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "separators": separators,
    }
    redis_conn.sadd(WATCHED_DIRS_KEY, json.dumps(entry))


def list_watched_directories(redis_conn) -> list[dict]:
    return [json.loads(raw) for raw in redis_conn.smembers(WATCHED_DIRS_KEY)]


def unregister_watch(redis_conn, dir_path: str) -> bool:
    """Remove a previously registered watch entry for dir_path.

    Entries are stored as JSON-encoded dicts via register_watch's `sadd`
    (the member is the full JSON string, including recursive/chunk
    options, not just the bare path) -- so a plain
    `redis_conn.srem(WATCHED_DIRS_KEY, dir_path)` would never match a
    stored member. This scans the set's members, parses each one back to a
    dict, and removes every member whose "path" equals dir_path (normally
    there's at most one, but this doesn't assume it).

    This is how a bad/stale watch entry already persisted in Redis (e.g.
    one registered before Critical Finding 3's path validation existed, or
    a directory that was later deleted) can be cleaned up without manual
    Redis surgery -- see DELETE /directories/watch.

    Returns True if at least one matching entry was removed.
    """
    removed = False
    for raw in redis_conn.smembers(WATCHED_DIRS_KEY):
        entry = json.loads(raw)
        if entry.get("path") == dir_path:
            redis_conn.srem(WATCHED_DIRS_KEY, raw)
            removed = True
    return removed


def reconcile_directory(dir_path: str, recursive: bool, known_hashes: dict[str, str]) -> dict:
    directory = Path(dir_path)
    pattern = "**/*" if recursive else "*"
    to_ingest, unchanged = [], []

    for path in directory.glob(pattern):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        current_hash = sha256_file(path)
        if known_hashes.get(str(path)) == current_hash:
            unchanged.append(str(path))
        else:
            to_ingest.append(str(path))

    return {"to_ingest": to_ingest, "unchanged": unchanged}


class _EventHandler(FileSystemEventHandler):
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None, separators: list[str] | None = None):
        self._debounce: dict[str, float] = {}
        self._chunk_options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

    def _should_handle(self, path: str) -> bool:
        if Path(path).suffix.lower() not in SUPPORTED_EXTS:
            return False
        now = time.time()
        last = self._debounce.get(path, 0)
        self._debounce[path] = now
        return now - last > 1.0

    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            ingest_file(Path(event.src_path), source_ref=event.src_path, chunk_options=self._chunk_options)

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            ingest_file(Path(event.src_path), source_ref=event.src_path, chunk_options=self._chunk_options)

    def on_deleted(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in SUPPORTED_EXTS:
            client = get_default_client()
            # The doc_id actually stored in Qdrant was computed from the file's
            # real content_hash at ingest time, which we no longer have (the file
            # is gone) — so it must be resolved via the source_path payload field
            # instead of recomputed. A path with nothing indexed under it (e.g. a
            # file that was never successfully ingested) resolves to None; skip
            # the delete rather than calling delete_document with a bogus doc_id.
            doc_id = find_doc_id_by_source_path(client, event.src_path)
            if doc_id is not None:
                delete_ingested_document(doc_id, client=client)
            # Known limitation: a scanned PDF page's image record is stored
            # under its own composite doc_id (f"{doc_id}:page{N}", source_path
            # f"{path}#page={N}") — see pipeline._ingest_scanned_pdf_page. This
            # lookup only resolves the plain path, so those per-page image
            # records are not cascaded and become orphaned in the images
            # collection when the source PDF is deleted.


class DirectoryWatchManager:
    def __init__(self, redis_conn):
        self._redis_conn = redis_conn
        self._observer = Observer()
        self._stop_event = threading.Event()

    def start(self) -> None:
        # Start the (as yet empty) observer *before* scheduling any watched
        # directory, then schedule each one individually, wrapped in its own
        # try/except. This ordering matters: watchdog's
        # BaseObserver.schedule() only starts a new watch's emitter
        # immediately -- and so only raises synchronously, per-directory --
        # when the observer is already alive (verified by reading
        # watchdog.observers.api.BaseObserver.schedule in the installed
        # watchdog package). Scheduling everything first and calling
        # self._observer.start() once at the end (the previous approach)
        # instead defers all the emitter-thread startup to that single
        # start() call, which raises FileNotFoundError for the *whole*
        # observer -- including every other, perfectly valid, watched
        # directory -- the moment any one registered path doesn't exist
        # (Critical Finding 3). Starting empty first means a bad entry can
        # be caught and skipped without taking down the others.
        self._observer.start()

        for entry in list_watched_directories(self._redis_conn):
            handler = _EventHandler(
                chunk_size=entry.get("chunk_size"),
                chunk_overlap=entry.get("chunk_overlap"),
                separators=entry.get("separators"),
            )
            try:
                self._observer.schedule(handler, entry["path"], recursive=entry["recursive"])
            except OSError as exc:
                logger.warning(
                    "skipping watch for %r: %s (register a corrected path, and remove this "
                    "one via DELETE /directories/watch)",
                    entry["path"],
                    exc,
                )
                continue

        thread = threading.Thread(target=self._reconcile_loop, daemon=True)
        thread.start()

    def _reconcile_loop(self) -> None:
        while not self._stop_event.wait(settings.watch_reconcile_interval_seconds):
            client = get_default_client()
            for entry in list_watched_directories(self._redis_conn):
                directory = Path(entry["path"])
                pattern = "**/*" if entry["recursive"] else "*"
                known_hashes = {}
                for path in directory.glob(pattern):
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
                        # Same resolution as on_deleted above: recompute the doc_id
                        # from a placeholder content_hash would never match the real
                        # stored point, so known_hashes would always end up empty and
                        # every file would be re-ingested (and re-embedded) on every
                        # hourly pass regardless of whether it actually changed.
                        doc_id = find_doc_id_by_source_path(client, str(path))
                        if doc_id is not None:
                            existing = get_existing_content_hash(client, doc_id)
                            if existing:
                                known_hashes[str(path)] = existing
                result = reconcile_directory(entry["path"], entry["recursive"], known_hashes)
                chunk_options = ChunkOptions(
                    chunk_size=entry.get("chunk_size"),
                    chunk_overlap=entry.get("chunk_overlap"),
                    separators=entry.get("separators"),
                )
                for file_path in result["to_ingest"]:
                    ingest_file(Path(file_path), source_ref=file_path, client=client, chunk_options=chunk_options)

    def stop(self) -> None:
        self._stop_event.set()
        self._observer.stop()
        self._observer.join()
