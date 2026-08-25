import hashlib
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_doc_id(source_ref: str, content_hash: str) -> str:
    return hashlib.sha256(f"{source_ref}|{content_hash}".encode("utf-8")).hexdigest()


def upload_source_ref(filename: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    return f"upload:{ts}:{filename}"
