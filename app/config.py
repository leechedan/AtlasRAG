from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_url: str = "redis://localhost:6379/0"
    data_dir: Path = Path("./data")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunk_separators: list[str] = ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
    text_embed_model: str = "BAAI/bge-m3"
    clip_model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    ocr_lang: str = "ch"
    watch_reconcile_interval_seconds: int = 3600

    # Plain (settable) fields, derived from data_dir at construction time.
    # Kept as real attributes (not read-only @property) so tests can
    # monkeypatch them directly, e.g. monkeypatch.setattr(settings, "thumbnails_dir", tmp_path).
    uploads_dir: Path = Path("./data/uploads")
    thumbnails_dir: Path = Path("./data/thumbnails")

    def model_post_init(self, __context: Any) -> None:
        if "uploads_dir" not in self.model_fields_set:
            self.uploads_dir = self.data_dir / "uploads"
        if "thumbnails_dir" not in self.model_fields_set:
            self.thumbnails_dir = self.data_dir / "thumbnails"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
