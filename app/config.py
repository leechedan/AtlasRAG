from pathlib import Path

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
    allowed_index_roots: list[Path] = []

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def effective_allowed_index_roots(self) -> list[Path]:
        """Directory roots POST /directories/index is allowed to index
        under (Important Finding I11). This service is explicitly no-auth
        (intranet-only v1), so without this restriction an unauthenticated
        request with e.g. {"path": "/", "recursive": true} would walk and
        index the entire filesystem the worker can see. Defaults to just
        the app's own data_dir unless explicitly configured otherwise via
        ALLOWED_INDEX_ROOTS.
        """
        return self.allowed_index_roots or [self.data_dir]


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
