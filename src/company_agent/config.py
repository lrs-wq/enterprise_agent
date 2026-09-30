from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from pydantic import AwareDatetime, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="AI_AGENT_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3.5:9b-mlx"
    embedding_model: str = "bge-m3"
    data_dir: Path = PROJECT_ROOT / "data"
    max_file_mb: int = 100
    max_chat_request_kb: int = Field(default=64, ge=4, le=1024)
    max_upload_request_mb: int = Field(default=100, ge=1, le=1024)
    max_upload_files: int = Field(default=5, ge=1, le=50)
    max_pdf_pages: int = Field(default=300, ge=1, le=5000)
    ocr_timeout_seconds: int = Field(default=240, ge=10, le=1800)
    max_spreadsheet_rows: int = Field(default=50_000, ge=1, le=1_000_000)
    max_archive_uncompressed_mb: int = Field(default=256, ge=1, le=4096)
    max_archive_members: int = Field(default=5000, ge=1, le=100_000)
    max_document_chunks: int = Field(default=20_000, ge=1, le=200_000)
    retrieval_top_k: int = 7
    min_retrieval_score: float = 0.22
    model_context: int = 16_384
    answer_cache_enabled: bool = True
    answer_cache_max_entries: int = Field(default=128, ge=1, le=10_000)
    answer_cache_ttl_seconds: int = Field(default=300, ge=1, le=86_400)
    shared_capacity_enabled: bool = True
    chat_max_concurrency: int = Field(default=1, ge=1, le=16)
    chat_max_queue: int = Field(default=2, ge=0, le=100)
    chat_request_timeout_seconds: int = Field(default=75, ge=5, le=600)
    chat_finalize_reserve_seconds: int = Field(default=7, ge=1, le=30)
    chat_rate_limit_requests: int = Field(default=12, ge=1, le=1000)
    chat_rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    model_max_concurrency: int = Field(default=1, ge=1, le=16)
    model_max_queue: int = Field(default=4, ge=0, le=100)
    model_request_timeout_seconds: int = Field(default=65, ge=5, le=600)
    model_repair_budget_seconds: int = Field(default=30, ge=1, le=120)
    topic_classification_timeout_seconds: int = Field(default=20, ge=1, le=60)
    embedding_request_timeout_seconds: int = Field(default=90, ge=5, le=900)
    admin_task_max_concurrency: int = Field(default=1, ge=1, le=8)
    admin_task_max_queue: int = Field(default=1, ge=0, le=20)
    admin_task_timeout_seconds: int = Field(default=280, ge=30, le=7200)
    admin_auth_failure_limit: int = Field(default=5, ge=1, le=100)
    admin_auth_failure_window_seconds: int = Field(default=300, ge=1, le=86_400)
    admin_token: str = Field(default="", repr=False)
    admin_previous_token: str = Field(default="", repr=False)
    admin_previous_token_expires_at: AwareDatetime | None = Field(default=None, repr=False)
    public_base_path: str = "/company-ai"

    @property
    def admin_auth_enabled(self) -> bool:
        return len(self.admin_token) >= 24 and self.admin_token != "replace-with-a-random-token"

    @property
    def admin_previous_token_enabled(self) -> bool:
        expires_at = self.admin_previous_token_expires_at
        return (
            len(self.admin_previous_token) >= 24
            and expires_at is not None
            and expires_at.astimezone(UTC) > datetime.now(UTC)
        )

    @property
    def boundary_policy_path(self) -> Path:
        return PROJECT_ROOT / "policies" / "answer-boundary.yaml"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "knowledge.db"

    @property
    def capacity_database_path(self) -> Path:
        return self.data_dir / "runtime_capacity.db"

    @property
    def document_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def trash_dir(self) -> Path:
        return self.data_dir / "trash"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def published_media_dir(self) -> Path:
        return self.data_dir / "published_media"

    def prepare_directories(self) -> None:
        for path in (
            self.data_dir,
            self.document_dir,
            self.trash_dir,
            self.media_dir,
            self.published_media_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare_directories()
    return settings
