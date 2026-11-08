"""Application settings loaded from environment variables and an optional .env file."""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Runtime configuration for the document assistant."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    google_api_key: str | None = None
    llm_model: str
    judge_model: str | None = None
    embedding_model: str
    chunk_size: int = Field(default=500, ge=100)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=6, ge=1)
    top_k_retry: int = Field(default=10, ge=1)
    max_retries: int = Field(default=1, ge=0)
    chroma_dir: str = "./chroma_db"
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @model_validator(mode="after")
    def set_judge_model(self) -> "Settings":
        if not self.judge_model:
            self.judge_model = self.llm_model
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


settings = Settings()
