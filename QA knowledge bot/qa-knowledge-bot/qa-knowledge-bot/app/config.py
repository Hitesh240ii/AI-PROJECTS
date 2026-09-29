"""Environment-based configuration.

All tunable values live in :class:`Settings`. They are read from environment
variables (optionally loaded from a ``.env`` file) so nothing is hardcoded and
secrets never appear in source control.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from app.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


def _read_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name) or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Defaults are sensible for local development."""

    # LLM (Groq)
    groq_api_key: str = ""
    groq_model: str = DEFAULT_GROQ_MODEL
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Qdrant: if qdrant_url is empty, embedded local storage at qdrant_path is used.
    qdrant_url: str = ""
    qdrant_path: Path = PROJECT_ROOT / "qdrant_storage"
    qdrant_collection: str = "knowledge_base"

    # Storage
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # Chunking (measured in WORDS). all-MiniLM-L6-v2 truncates input at 256
    # word-pieces (~190 English words), so the default chunk stays below that.
    chunk_size: int = 180
    chunk_overlap: int = 30
    min_chunk_words: int = 20

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.30
    max_context_chars: int = 6000

    # Limits
    max_upload_mb: int = 20
    max_doc_chunks: int = 5000

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        return cls(
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            groq_model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL,
            llm_temperature=_read_float("LLM_TEMPERATURE", 0.1),
            llm_max_tokens=_read_int("LLM_MAX_TOKENS", 700),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip(),
            embedding_batch_size=_read_int("EMBEDDING_BATCH_SIZE", 32),
            qdrant_url=os.getenv("QDRANT_URL", "").strip(),
            qdrant_path=_read_path("QDRANT_PATH", "qdrant_storage"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "knowledge_base").strip(),
            upload_dir=_read_path("UPLOAD_DIR", "data/uploads"),
            chunk_size=_read_int("CHUNK_SIZE", 180),
            chunk_overlap=_read_int("CHUNK_OVERLAP", 30),
            min_chunk_words=_read_int("MIN_CHUNK_WORDS", 20),
            top_k=_read_int("TOP_K", 5),
            similarity_threshold=_read_float("SIMILARITY_THRESHOLD", 0.30),
            max_context_chars=_read_int("MAX_CONTEXT_CHARS", 6000),
            max_upload_mb=_read_int("MAX_UPLOAD_MB", 20),
            max_doc_chunks=_read_int("MAX_DOC_CHUNKS", 5000),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (read once)."""
    return Settings.from_env()


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging once. Technical details go here, not to the UI."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
