"""Sentence Transformers wrapper.

The model is loaded ONCE (lazily, on first use) and reused for both document
chunks and user queries. Using the same model for both is mandatory: vectors
from different models live in different geometric spaces, so comparing them
is meaningless (and dimensions usually differ, so Qdrant would reject them).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.errors import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Load a SentenceTransformer once and expose embed_documents / embed_query."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32, model: Any | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = model  # injectable for tests
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("Loading embedding model %s", self.model_name)
                        self._model = SentenceTransformer(self.model_name)
                    except Exception as exc:
                        raise EmbeddingError(
                            "The embedding model could not be loaded. Check your internet "
                            "connection (needed on first run to download it) and EMBEDDING_MODEL.",
                            detail=repr(exc),
                        ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        """Vector size; used to create the Qdrant collection automatically."""
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # unit length -> cosine == dot product
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [v.tolist() for v in vectors]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Failed to create embeddings for the text.", detail=repr(exc)) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunk texts (done once, at ingestion time)."""
        return self._encode(texts) if texts else []

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user question."""
        return self._encode([text])[0]
