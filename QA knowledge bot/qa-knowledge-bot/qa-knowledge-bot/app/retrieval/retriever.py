"""Semantic retrieval, independent of the LLM."""

from __future__ import annotations

import logging

from app.embeddings.embedding_service import EmbeddingService
from app.errors import KnowledgeBotError
from app.models.schemas import RetrievedChunk
from app.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

MAX_TOP_K = 20


class Retriever:
    """question -> embedding -> Qdrant similarity search -> scored chunks."""

    def __init__(self, embedder: EmbeddingService, store: QdrantStore, default_top_k: int = 5, default_threshold: float = 0.30) -> None:
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def retrieve(self, query: str, top_k: int | None = None, score_threshold: float | None = None) -> list[RetrievedChunk]:
        """Return up to *top_k* chunks scoring at least *score_threshold* (best first).

        An empty list means "nothing relevant" - callers must handle it.
        """
        query = query.strip()
        if not query:
            raise KnowledgeBotError("Please enter a question.")
        k = max(1, min(top_k if top_k is not None else self.default_top_k, MAX_TOP_K))
        threshold = self.default_threshold if score_threshold is None else score_threshold
        vector = self.embedder.embed_query(query)
        results = self.store.search(vector, top_k=k, score_threshold=threshold)
        logger.info("Retrieved %d chunk(s) (k=%d, threshold=%.2f)", len(results), k, threshold)
        return results
