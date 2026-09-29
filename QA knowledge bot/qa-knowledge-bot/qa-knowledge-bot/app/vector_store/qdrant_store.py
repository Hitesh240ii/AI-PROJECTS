"""Qdrant persistence layer.

Every point = vector + payload. The payload holds the chunk text AND its
metadata, so a search hit contains everything needed for the answer and the
citations without any second lookup.
"""

from __future__ import annotations

import functools
import logging
import uuid
from collections import Counter
from typing import Any, Callable, TypeVar

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.errors import KnowledgeBotError, VectorStoreError
from app.models.schemas import DocumentChunk, IndexedDocument, RetrievedChunk

logger = logging.getLogger(__name__)

_NAMESPACE = uuid.UUID("6f1c1a52-8f0e-4b5e-9c55-1d6a5f0f7a11")
_UPSERT_BATCH = 64
F = TypeVar("F", bound=Callable[..., Any])


def _guard(func: F) -> F:
    """Convert unexpected Qdrant errors into a user-friendly VectorStoreError."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except KnowledgeBotError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                "Could not talk to the Qdrant vector database. If you use QDRANT_URL, make sure "
                "Qdrant is running; if you use local storage, make sure no other app instance "
                "is using the same qdrant_storage folder.",
                detail=repr(exc),
            ) from exc

    return wrapper  # type: ignore[return-value]


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """Remote Qdrant if QDRANT_URL is set, otherwise embedded local storage."""
    try:
        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url, timeout=15)
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_path))
    except Exception as exc:
        raise VectorStoreError(
            "Could not open the Qdrant vector database. Is another instance of the app "
            "using the same qdrant_storage folder, or is QDRANT_URL wrong?",
            detail=repr(exc),
        ) from exc


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs/ints; derive a stable UUID from our chunk id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantStore:
    """Thin, purpose-built wrapper around QdrantClient."""

    def __init__(self, client: QdrantClient, collection: str, vector_size: int) -> None:
        self.client = client
        self.collection = collection
        self.vector_size = vector_size

    @_guard
    def ensure_collection(self) -> None:
        """Create the collection (cosine distance) or verify its dimension matches."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )
            logger.info("Created collection %s (dim=%s)", self.collection, self.vector_size)
            return
        existing = self.client.get_collection(self.collection).config.params.vectors
        existing_size = getattr(existing, "size", None)
        if existing_size is not None and existing_size != self.vector_size:
            raise VectorStoreError(
                f"The existing Qdrant collection '{self.collection}' uses {existing_size}-dimensional "
                f"vectors but the current embedding model produces {self.vector_size}. Use 'Clear "
                "knowledge base' in the sidebar or change QDRANT_COLLECTION."
            )

    @_guard
    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        """Insert (or overwrite, thanks to stable ids) chunks with their vectors."""
        if len(chunks) != len(vectors):
            raise VectorStoreError("Internal error: chunk/vector count mismatch.")
        points = [
            models.PointStruct(id=point_id(c.metadata.chunk_id), vector=v, payload=c.to_payload())
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(collection_name=self.collection, points=points[i : i + _UPSERT_BATCH])

    @_guard
    def search(self, query_vector: list[float], top_k: int, score_threshold: float | None = None) -> list[RetrievedChunk]:
        """Top-K nearest chunks by cosine similarity, best first."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [RetrievedChunk.from_payload(p.payload or {}, p.score) for p in response.points]

    @staticmethod
    def _doc_filter(document_id: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        )

    @_guard
    def document_exists(self, document_id: str) -> bool:
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=self._doc_filter(document_id),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    @_guard
    def list_documents(self) -> list[IndexedDocument]:
        """Aggregate stored chunks into one row per document."""
        counts: Counter[str] = Counter()
        info: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=["document_id", "filename", "file_type", "uploaded_at"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("document_id", "")
                counts[doc_id] += 1
                info.setdefault(doc_id, payload)
            if offset is None:
                break
        docs = [
            IndexedDocument(
                document_id=doc_id,
                filename=info[doc_id].get("filename", "?"),
                file_type=info[doc_id].get("file_type", ""),
                chunks=counts[doc_id],
                uploaded_at=info[doc_id].get("uploaded_at", ""),
            )
            for doc_id in counts
        ]
        return sorted(docs, key=lambda d: (d.uploaded_at, d.filename))

    @_guard
    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    @_guard
    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._doc_filter(document_id)),
        )

    @_guard
    def clear(self) -> None:
        """Drop and recreate the collection (removes everything)."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure_collection()
