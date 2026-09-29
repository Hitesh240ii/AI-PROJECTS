"""Pydantic data models shared by every layer.

Keeping these in one place means metadata has ONE definition and flows
unchanged from ingestion -> Qdrant payload -> retrieval -> answer -> UI.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class Section(BaseModel):
    """Raw text extracted by a loader. ``page`` is None for formats without pages."""

    text: str
    page: int | None = None


class ChunkMetadata(BaseModel):
    """Everything we need to attribute an answer back to its source."""

    document_id: str
    filename: str
    file_type: str
    page: int | None = None
    chunk_id: str
    chunk_index: int
    uploaded_at: str


class DocumentChunk(BaseModel):
    """A piece of text ready to be embedded, plus its metadata."""

    text: str
    metadata: ChunkMetadata

    def to_payload(self) -> dict[str, Any]:
        """Flat dict stored next to the vector in Qdrant (includes the text)."""
        return {**self.metadata.model_dump(), "text": self.text}


class RetrievedChunk(BaseModel):
    """A chunk returned by vector search, with its cosine similarity score."""

    text: str
    score: float
    metadata: ChunkMetadata

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float) -> "RetrievedChunk":
        return cls(
            text=payload["text"],
            score=score,
            metadata=ChunkMetadata.model_validate({k: v for k, v in payload.items() if k != "text"}),
        )


class SourceReference(BaseModel):
    """One numbered source shown to the user under an answer."""

    index: int
    filename: str
    page: int | None
    chunk_id: str
    score: float
    text: str
    cited: bool = False  # True if the LLM referenced "[Source n]" in its answer

    @property
    def label(self) -> str:
        """Human label: page if known, otherwise chunk id."""
        location = f"Page {self.page}" if self.page is not None else f"Chunk {self.chunk_id}"
        return f"{self.filename} — {location}"

    @property
    def preview(self) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= 160 else flat[:157] + "..."


class Answer(BaseModel):
    """Final result of a question."""

    question: str
    answer: str
    sources: list[SourceReference] = []
    llm_used: bool = False  # False when we refused before calling the LLM


class IngestResult(BaseModel):
    """Outcome of indexing one uploaded file."""

    filename: str
    status: Literal["indexed", "duplicate", "failed"]
    message: str
    document_id: str | None = None
    chunks: int = 0


class IndexedDocument(BaseModel):
    """Summary row for the sidebar document list."""

    document_id: str
    filename: str
    file_type: str
    chunks: int
    uploaded_at: str = ""
