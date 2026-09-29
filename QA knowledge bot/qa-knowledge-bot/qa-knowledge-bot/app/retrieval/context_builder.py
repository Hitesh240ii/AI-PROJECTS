"""Turn retrieved chunks into the context block sent to the LLM."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import RetrievedChunk


@dataclass
class BuiltContext:
    """Formatted context text plus the chunks actually included (numbered 1..n)."""

    text: str
    chunks: list[RetrievedChunk]


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


class ContextBuilder:
    """Deduplicate, cap the size, and label each chunk as [Source n]."""

    def __init__(self, max_chars: int = 6000) -> None:
        self.max_chars = max_chars

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        kept: list[RetrievedChunk] = []
        kept_norm: list[str] = []
        blocks: list[str] = []
        used = 0
        for chunk in chunks:  # already sorted best-first by Qdrant
            norm = _normalise(chunk.text)
            if any(norm == k or norm in k for k in kept_norm):
                continue  # exact or contained duplicate
            block = self._format(len(kept) + 1, chunk)
            if kept and used + len(block) > self.max_chars:
                break  # budget reached; lower-ranked chunks are dropped
            kept.append(chunk)
            kept_norm.append(norm)
            blocks.append(block)
            used += len(block)
        return BuiltContext(text="\n\n".join(blocks), chunks=kept)

    @staticmethod
    def _format(number: int, chunk: RetrievedChunk) -> str:
        meta = chunk.metadata
        location = f"Page: {meta.page}" if meta.page is not None else f"Chunk: {meta.chunk_id}"
        return f"[Source {number}]\nFile: {meta.filename}\n{location}\n\n{chunk.text}"
