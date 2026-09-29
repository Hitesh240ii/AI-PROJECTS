"""Word-based sliding-window chunker.

Design choices (and why):
* Chunks are measured in WORDS, not characters, so we never cut a word in half.
* Chunking happens per Section (i.e. per PDF page), so each chunk keeps ONE
  exact page number for citations. The price: text is not merged across pages.
* Consecutive chunks overlap so a sentence that straddles a boundary still
  appears whole in at least one chunk.
* A tiny trailing window (fewer than ``min_chunk_words`` NEW words) is merged
  into the previous chunk instead of becoming its own low-value chunk.
* Chunk IDs are ``<document_id>-<index:04d>`` -> stable and deterministic:
  re-indexing the same file yields the same IDs (Qdrant upsert overwrites).

Trade-offs:
* small chunks -> precise retrieval, but less context per chunk
* large chunks -> more context, but diluted embeddings (and MiniLM truncates
  input beyond ~256 word-pieces)
* more overlap -> better boundary coverage, but more storage/duplicates
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models.schemas import ChunkMetadata, DocumentChunk, Section


class Chunker:
    """Split sections into overlapping word windows."""

    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 30, min_chunk_words: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if not 0 <= min_chunk_words < chunk_size:
            raise ValueError("min_chunk_words must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words

    def split_text(self, text: str) -> list[str]:
        """Split one block of text into overlapping chunks of words."""
        words = text.split()
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [" ".join(words)]  # whole block is one chunk, even if short

        step = self.chunk_size - self.chunk_overlap
        windows: list[tuple[int, int]] = []
        start = 0
        while True:
            end = min(start + self.chunk_size, len(words))
            windows.append((start, end))
            if end == len(words):
                break
            start += step

        # Merge a tiny tail into the previous window.
        if len(windows) > 1:
            prev_start, prev_end = windows[-2]
            _, last_end = windows[-1]
            if last_end - prev_end < self.min_chunk_words:
                windows[-2] = (prev_start, last_end)
                windows.pop()

        return [" ".join(words[s:e]) for s, e in windows]

    def chunk_sections(
        self,
        sections: list[Section],
        *,
        document_id: str,
        filename: str,
        file_type: str,
        uploaded_at: str | None = None,
    ) -> list[DocumentChunk]:
        """Chunk every section and attach metadata to each chunk."""
        timestamp = uploaded_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        chunks: list[DocumentChunk] = []
        index = 0
        for section in sections:
            for text in self.split_text(section.text):
                chunks.append(
                    DocumentChunk(
                        text=text,
                        metadata=ChunkMetadata(
                            document_id=document_id,
                            filename=filename,
                            file_type=file_type,
                            page=section.page,
                            chunk_id=f"{document_id}-{index:04d}",
                            chunk_index=index,
                            uploaded_at=timestamp,
                        ),
                    )
                )
                index += 1
        return chunks
