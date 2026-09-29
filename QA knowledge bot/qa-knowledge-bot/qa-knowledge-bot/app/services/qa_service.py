"""Orchestration: ingestion pipeline and grounded question answering."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.embeddings.embedding_service import EmbeddingService
from app.errors import DocumentProcessingError, KnowledgeBotError
from app.ingestion.chunker import Chunker
from app.ingestion.loader import SUPPORTED_EXTENSIONS, load_document
from app.llm.groq_client import GroqClient
from app.models.schemas import Answer, IndexedDocument, IngestResult, SourceReference
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from app.vector_store.qdrant_store import QdrantStore, create_qdrant_client

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = "I couldn't find sufficient information in the uploaded documents to answer this question."
NO_DOCUMENTS_MESSAGE = "No documents are indexed yet. Upload a PDF, DOCX or TXT file in the sidebar first."


class QAService:
    """Facade used by the UI. Holds the pipeline components."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingService,
        store: QdrantStore,
        llm: GroqClient,
        chunker: Chunker | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunker = chunker or Chunker(settings.chunk_size, settings.chunk_overlap, settings.min_chunk_words)
        self.retriever = Retriever(embedder, store, settings.top_k, settings.similarity_threshold)
        self.context_builder = ContextBuilder(settings.max_context_chars)

    @classmethod
    def build(cls, settings: Settings) -> "QAService":
        """Wire up real components (loads the embedding model, connects to Qdrant)."""
        embedder = EmbeddingService(settings.embedding_model, settings.embedding_batch_size)
        client = create_qdrant_client(settings)
        store = QdrantStore(client, settings.qdrant_collection, embedder.dimension)
        store.ensure_collection()
        llm = GroqClient(settings.groq_api_key, settings.groq_model, settings.llm_temperature, settings.llm_max_tokens)
        return cls(settings, embedder, store, llm)

    # ------------------------------------------------------------------ ingestion
    def ingest_file(self, filename: str, data: bytes) -> IngestResult:
        """Index one file. Never raises for bad input; returns a status instead."""
        safe_name = Path(filename).name or "unnamed"
        try:
            return self._ingest(safe_name, data)
        except KnowledgeBotError as exc:
            logger.warning("Ingestion of %s failed: %s", safe_name, exc.detail)
            return IngestResult(filename=safe_name, status="failed", message=exc.user_message)
        except Exception:
            logger.exception("Unexpected error while ingesting %s", safe_name)
            return IngestResult(filename=safe_name, status="failed", message="Unexpected error while processing this file.")

    def ingest_files(self, files: list[tuple[str, bytes]]) -> list[IngestResult]:
        return [self.ingest_file(name, data) for name, data in files]

    def _ingest(self, filename: str, data: bytes) -> IngestResult:
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file type '{extension or 'none'}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
            )
        if not data:
            raise DocumentProcessingError(f"'{filename}' is empty.")
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise DocumentProcessingError(f"'{filename}' is larger than the {self.settings.max_upload_mb} MB limit.")

        document_id = hashlib.sha256(data).hexdigest()[:16]  # content hash -> duplicate detection
        if self.store.document_exists(document_id):
            return IngestResult(filename=filename, status="duplicate", document_id=document_id,
                                message="Already indexed (identical content) - skipped.")

        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        saved_path = self.settings.upload_dir / f"{document_id}_{filename}"
        saved_path.write_bytes(data)
        try:
            sections = load_document(saved_path)
            chunks = self.chunker.chunk_sections(
                sections,
                document_id=document_id,
                filename=filename,
                file_type=extension.lstrip("."),
                uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            if not chunks:
                raise DocumentProcessingError(
                    f"No extractable text found in '{filename}'. Scanned/image-only PDFs are not supported (no OCR)."
                )
            if len(chunks) > self.settings.max_doc_chunks:
                raise DocumentProcessingError(
                    f"'{filename}' is too long ({len(chunks)} chunks; limit {self.settings.max_doc_chunks}). Please split it."
                )
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.store.upsert_chunks(chunks, vectors)
        except Exception:
            saved_path.unlink(missing_ok=True)
            raise
        logger.info("Indexed %s: %d chunks", filename, len(chunks))
        return IngestResult(filename=filename, status="indexed", document_id=document_id,
                            chunks=len(chunks), message=f"Indexed {len(chunks)} chunks.")

    # ------------------------------------------------------------------ answering
    def ask(self, question: str, top_k: int | None = None, score_threshold: float | None = None) -> Answer:
        """Retrieve -> (refuse if nothing relevant) -> build context -> generate -> attribute."""
        question = question.strip()
        if not question:
            raise KnowledgeBotError("Please enter a question.")
        if self.store.count() == 0:
            return Answer(question=question, answer=NO_DOCUMENTS_MESSAGE)

        retrieved = self.retriever.retrieve(question, top_k, score_threshold)
        if not retrieved:
            # Nothing above the similarity threshold: do NOT call the LLM with irrelevant context.
            return Answer(question=question, answer=NO_ANSWER_MESSAGE)

        context = self.context_builder.build(retrieved)
        answer_text = self.llm.generate(question, context.text)

        cited = {int(n) for n in re.findall(r"Source\s*(\d+)", answer_text)}
        sources = [
            SourceReference(
                index=i,
                filename=c.metadata.filename,
                page=c.metadata.page,
                chunk_id=c.metadata.chunk_id,
                score=round(c.score, 4),
                text=c.text,
                cited=i in cited,
            )
            for i, c in enumerate(context.chunks, start=1)
        ]
        return Answer(question=question, answer=answer_text, sources=sources, llm_used=True)

    # ------------------------------------------------------------------ management
    def list_documents(self) -> list[IndexedDocument]:
        return self.store.list_documents()

    def delete_document(self, document_id: str) -> None:
        self.store.delete_document(document_id)
        for path in self.settings.upload_dir.glob(f"{document_id}_*"):
            path.unlink(missing_ok=True)

    def clear_knowledge_base(self) -> None:
        self.store.clear()
        if self.settings.upload_dir.exists():
            for path in self.settings.upload_dir.iterdir():
                if path.is_file() and path.name != ".gitkeep":
                    path.unlink(missing_ok=True)
