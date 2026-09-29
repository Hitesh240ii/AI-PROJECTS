from app.models.schemas import ChunkMetadata, RetrievedChunk
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever
from tests.conftest import CAFETERIA_TEXT, LEAVE_TEXT
from tests.helpers import build_pdf


def _index(service):
    service.ingest_file("leave.txt", LEAVE_TEXT)
    service.ingest_file("cafeteria.txt", CAFETERIA_TEXT)


def test_retrieval_returns_most_relevant_first(service, embedder, store):
    _index(service)
    results = Retriever(embedder, store, 5, 0.2).retrieve("How many days of annual leave do employees get?")
    assert results and results[0].metadata.filename == "leave.txt"
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_empty_retrieval_for_unrelated_question(service, embedder, store):
    _index(service)
    assert Retriever(embedder, store, 5, 0.2).retrieve("What is the capital of France?") == []


def test_top_k_is_respected(service, embedder, store):
    _index(service)
    assert len(Retriever(embedder, store, 5, 0.0).retrieve("leave lunch", top_k=1)) == 1


def test_metadata_survives_pdf_to_qdrant_to_retrieval(service, embedder, store):
    pdf = build_pdf(["Cafeteria opening hours and lunch menu.", "Employees receive 20 days annual leave."])
    result = service.ingest_file("handbook.pdf", pdf)
    assert result.status == "indexed"
    hit = Retriever(embedder, store, 5, 0.2).retrieve("annual leave days")[0]
    assert hit.metadata.filename == "handbook.pdf"
    assert hit.metadata.page == 2
    assert hit.metadata.file_type == "pdf"
    assert hit.metadata.chunk_id.startswith(result.document_id)
    assert "annual leave" in hit.text


def _chunk(text: str, chunk_id: str, page=1) -> RetrievedChunk:
    meta = ChunkMetadata(document_id="d", filename="f.pdf", file_type="pdf", page=page,
                         chunk_id=chunk_id, chunk_index=0, uploaded_at="t")
    return RetrievedChunk(text=text, score=0.9, metadata=meta)


def test_context_builder_formats_sources():
    built = ContextBuilder().build([_chunk("Alpha text", "d-0000", page=5), _chunk("Beta text", "d-0001", page=None)])
    assert "[Source 1]\nFile: f.pdf\nPage: 5" in built.text
    assert "[Source 2]\nFile: f.pdf\nChunk: d-0001" in built.text
    assert len(built.chunks) == 2


def test_context_builder_removes_duplicates():
    built = ContextBuilder().build([_chunk("Same   text here", "a"), _chunk("same text here", "b"),
                                    _chunk("text", "c")])  # exact + contained duplicate
    assert len(built.chunks) == 1


def test_context_builder_respects_size_limit():
    chunks = [_chunk(f"unique{i} " * 100, f"c{i}") for i in range(10)]
    built = ContextBuilder(max_chars=2000).build(chunks)
    assert 1 <= len(built.chunks) < 10
    assert len(built.text) <= 2100


def test_context_builder_empty_input():
    assert ContextBuilder().build([]).text == ""
