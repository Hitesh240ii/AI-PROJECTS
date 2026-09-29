import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.embeddings.embedding_service import EmbeddingService  # noqa: E402
from app.services.qa_service import QAService  # noqa: E402
from app.vector_store.qdrant_store import QdrantStore  # noqa: E402
from tests.helpers import FakeLLM, FakeSentenceTransformer  # noqa: E402


@pytest.fixture()
def fake_model():
    return FakeSentenceTransformer()


@pytest.fixture()
def embedder(fake_model):
    return EmbeddingService(model=fake_model)


@pytest.fixture()
def store(embedder):
    """Real Qdrant engine, in-memory (no server, no disk)."""
    s = QdrantStore(QdrantClient(":memory:"), "test_collection", embedder.dimension)
    s.ensure_collection()
    return s


@pytest.fixture()
def llm():
    return FakeLLM()


@pytest.fixture()
def settings(tmp_path):
    return Settings(upload_dir=tmp_path / "uploads", similarity_threshold=0.2, chunk_size=50,
                    chunk_overlap=10, min_chunk_words=5)


@pytest.fixture()
def service(settings, embedder, store, llm):
    return QAService(settings, embedder, store, llm)


LEAVE_TEXT = (b"Employees are entitled to 20 days of paid annual leave each year. "
              b"Leave requests must be approved by the line manager.")
CAFETERIA_TEXT = b"The office cafeteria serves lunch from noon until two and offers vegetarian meals."
