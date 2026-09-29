import pytest

from app.ingestion.chunker import Chunker
from app.models.schemas import Section


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_single_chunk():
    assert Chunker(50, 10, 5).split_text(words(20)) == [words(20)]


def test_empty_text_gives_no_chunks():
    assert Chunker().split_text("   \n  ") == []


def test_chunk_size_and_overlap():
    chunks = Chunker(50, 10, 5).split_text(words(120))
    assert all(len(c.split()) <= 55 for c in chunks)
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-10:] == second[:10]  # 10-word overlap


def test_all_words_are_covered():
    chunks = Chunker(50, 10, 5).split_text(words(137))
    seen = {w for c in chunks for w in c.split()}
    assert seen == set(words(137).split())


def test_tiny_tail_is_merged():
    # 100 words, size 50, step 40 -> windows (0,50),(40,90),(80,100); tail adds 10 new words (<15) -> merged
    chunks = Chunker(50, 10, 15).split_text(words(100))
    assert len(chunks) == 2
    assert chunks[-1].split()[-1] == "w99"


def test_metadata_and_stable_ids():
    sections = [Section(text=words(120), page=1), Section(text=words(30), page=2)]
    kwargs = dict(document_id="abc123", filename="a.pdf", file_type="pdf", uploaded_at="2025-01-01T00:00:00")
    chunks = Chunker(50, 10, 5).chunk_sections(sections, **kwargs)
    again = Chunker(50, 10, 5).chunk_sections(sections, **kwargs)
    assert [c.metadata.chunk_id for c in chunks] == [c.metadata.chunk_id for c in again]
    assert chunks[0].metadata.chunk_id == "abc123-0000"
    assert {c.metadata.page for c in chunks} == {1, 2}
    assert chunks[-1].metadata.page == 2
    assert all(c.metadata.filename == "a.pdf" and c.metadata.document_id == "abc123" for c in chunks)
    assert len({c.metadata.chunk_id for c in chunks}) == len(chunks)


@pytest.mark.parametrize("size,overlap,minimum", [(0, 0, 0), (10, 10, 0), (10, -1, 0), (10, 2, 10)])
def test_invalid_parameters(size, overlap, minimum):
    with pytest.raises(ValueError):
        Chunker(size, overlap, minimum)
