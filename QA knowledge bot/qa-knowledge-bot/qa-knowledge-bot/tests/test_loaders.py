import pytest
from docx import Document

from app.errors import DocumentProcessingError
from app.ingestion.docx_loader import load_docx
from app.ingestion.loader import load_document
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.txt_loader import load_txt
from tests.helpers import build_pdf


def test_pdf_keeps_page_numbers(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(build_pdf(["Leave policy text on page one.", "Remote work text on page two."]))
    sections = load_pdf(path)
    assert [s.page for s in sections] == [1, 2]
    assert "Remote work" in sections[1].text


def test_invalid_pdf_raises_friendly_error(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(DocumentProcessingError):
        load_pdf(path)


def test_txt_utf8_and_latin1_fallback(tmp_path):
    (tmp_path / "a.txt").write_text("café policy", encoding="utf-8")
    (tmp_path / "b.txt").write_bytes("café policy".encode("latin-1"))
    assert "café" in load_txt(tmp_path / "a.txt")[0].text
    assert "caf" in load_txt(tmp_path / "b.txt")[0].text


def test_empty_txt_returns_no_sections(tmp_path):
    (tmp_path / "e.txt").write_text("   \n")
    assert load_txt(tmp_path / "e.txt") == []


def test_docx_paragraphs_and_tables(tmp_path):
    doc = Document()
    doc.add_paragraph("Paragraph about leave.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Role"
    table.rows[0].cells[1].text = "Manager"
    path = tmp_path / "a.docx"
    doc.save(path)
    text = load_docx(path)[0].text
    assert "Paragraph about leave." in text and "Role | Manager" in text


def test_invalid_docx_raises(tmp_path):
    path = tmp_path / "bad.docx"
    path.write_bytes(b"not a zip")
    with pytest.raises(DocumentProcessingError):
        load_docx(path)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "x.xyz"
    path.write_text("hi")
    with pytest.raises(DocumentProcessingError):
        load_document(path)
