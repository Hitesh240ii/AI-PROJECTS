"""DOCX loader (paragraphs and table cells)."""

from __future__ import annotations

from pathlib import Path

from docx import Document

from app.errors import DocumentProcessingError
from app.models.schemas import Section


def load_docx(path: Path) -> list[Section]:
    """Extract paragraph and table text. DOCX has no fixed pages, so page=None."""
    try:
        document = Document(str(path))
        parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
    except Exception as exc:
        raise DocumentProcessingError(
            f"'{path.name}' does not look like a valid DOCX file.", detail=repr(exc)
        ) from exc
    text = "\n\n".join(parts)
    return [Section(text=text, page=None)] if text.strip() else []
