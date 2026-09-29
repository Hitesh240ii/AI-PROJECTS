"""PDF text extraction (page by page, keeping page numbers)."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from app.errors import DocumentProcessingError
from app.models.schemas import Section

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> list[Section]:
    """Extract text from each page. Pages with no text layer are skipped."""
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentProcessingError(
                f"'{path.name}' is password-protected and cannot be read."
            )
        sections: list[Section] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # one bad page must not kill the document
                logger.warning("Could not extract page %s of %s: %s", page_number, path.name, exc)
                continue
            if text.strip():
                sections.append(Section(text=text, page=page_number))
        return sections
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            f"'{path.name}' does not look like a valid PDF file.", detail=repr(exc)
        ) from exc
