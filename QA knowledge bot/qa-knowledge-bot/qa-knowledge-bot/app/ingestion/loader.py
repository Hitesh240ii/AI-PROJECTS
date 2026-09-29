"""Dispatch a file to the right loader based on its extension."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.errors import DocumentProcessingError
from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.txt_loader import load_txt
from app.models.schemas import Section

LOADERS: dict[str, Callable[[Path], list[Section]]] = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".docx": load_docx,
}
SUPPORTED_EXTENSIONS = tuple(LOADERS)


def load_document(path: Path) -> list[Section]:
    """Return extracted sections for *path* or raise DocumentProcessingError."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise DocumentProcessingError(
            f"Unsupported file type '{path.suffix}'. Supported: "
            + ", ".join(SUPPORTED_EXTENSIONS)
        )
    return loader(path)
