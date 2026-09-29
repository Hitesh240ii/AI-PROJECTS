"""Plain-text loader."""

from __future__ import annotations

import logging
from pathlib import Path

from app.errors import DocumentProcessingError
from app.models.schemas import Section

logger = logging.getLogger(__name__)


def load_txt(path: Path) -> list[Section]:
    """Read a text file as UTF-8 (BOM tolerated); fall back to latin-1 if needed."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DocumentProcessingError(f"'{path.name}' could not be read.", detail=repr(exc)) from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s is not valid UTF-8; falling back to latin-1", path.name)
        text = raw.decode("latin-1")
    return [Section(text=text, page=None)] if text.strip() else []
