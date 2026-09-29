"""Application-specific exceptions.

Every exception carries a *user_message* (safe to show in the UI) and an
optional *detail* (technical information that is only logged, never shown).
"""

from __future__ import annotations


class KnowledgeBotError(Exception):
    """Base class for errors whose message is safe to display to end users."""

    def __init__(self, user_message: str, *, detail: str | None = None) -> None:
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail or user_message


class ConfigurationError(KnowledgeBotError):
    """Missing or invalid configuration (e.g. no GROQ_API_KEY)."""


class DocumentProcessingError(KnowledgeBotError):
    """A document could not be read, parsed or was empty."""


class EmbeddingError(KnowledgeBotError):
    """The embedding model failed to load or to encode text."""


class VectorStoreError(KnowledgeBotError):
    """Qdrant is unreachable or rejected an operation."""


class LLMError(KnowledgeBotError):
    """The Groq API call failed."""
