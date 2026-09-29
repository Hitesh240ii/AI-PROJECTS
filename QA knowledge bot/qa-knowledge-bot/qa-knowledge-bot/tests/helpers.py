"""Test utilities: a fake SentenceTransformer, a fake LLM and a tiny PDF builder."""

from __future__ import annotations

import hashlib
import re
import textwrap

import numpy as np

_STOP = {"the", "a", "an", "of", "is", "are", "to", "in", "on", "for", "and", "or", "how", "what", "who",
         "many", "much", "do", "does", "can", "per", "be", "by", "must", "this", "that", "with", "at", "it"}


class FakeSentenceTransformer:
    """Deterministic bag-of-words hashing model: same words -> similar vectors."""

    DIM = 256

    def __init__(self) -> None:
        self.encode_calls = 0

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM

    def encode(self, sentences, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True):
        self.encode_calls += 1
        items = [sentences] if isinstance(sentences, str) else list(sentences)
        vectors = np.zeros((len(items), self.DIM), dtype=np.float32)
        for row, text in enumerate(items):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if token not in _STOP:
                    vectors[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors


class FakeLLM:
    """Records calls and returns a canned answer."""

    is_configured = True

    def __init__(self, reply: str = "Employees receive 20 days of annual leave [Source 1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        return self.reply


def build_pdf(pages: list[str]) -> bytes:
    """Create a minimal valid PDF with one text block per page (no external libs)."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        lines = textwrap.wrap(text, 85) or [""]
        escaped = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in lines]
        body = "BT /F1 11 Tf 14 TL 50 740 Td " + " T* ".join(f"({ln}) Tj" for ln in escaped) + " ET"
        objs.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out
