"""Sidebar: upload, document list, retrieval settings, clear KB."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.errors import KnowledgeBotError
from app.ingestion.loader import SUPPORTED_EXTENSIONS
from app.services.qa_service import QAService


@dataclass
class SearchParams:
    top_k: int
    threshold: float


def render_sidebar(service: QAService) -> SearchParams:
    settings = service.settings
    with st.sidebar:
        st.header("📄 Documents")
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        files = st.file_uploader(
            "Upload PDF, DOCX or TXT files",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Index documents", type="primary", disabled=not files, use_container_width=True):
            with st.spinner("Indexing documents (extracting, chunking, embedding, storing)…"):
                results = service.ingest_files([(f.name, f.getvalue()) for f in files])
            st.session_state.ingest_results = results
            st.session_state.uploader_key += 1  # reset the uploader widget
            st.rerun()

        for result in st.session_state.pop("ingest_results", []):
            text = f"**{result.filename}**: {result.message}"
            if result.status == "indexed":
                st.success(text)
            elif result.status == "duplicate":
                st.info(text)
            else:
                st.error(text)

        st.subheader("Indexed documents")
        try:
            docs = service.list_documents()
        except KnowledgeBotError as exc:
            st.error(exc.user_message)
            docs = []
        if not docs:
            st.caption("Nothing indexed yet.")
        else:
            st.caption(f"{len(docs)} document(s) · {sum(d.chunks for d in docs)} chunks")
            for doc in docs:
                col_name, col_btn = st.columns([5, 1])
                col_name.markdown(f"**{doc.filename}**  \n`{doc.chunks} chunks`")
                if col_btn.button("🗑", key=f"del_{doc.document_id}", help="Remove this document"):
                    try:
                        service.delete_document(doc.document_id)
                    except KnowledgeBotError as exc:
                        st.error(exc.user_message)
                    st.rerun()

        st.divider()
        st.header("⚙️ Retrieval settings")
        top_k = st.slider("Top-K chunks", 1, 10, settings.top_k, help="How many chunks to retrieve from Qdrant.")
        threshold = st.slider(
            "Similarity threshold", 0.0, 0.9, float(settings.similarity_threshold), 0.05,
            help="Chunks scoring below this cosine similarity are discarded. Higher = stricter.",
        )

        st.divider()
        confirm = st.checkbox("I want to delete everything")
        if st.button("Clear knowledge base", disabled=not confirm, use_container_width=True):
            try:
                service.clear_knowledge_base()
                st.session_state.messages = []
                st.success("Knowledge base cleared.")
            except KnowledgeBotError as exc:
                st.error(exc.user_message)
            st.rerun()

        st.caption(f"Model: `{settings.groq_model}` · Embeddings: `{settings.embedding_model}`")
    return SearchParams(top_k=top_k, threshold=threshold)
