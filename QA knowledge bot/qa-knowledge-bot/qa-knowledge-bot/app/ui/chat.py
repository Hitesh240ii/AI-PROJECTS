"""Main chat area."""

from __future__ import annotations

import logging

import streamlit as st

from app.errors import KnowledgeBotError
from app.models.schemas import Answer
from app.services.qa_service import QAService
from app.ui.sidebar import SearchParams

logger = logging.getLogger(__name__)


def render_sources(answer: Answer) -> None:
    if not answer.sources:
        return
    st.markdown("**Sources**")
    for src in answer.sources:
        badge = "✅ cited" if src.cited else "retrieved"
        with st.expander(f"{src.index}. {src.label} · score {src.score:.2f} · {badge}"):
            st.caption(f"Chunk ID: {src.chunk_id}")
            st.write(src.text)


def render_chat(service: QAService, params: SearchParams) -> None:
    st.session_state.setdefault("messages", [])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("answer"):
                render_sources(message["answer"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer…"):
                answer = service.ask(question, top_k=params.top_k, score_threshold=params.threshold)
            st.markdown(answer.answer)
            render_sources(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer.answer, "answer": answer})
        except KnowledgeBotError as exc:
            logger.warning("Question failed: %s", exc.detail)
            st.error(exc.user_message)
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {exc.user_message}"})
        except Exception:
            logger.exception("Unexpected error while answering")
            msg = "Something went wrong while answering. Details were written to the log."
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {msg}"})
