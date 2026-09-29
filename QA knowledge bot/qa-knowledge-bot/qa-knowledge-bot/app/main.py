"""Streamlit entry point:  streamlit run app/main.py"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``import app...`` work no matter where streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.config import configure_logging, get_settings  # noqa: E402
from app.errors import KnowledgeBotError  # noqa: E402
from app.services.qa_service import QAService  # noqa: E402
from app.ui.chat import render_chat  # noqa: E402
from app.ui.sidebar import render_sidebar  # noqa: E402

st.set_page_config(page_title="QA Knowledge Bot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and connecting to Qdrant…")
def get_service() -> QAService:
    """Build the pipeline ONCE per process (model + Qdrant client are reused)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    return QAService.build(settings)


def main() -> None:
    st.title("QA Knowledge Bot")
    st.caption("Ask questions about your uploaded documents.")

    try:
        service = get_service()
    except KnowledgeBotError as exc:
        st.error(exc.user_message)
        st.stop()
    except Exception:
        st.error("The application failed to start. Check the terminal log for details.")
        raise

    if not service.llm.is_configured:
        st.warning("GROQ_API_KEY is not set. You can index documents, but answering needs the key (see README).")

    params = render_sidebar(service)
    render_chat(service, params)


main()
