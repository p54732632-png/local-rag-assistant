"""Streamlit web application for the local document RAG assistant."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Set

import streamlit as st

from ingestion import DocumentIndexer
from rag_engine import RAGEngine

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_indexer() -> DocumentIndexer:
    """Return the cached document indexer using script-relative paths."""
    return DocumentIndexer(
        upload_dir=str(UPLOAD_DIR),
        persist_dir=str(PERSIST_DIR),
    )


@st.cache_resource
def get_engine() -> RAGEngine:
    """Return the cached RAG engine using script-relative paths."""
    return RAGEngine(persist_dir=str(PERSIST_DIR))


def _initialize_session_state() -> None:
    """Initialize UI state without overwriting state across Streamlit reruns."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "indexer" not in st.session_state:
        st.session_state.indexer = None


def _get_indexed_files(indexer: DocumentIndexer) -> Set[str]:
    """Collect PDF names found locally and in the persisted Chroma collection."""
    indexed_files = {
        path.name for path in UPLOAD_DIR.glob("*.pdf") if path.is_file()
    }
    try:
        collection_data = indexer.get_vector_store().get()
        for metadata in collection_data.get("metadatas", []):
            source_file = metadata.get("source_file") if metadata else None
            if source_file:
                indexed_files.add(str(source_file))
    except Exception:
        logger.exception("Could not inspect persisted indexed documents")
    return indexed_files


def _save_uploaded_file(uploaded_file: Any) -> Path:
    """Persist an uploaded PDF under a safe, script-relative filename."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / Path(uploaded_file.name).name
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def _reset_upload_directory() -> None:
    """Remove every local upload and recreate the upload directory."""
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _render_sidebar(indexer: DocumentIndexer, engine: RAGEngine) -> str | None:
    """Render document management controls and return the selected filter."""
    with st.sidebar:
        st.title("Document Vault")
        st.sidebar.caption(f"Signed in as: **{st.user.email}**")
        if st.sidebar.button("Log out"):
            st.logout()
        if st.session_state.pop("reset_notice", False):
            st.success("All documents and the search index were cleared.")

        if st.button("Clear All Documents & Reset Index", type="secondary"):
            try:
                with st.spinner("Clearing documents and search index..."):
                    engine.clear_database()
                    _reset_upload_directory()
                st.session_state.chat_history = []
                st.session_state.indexed_files = set()
                st.session_state.processed_uploads = set()
                st.session_state.reset_notice = True
                st.rerun()
            except Exception:
                logger.exception("Could not reset the document collection")
                st.error("The document collection could not be reset.")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            processed_files = st.session_state.setdefault("processed_uploads", set())
            for uploaded_file in uploaded_files:
                upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
                if upload_key in processed_files:
                    continue
                try:
                    file_path = _save_uploaded_file(uploaded_file)
                    with st.spinner("Processing & embedding document..."):
                        vector_store = indexer.index_document(str(file_path))
                    chunk_count = vector_store.get(
                        where={"source_file": file_path.name},
                    ).get("ids", [])
                    st.success(
                        f"Indexed {file_path.name}: {len(chunk_count)} chunks stored."
                    )
                    processed_files.add(upload_key)
                except ValueError as exc:
                    st.error(str(exc))
                except (FileNotFoundError, OSError) as exc:
                    st.error(f"Could not index {uploaded_file.name}: {exc}")
                except Exception:
                    logger.exception("Unexpected upload/indexing failure")
                    st.error(f"Could not index {uploaded_file.name}.")

        indexed_files = _get_indexed_files(indexer)
        st.session_state.indexed_files = indexed_files
        st.subheader("Indexed PDFs")
        if indexed_files:
            for file_name in sorted(indexed_files):
                st.caption(f"📄 {file_name}")
        else:
            st.info("No indexed PDFs yet. Upload a document to begin.")

        filter_options = ["All Documents", *sorted(indexed_files)]
        selected_filter = st.selectbox("Search scope", filter_options)
        return None if selected_filter == "All Documents" else selected_filter


def _render_chat_history() -> None:
    """Render all previously submitted messages and their citations."""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("🔍 View Sources & Citations"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**File:** {source['source_file']}  \n"
                            f"**Page:** {source['page_number']}  \n"
                            f"**Relevance Score:** "
                            f"{source.get('relevance_score', 0.0):.2f} / 1.00"
                        )
                        st.caption(source["snippet"])


def _handle_query(engine: RAGEngine, selected_file: str | None) -> None:
    """Read and process the current chat input."""
    query = st.chat_input("Ask a question about your documents...")
    if query is None:
        return
    if not query.strip():
        st.warning("Please enter a question about your documents.")
        return
    if not st.session_state.indexed_files:
        st.warning("Upload and index at least one PDF before asking a question.")
        return

    st.session_state.chat_history.append(
        {"role": "user", "content": query, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(query)

    try:
        with st.spinner("Searching your documents..."):
            result = engine.answer_query(query, file_filter=selected_file)
    except Exception:
        logger.exception("Query processing failed")
        st.error("The question could not be processed. Check the application logs.")
        return

    if not result["sources"]:
        st.warning(result["answer"])
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("🔍 View Sources & Citations"):
                for source in result["sources"]:
                    st.markdown(
                        f"**File:** {source['source_file']}  \n"
                        f"**Page:** {source['page_number']}  \n"
                        f"**Relevance Score:** "
                        f"{source.get('relevance_score', 0.0):.2f} / 1.00"
                    )
                    st.caption(source["snippet"])

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Local RAG Assistant",
        page_icon="📄",
        layout="wide",
    )
    if not st.user.is_logged_in:
        st.title("🔒 TalkToDock - Access Restricted")
        st.info("Sign in with your Google account to access your document assistant.")
        if st.button("Log in with Google", type="primary"):
            st.login("google")
        st.stop()

    _initialize_session_state()

    try:
        if st.session_state.indexer is None:
            st.session_state.indexer = get_indexer()
        if st.session_state.engine is None:
            st.session_state.engine = get_engine()
    except EnvironmentError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        logger.exception("Could not initialize RAG services")
        st.error("The RAG services could not be initialized. Check the application logs.")
        st.stop()

    selected_file = _render_sidebar(
        st.session_state.indexer,
        st.session_state.engine,
    )
    st.title("Local RAG Assistant")
    st.caption("Ask grounded questions about the PDFs in your Document Vault.")
    _render_chat_history()
    _handle_query(st.session_state.engine, selected_file)


if __name__ == "__main__":
    main()