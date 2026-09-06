"""Streamlit web application for the local document RAG assistant."""

from __future__ import annotations

import html
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


def _inject_design_system() -> None:
    """Apply the TalkToDock visual language to Streamlit's native controls."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --td-ink: #0f172a;
            --td-muted: #64748b;
            --td-line: rgba(226, 232, 240, 0.82);
            --td-indigo: #4f46e5;
            --td-violet: #7c3aed;
            --td-soft: #f8fafc;
            --td-glass: rgba(255, 255, 255, 0.78);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--td-ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 85% 0%, rgba(124, 58, 237, 0.10), transparent 28rem),
                radial-gradient(circle at 15% 20%, rgba(79, 70, 229, 0.07), transparent 24rem),
                var(--td-soft);
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.90);
            border-right: 1px solid var(--td-line);
        }
        [data-testid="stSidebarContent"] { padding: 1.35rem 1.1rem 2rem; }
        [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 3.1rem; }

        .td-kicker {
            color: var(--td-violet);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .td-hero { margin: 0.3rem 0 2.6rem; }
        .td-hero h1 {
            color: var(--td-ink);
            font-size: clamp(2.35rem, 5vw, 4.35rem);
            font-weight: 800;
            letter-spacing: -0.055em;
            line-height: 0.98;
            margin: 0.45rem 0 0.95rem;
        }
        .td-gradient-text {
            background: linear-gradient(110deg, var(--td-indigo), var(--td-violet));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .td-badge, .td-pill {
            background: linear-gradient(110deg, var(--td-indigo), var(--td-violet));
            border-radius: 999px;
            color: white;
            display: inline-flex;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 0.34rem 0.72rem;
            vertical-align: middle;
        }
        .td-hero p { color: var(--td-muted); font-size: 1.08rem; max-width: 650px; }

        .td-card {
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            background: var(--td-glass);
            border: 1px solid var(--td-line);
            border-radius: 18px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        }
        .td-onboarding { margin: 1rem 0 2rem; padding: 1.35rem 1.45rem; }
        .td-onboarding h3 { margin: 0 0 0.25rem; }
        .td-onboarding > p { color: var(--td-muted); margin: 0 0 1.1rem; }
        .td-steps { display: grid; gap: 0.7rem; grid-template-columns: repeat(3, 1fr); }
        .td-step { background: rgba(248, 250, 252, 0.82); border: 1px solid var(--td-line); border-radius: 12px; padding: 0.9rem; }
        .td-step strong { display: block; font-size: 0.9rem; margin-top: 0.35rem; }
        .td-step span { color: var(--td-muted); font-size: 0.78rem; }
        .td-step-number { color: var(--td-violet); font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; font-weight: 600; }

        .td-profile { align-items: center; background: linear-gradient(135deg, rgba(79, 70, 229, 0.10), rgba(124, 58, 237, 0.06)); border: 1px solid rgba(99, 102, 241, 0.18); border-radius: 15px; display: flex; gap: 0.7rem; margin: 0.3rem 0 1rem; padding: 0.78rem; }
        .td-avatar { align-items: center; background: linear-gradient(135deg, var(--td-indigo), var(--td-violet)); border-radius: 50%; color: white; display: flex; flex: 0 0 2rem; font-size: 0.85rem; font-weight: 700; height: 2rem; justify-content: center; }
        .td-profile-email { color: var(--td-ink); font-size: 0.76rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .td-status { color: #16a34a; display: block; font-size: 0.68rem; margin-top: 0.12rem; }
        .td-sidebar-heading { color: var(--td-ink); font-size: 1.15rem; font-weight: 800; letter-spacing: -0.03em; margin: 1.1rem 0 0.8rem; }
        .td-dropzone-label { color: var(--td-muted); font-size: 0.78rem; margin: 0.2rem 0 0.4rem; }
        [data-testid="stFileUploader"] { background: rgba(248, 250, 252, 0.72); border: 1px dashed rgba(99, 102, 241, 0.48); border-radius: 14px; padding: 0.35rem; transition: border-color 160ms ease, background 160ms ease, transform 160ms ease; }
        [data-testid="stFileUploader"]:hover { background: rgba(238, 242, 255, 0.86); border-color: var(--td-indigo); transform: translateY(-1px); }
        .td-file-list { display: grid; gap: 0.45rem; margin: 0.45rem 0 1rem; }
        .td-file { align-items: center; background: rgba(248, 250, 252, 0.9); border: 1px solid var(--td-line); border-radius: 10px; display: flex; gap: 0.55rem; padding: 0.6rem 0.65rem; }
        .td-file-icon { color: var(--td-violet); font-size: 1rem; }
        .td-file-name { flex: 1; font-size: 0.73rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .td-file-count { color: var(--td-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; white-space: nowrap; }

        .stButton > button { border-radius: 10px; font-weight: 600; transition: transform 160ms ease, box-shadow 160ms ease; }
        .stButton > button:hover { box-shadow: 0 7px 18px rgba(79, 70, 229, 0.15); transform: translateY(-1px); }
        [data-testid="stSidebar"] .stButton > button { font-size: 0.76rem; }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] { border-color: rgba(148, 163, 184, 0.55); color: #475569; }
        .stChatMessage { border: 1px solid var(--td-line); border-radius: 17px; margin: 0.7rem 0; padding: 0.95rem 1.1rem; }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: rgba(255, 255, 255, 0.86); margin-left: 12%; }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: linear-gradient(135deg, rgba(238, 242, 255, 0.78), rgba(245, 243, 255, 0.78)); border-color: rgba(129, 140, 248, 0.25); margin-right: 6%; }
        [data-testid="stBottom"],
        [data-testid="stChatInput"] {
            box-shadow: 0 -8px 24px -4px rgba(15, 23, 42, 0.06) !important;
            border-radius: 16px !important;
        }
        [data-testid="stChatInput"] textarea {
            box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05) !important;
            border: 1px solid rgba(226, 232, 240, 0.9) !important;
            transition: all 0.2s ease-in-out !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        }
        [data-testid="stExpander"] { background: rgba(255, 255, 255, 0.48); border: 1px solid var(--td-line); border-radius: 12px; }
        .td-citation { margin: 0.25rem 0 0.8rem; }
        .td-citation-meta { align-items: center; display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .td-citation-meta .td-pill { background: rgba(79, 70, 229, 0.10); color: var(--td-indigo); font-family: 'JetBrains Mono', monospace; font-size: 0.64rem; padding: 0.28rem 0.55rem; }
        .td-citation blockquote { border-left: 3px solid rgba(124, 58, 237, 0.5); color: #475569; font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; margin: 0.55rem 0 0; padding: 0.45rem 0.75rem; }
        @media (max-width: 720px) { .td-steps { grid-template-columns: 1fr; } .td-hero { margin-bottom: 1.6rem; } [data-testid="stMainBlockContainer"] { padding-top: 1.7rem; } [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { margin-left: 0; } [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { margin-right: 0; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    """Render the product header and grounded-intelligence promise."""
    st.markdown(
        """
        <section class="td-hero">
            <div class="td-kicker">Private document intelligence <span class="td-badge">RAG v2.5</span></div>
            <h1>TalkTo<span class="td-gradient-text">Dock</span></h1>
            <p>Ask precise questions across your documents and get answers grounded in the files you trust.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_onboarding() -> None:
    """Show a useful first-run path when the document vault has no files."""
    st.markdown(
        """
        <section class="td-card td-onboarding">
            <h3>Your document workspace is ready</h3>
            <p>Bring in a PDF from the sidebar and turn it into a searchable source of truth.</p>
            <div class="td-steps">
                <div class="td-step"><div class="td-step-number">01 / UPLOAD</div><strong>Upload PDF</strong><span>Add a document to your vault.</span></div>
                <div class="td-step"><div class="td-step-number">02 / INDEX</div><strong>Vector indexing</strong><span>We map meaning, not just keywords.</span></div>
                <div class="td-step"><div class="td-step-number">03 / ASK</div><strong>Grounded Q&amp;A</strong><span>Get answers with source context.</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render compact, readable citation cards for an assistant response."""
    for source in sources:
        source_file = html.escape(str(source["source_file"]))
        page_number = html.escape(str(source["page_number"]))
        relevance = float(source.get("relevance_score", 0.0))
        snippet = html.escape(str(source["snippet"]))
        st.markdown(
            f"""
            <div class="td-citation">
                <div class="td-citation-meta">
                    <span class="td-pill">📄 {source_file}</span>
                    <span class="td-pill">Page {page_number}</span>
                    <span class="td-pill">Relevance {relevance:.2f} / 1.00</span>
                </div>
                <blockquote>{snippet}</blockquote>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _get_chunk_count(indexer: DocumentIndexer, file_name: str) -> int:
    """Return the persisted chunk count for a file without changing its index."""
    try:
        return len(
            indexer.get_vector_store().get(where={"source_file": file_name}).get("ids", [])
        )
    except Exception:
        logger.exception("Could not inspect chunk count for %s", file_name)
        return 0


def _render_sidebar(indexer: DocumentIndexer, engine: RAGEngine) -> str | None:
    """Render document management controls and return the selected filter."""
    with st.sidebar:
        user_email = getattr(st.user, "email", "Authenticated User")
        email = html.escape(str(user_email))
        st.markdown(
            f"""
            <div class="td-profile">
                <div class="td-avatar">{email[:1].upper()}</div>
                <div>
                    <div class="td-profile-email">{email}</div>
                    <span class="td-status">● Active</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Log out"):
            st.logout()
        st.markdown('<div class="td-sidebar-heading">Document Vault</div>', unsafe_allow_html=True)
        if st.session_state.pop("reset_notice", False):
            st.success("All documents and the search index were cleared.")

        if st.button("Reset vault", type="secondary"):
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

        st.markdown(
            '<div class="td-dropzone-label">Drop PDFs here or browse your files</div>',
            unsafe_allow_html=True,
        )
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
        st.markdown('<div class="td-sidebar-heading">Indexed documents</div>', unsafe_allow_html=True)
        if indexed_files:
            file_cards = []
            for file_name in sorted(indexed_files):
                safe_name = html.escape(file_name)
                chunk_count = _get_chunk_count(indexer, file_name)
                file_cards.append(
                    f'<div class="td-file"><span class="td-file-icon">📄</span>'
                    f'<span class="td-file-name">{safe_name}</span>'
                    f'<span class="td-file-count">{chunk_count} chunks</span></div>'
                )
            st.markdown(
                f'<div class="td-file-list">{"".join(file_cards)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="td-file-list"><div class="td-file"><span class="td-file-icon">＋</span>'
                '<span class="td-file-name">Your vault is empty</span></div></div>',
                unsafe_allow_html=True,
            )

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
                    _render_sources(message["sources"])


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
                _render_sources(result["sources"])

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
        page_title="TalkToDock",
        page_icon="📄",
        layout="wide",
    )
    user_logged_in = getattr(st.user, "is_logged_in", False)
    if not user_logged_in:
        _, auth_column, _ = st.columns([1, 2, 1])
        with auth_column:
            st.title("🔒 TalkToDock - Access Restricted")
            st.info(
                "Sign in with your Google account to access your document assistant."
            )
            if st.button("Log in with Google", type="primary"):
                if hasattr(st, "login"):
                    st.login("google")
                else:
                    st.error(
                        "Authentication requires Streamlit >= 1.42. "
                        "Run: pip install --upgrade streamlit authlib"
                    )
        st.stop()

    _inject_design_system()
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
    _render_hero()
    if not st.session_state.indexed_files:
        _render_onboarding()
    _render_chat_history()
    _handle_query(st.session_state.engine, selected_file)


if __name__ == "__main__":
    main()