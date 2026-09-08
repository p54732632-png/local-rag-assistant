"""Semantic retrieval and grounded answer generation for the local RAG store."""

from __future__ import annotations

import ast
import json
import logging
import os
import time
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

NO_ANSWER = "I cannot find the answer in the provided documents."
NO_RELEVANT_INFO = (
    "I cannot find any relevant information in the provided documents "
    "to answer this question."
)


def _get_secret(key: str) -> Optional[str]:
    """Retrieve secret safely from Streamlit secrets or OS environment."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


def _extract_human_text(content: object) -> str:
    """Return readable text from plain, structured, or serialized model content."""
    if isinstance(content, str):
        stripped_content = content.strip()
        if stripped_content and stripped_content[0] in "[{\"'":
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed_content = parser(stripped_content)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                if parsed_content != content:
                    return _extract_human_text(parsed_content)
        return stripped_content

    if isinstance(content, dict):
        return _extract_human_text(content.get("text", ""))

    if isinstance(content, list):
        return "\n".join(
            text
            for text in (_extract_human_text(item) for item in content)
            if text
        ).strip()

    return str(content).strip()


class SourceCitation(TypedDict):
    """A source excerpt supporting a generated answer."""

    source_file: str
    page_number: int
    snippet: str
    relevance_score: float


class QueryAnswer(TypedDict):
    """The public result returned by :meth:`RAGEngine.answer_query`."""

    query: str
    answer: str
    sources: List[SourceCitation]


class RAGEngine:
    """Retrieve indexed PDF chunks and generate strictly grounded answers."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_knowledge_base",
    ) -> None:
        """Connect to an existing Chroma collection and initialize the chat model."""
        gemini_key = _get_secret("GEMINI_API_KEY")
        if not gemini_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to Streamlit secrets or .env file."
            )

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.upload_dir = PROJECT_ROOT / "uploads"

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=gemini_key,
        )
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        self.llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=gemini_key,
    temperature=0.2,
)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are an accurate, strict research assistant. "
                    "You must answer the user query based solely on the provided context.\n"
                    "If the answer cannot be determined strictly from the provided context, "
                    f"state: '{NO_ANSWER}' Do not speculate, extrapolate, or use outside "
                    "training knowledge.\n"
                    "Every factual claim must cite its source in brackets referencing the file "
                    "name and page number, like [sample.pdf, Page 1]."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Context:\n{context}\n\nUser query:\n{query}"
                ),
            ]
        )

    def retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        file_filter: Optional[str] = None,
        score_threshold: float = 0.55,
    ) -> List[Tuple[Document, float]]:
        """Return non-empty chunks whose relevance score meets the threshold."""
        if not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")

        search_kwargs = {}
        if file_filter:
            search_kwargs["filter"] = {"source_file": file_filter}

        scored_documents = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k,
            **search_kwargs,
        )
        return [
            (document, float(score))
            for document, score in scored_documents
            if document.page_content.strip() and float(score) >= score_threshold
        ]

    def format_context(self, docs: List[Tuple[Document, float]]) -> str:
        """Format retrieved chunks with stable source and page delimiters."""
        formatted_chunks = []
        for document, _score in docs:
            source_file = str(document.metadata.get("source_file", "unknown"))
            page_number = document.metadata.get("page_number", 0)
            formatted_chunks.append(
                f"[Source: {source_file}, Page: {page_number}]\n"
                f"{document.page_content.strip()}"
            )
        return "\n\n".join(formatted_chunks)

    def answer_query(
        self,
        query: str,
        file_filter: Optional[str] = None,
        top_k: int = 4,
        score_threshold: float = 0.55,
    ) -> QueryAnswer:
        """Retrieve context and generate an answer grounded only in that context."""
        documents = self.retrieve_context(
            query,
            top_k=top_k,
            file_filter=file_filter,
            score_threshold=score_threshold,
        )
        sources = [
            SourceCitation(
                source_file=str(document.metadata.get("source_file", "unknown")),
                page_number=int(document.metadata.get("page_number", 0)),
                snippet=document.page_content.strip(),
                relevance_score=score,
            )
            for document, score in documents
        ]
        if not documents:
            return {"query": query, "answer": NO_RELEVANT_INFO, "sources": []}

        context = self.format_context(documents)
        response = (self.prompt | self.llm).invoke(
            {"context": context, "query": query}
        )

        answer = _extract_human_text(response.content)
        return {"query": query, "answer": answer, "sources": sources}

    def clear_database(self) -> None:
        """Clear the Chroma collection and local uploaded PDFs."""
        try:
            self.vector_store.delete_collection()
        except Exception:
            logger.exception("Could not delete Chroma collection during reset")
            raise

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        for pdf_path in (*self.upload_dir.glob("*.pdf"), *self.upload_dir.glob("*.PDF")):
            if pdf_path.is_file():
                pdf_path.unlink()

        logger.info(
            "Cleared Chroma database collection '%s' and uploads directory '%s'",
            self.collection_name,
            self.upload_dir,
        )


def _run_smoke_test() -> None:
    """Run a sample grounded query and report latency and citations."""
    started_at = time.perf_counter()
    engine = RAGEngine()
    result = engine.answer_query(
        "What topics are covered in sample.pdf?",
        file_filter="sample.pdf",
    )
    latency_ms = (time.perf_counter() - started_at) * 1000
    pprint({"answer": result["answer"], "sources": result["sources"]})
    logger.info("Total retrieval + generation latency: %.2f ms", latency_ms)


if __name__ == "__main__":
    _run_smoke_test()