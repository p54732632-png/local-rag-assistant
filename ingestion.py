"""PDF ingestion and local Chroma vector storage for a RAG pipeline."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError, PdfStreamError
from pypdf.generic import (
    DictionaryObject,
    NameObject,
    DecodedStreamObject,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_secret(key: str) -> Optional[str]:
    """Retrieve secret safely from Streamlit secrets or OS environment."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


class DocumentIndexer:
    """Load PDF documents, create Gemini embeddings, and persist them in Chroma."""

    def __init__(
        self,
        upload_dir: str = "./uploads",
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_knowledge_base",
    ) -> None:
        """Initialize storage paths and the Chroma collection configuration."""
        self.upload_dir = Path(upload_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        api_key = _get_secret("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to Streamlit secrets or a .env file."
            )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key,
        )

    def load_and_chunk_pdf(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> List[Document]:
        """Load a PDF and split its pages into metadata-preserving text chunks."""
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            with path.open("rb") as pdf_file:
                if pdf_file.read(4) != b"%PDF":
                    logger.error("Invalid PDF header in %s", path)
                    return []

            loader = PyPDFLoader(str(path))
            pages = loader.load()
            if len(pages) > 100:
                raise ValueError(
                    "Document exceeds the 100-page limit. Please upload a shorter "
                    "document or split the file."
                )
        except (PdfStreamError, PdfReadError) as exc:
            logger.error("Could not read malformed PDF %s: %s", path, exc)
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        raw_chunks = splitter.split_documents(pages)
        file_name = path.name
        chunks: List[Document] = []
        for chunk_index, chunk in enumerate(raw_chunks):
            page_number = int(chunk.metadata.get("page", 0)) + 1
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "source_file": file_name,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )
            chunks.append(
                Document(
                    page_content=chunk.page_content.strip(),
                    metadata=metadata,
                )
            )
        logger.info("Loaded %s and created %d chunks", path, len(chunks))
        return chunks

    def index_document(self, pdf_path: str) -> Chroma:
        """Embed and persist all chunks from one PDF, returning the vector store."""
        chunks = self.load_and_chunk_pdf(pdf_path)
        if not chunks:
            raise ValueError(f"No extractable text found in PDF: {pdf_path}")

        vector_store = self.get_vector_store()
        file_name = Path(pdf_path).name
        existing_chunks = vector_store.get(where={"source_file": file_name})
        existing_count = len(existing_chunks.get("ids", []))
        if existing_count:
            logger.info(
                "File %s is already indexed with %d chunks; skipping re-embedding",
                file_name,
                existing_count,
            )
            return vector_store

        chunk_ids = [
            f"{file_name}_p{chunk.metadata['page_number']}_c{chunk.metadata['chunk_index']}"
            for chunk in chunks
        ]
        vector_store.add_documents(documents=chunks, ids=chunk_ids)
        logger.info(
            "Persisted %d chunks to %s (collection=%s)",
            len(chunks),
            self.persist_dir,
            self.collection_name,
        )
        return vector_store

    def get_vector_store(self) -> Chroma:
        """Connect to the persisted Chroma collection without loading any PDF."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def clear_database(self) -> None:
        """Delete the Chroma collection and restore an empty persistent directory."""
        vector_store = self.get_vector_store()
        try:
            vector_store.delete_collection()
        except Exception:
            logger.exception("Could not delete Chroma collection during reset")
            raise
        self.persist_dir.mkdir(parents=True, exist_ok=True)


def _run_sample() -> None:
    """Index the conventional sample PDF and report the result."""
    upload_sample_path = PROJECT_ROOT / "uploads" / "sample.pdf"
    root_sample_path = PROJECT_ROOT / "sample.pdf"

    if _is_valid_pdf(upload_sample_path):
        sample_path = upload_sample_path
        logger.info("Detected sample PDF at %s", sample_path)
    elif _is_valid_pdf(root_sample_path):
        upload_sample_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root_sample_path, upload_sample_path)
        sample_path = upload_sample_path
        logger.info(
            "Detected sample PDF at %s and copied it to %s",
            root_sample_path,
            sample_path,
        )
    else:
        logger.warning(
            "No valid sample PDF found at %s or %s; generating one.",
            upload_sample_path,
            root_sample_path,
        )
        _create_sample_pdf(upload_sample_path)
        sample_path = upload_sample_path
        logger.info("Generated valid sample PDF at %s", sample_path)

    try:
        indexer = DocumentIndexer()
        vector_store = indexer.index_document(str(sample_path))
        indexed_count = vector_store._collection.count()
        verification_passed = indexed_count > 0
        logger.info("Indexed chunk count: %d", indexed_count)
        logger.info("Verification status: %s", "PASSED" if verification_passed else "FAILED")
    except (EnvironmentError, FileNotFoundError, ValueError) as exc:
        logger.error("Indexing failed: %s", exc)
    except Exception as exc:
        logger.error("Sample indexing failed: %s", exc)


def _is_valid_pdf(pdf_path: Path) -> bool:
    """Return whether a file has a valid PDF header and readable pages."""
    if not pdf_path.is_file():
        return False

    try:
        with pdf_path.open("rb") as pdf_file:
            if pdf_file.read(4) != b"%PDF":
                logger.error("Invalid PDF header in %s", pdf_path)
                return False
        reader = PdfReader(str(pdf_path), strict=False)
        return len(reader.pages) > 0
    except (PdfStreamError, PdfReadError, OSError) as exc:
        logger.error("Invalid or unreadable PDF %s: %s", pdf_path, exc)
        return False


def _create_sample_pdf(pdf_path: Path) -> None:
    """Create a readable two-page PDF containing sample RAG documentation."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    page_texts = [
        "Machine learning enables systems to learn patterns from data. "
        "Supervised learning uses labeled examples for tasks such as classification.",
        "Retrieval augmented generation combines document retrieval with language "
        "generation so answers can be grounded in relevant source material.",
    ]

    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        content = DecodedStreamObject()
        content.set_data(
            f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(content)

    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)


if __name__ == "__main__":
    _run_sample()
