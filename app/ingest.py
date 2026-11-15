"""PDF parsing, page-aware chunking, and deterministic Chroma indexing."""

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class Page:
    """Text and one-based page number extracted from a PDF."""

    text: str
    page: int
    source: str


@dataclass(frozen=True)
class Chunk:
    """A text segment with its PDF source location."""

    text: str
    source: str
    page: int


def load_pdf(path: str | Path) -> list[Page]:
    """Extract non-empty text pages from a PDF, preserving one-based page numbers."""
    from pypdf import PdfReader

    pdf_path = Path(path)
    pages: list[Page] = []
    for page_number, pdf_page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if text:
            pages.append(Page(text=text, page=page_number, source=pdf_path.name))
    return pages


def chunk_pages(
    pages: list[Page], size: int | None = None, overlap: int | None = None
) -> list[Chunk]:
    """Split each page into overlapping chunks without losing page metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size = settings.chunk_size if size is None else size
    chunk_overlap = settings.chunk_overlap if overlap is None else overlap
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(
            Chunk(text=text, source=page.source, page=page.page)
            for text in splitter.split_text(page.text)
            if text.strip()
        )
    return chunks


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """Load and cache the local sentence-transformers embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


_vector_store_lock = threading.Lock()


DEFAULT_COLLECTION = "docs"


@lru_cache(maxsize=None)
def _open_vector_store(collection_name: str, directory: str | None) -> Any:
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=directory,
    )


def get_vector_store(
    collection_name: str = DEFAULT_COLLECTION, persist_directory: str | Path | None = None
) -> Any:
    """Open a Chroma collection once per (name, directory).

    Without a persist directory the collection is in-memory, so each app session
    (which owns a unique collection name) starts empty and never sees other sessions' files.
    Clients are cached and creation is locked because concurrent creation under
    Streamlit reruns/threads breaks chromadb's Rust bindings.
    """
    directory = str(persist_directory) if persist_directory else None
    with _vector_store_lock:
        return _open_vector_store(collection_name, directory)


def clear_collection(collection_name: str = DEFAULT_COLLECTION) -> None:
    """Delete every chunk in a collection."""
    collection = get_vector_store(collection_name)._collection
    indexed_ids = collection.get().get("ids", [])
    if indexed_ids:
        collection.delete(ids=indexed_ids)


def build_index(
    chunks: list[Chunk],
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str | Path | None = None,
) -> None:
    """Upsert chunks using stable source/page/index IDs to prevent duplicates."""
    if not chunks:
        return
    from langchain_core.documents import Document

    documents = [
        Document(page_content=chunk.text, metadata={"source": chunk.source, "page": chunk.page})
        for chunk in chunks
    ]
    ids = [f"{chunk.source}-{chunk.page}-{chunk_index}" for chunk_index, chunk in enumerate(chunks)]
    get_vector_store(collection_name, persist_directory).add_documents(documents=documents, ids=ids)


def ingest_file(
    path: str | Path,
    collection_name: str = DEFAULT_COLLECTION,
    persist_directory: str | Path | None = None,
) -> int:
    """Parse, chunk, and index a PDF; return the number of chunks processed."""
    chunks = chunk_pages(load_pdf(path))
    build_index(chunks, collection_name, persist_directory)
    return len(chunks)
