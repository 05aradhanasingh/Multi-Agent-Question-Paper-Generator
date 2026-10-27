from __future__ import annotations

from pathlib import Path

import chromadb
from pypdf import PdfReader

from src.config import COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, ensure_data_dirs
from src.rag.embeddings import build_embedding_function

CHUNK_CHARS = 3200  # ~800 tokens
CHUNK_OVERLAP = 600  # ~150 tokens


def _chunk_text(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + CHUNK_CHARS)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def ingest_pdfs(
    pdf_paths: list[Path],
    persist_dir: Path,
    api_key: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, int]:
    """Extract, chunk, embed, and store syllabus PDFs in Chroma."""
    ensure_data_dirs()
    persist_dir.mkdir(parents=True, exist_ok=True)

    embedding_fn, backend = build_embedding_function(
        persist_dir=persist_dir,
        api_key=api_key,
        embedding_model=embedding_model,
        prefer_openrouter=True,
    )
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    pages = 0

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        pages += len(reader.pages)
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk_index, chunk in enumerate(_chunk_text(text)):
                ids.append(f"{pdf_path.name}-p{page_index}-c{chunk_index}")
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": pdf_path.name,
                        "page": page_index,
                    }
                )

    if not documents:
        raise ValueError("No extractable text found in the uploaded PDF(s).")

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    return {
        "chunks": len(documents),
        "pages": pages,
        "files": len(pdf_paths),
        "embedding_backend": backend,
    }
