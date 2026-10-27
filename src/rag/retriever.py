from __future__ import annotations

from pathlib import Path

import chromadb
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from src.config import COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
from src.rag.embeddings import build_embedding_function


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int
    score: float


class SyllabusRetriever:
    def __init__(
        self,
        persist_dir: Path,
        api_key: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        embedding_fn, _backend = build_embedding_function(
            persist_dir=persist_dir,
            api_key=api_key,
            embedding_model=embedding_model,
            prefer_openrouter=True,
        )
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, k: int = 6) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        chunks: list[RetrievedChunk] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            similarity = max(0.0, 1.0 - float(distance))
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(meta.get("source", "unknown")),
                    page=int(meta.get("page", 0)),
                    score=round(similarity, 4),
                )
            )
        return chunks

    def format_chunks(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "NO_SYLLABUS_CONTEXT"
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Chunk {i} | {chunk.source} p.{chunk.page} | score={chunk.score}]\n{chunk.text}"
            )
        return "\n\n".join(parts)


class SearchSyllabusInput(BaseModel):
    query: str = Field(..., description="Concept, topic, or formula to retrieve from the syllabus.")
    k: int = Field(6, description="Number of chunks to return.")


class SearchSyllabusTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "search_syllabus"
    description: str = (
        "Retrieve syllabus/textbook chunks that already exist in the ingested documents. "
        "Use this before drafting any question. Never invent content not present in results."
    )
    args_schema: type[BaseModel] = SearchSyllabusInput
    retriever: SyllabusRetriever

    def _run(self, query: str, k: int = 6) -> str:
        chunks = self.retriever.search(query, k=k)
        return self.retriever.format_chunks(chunks)
