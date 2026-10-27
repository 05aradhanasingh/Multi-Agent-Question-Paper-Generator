from __future__ import annotations

from pathlib import Path

from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    OpenAIEmbeddingFunction,
)
from openai import OpenAI, RateLimitError

from src.config import DEFAULT_EMBEDDING_MODEL, OPENROUTER_BASE_URL

BACKEND_FILE = "embedding_backend.txt"


def _write_backend(persist_dir: Path, name: str) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / BACKEND_FILE).write_text(name, encoding="utf-8")


def read_backend(persist_dir: Path) -> str | None:
    path = persist_dir / BACKEND_FILE
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def openrouter_embeddings_available(api_key: str, model: str) -> bool:
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    try:
        client.embeddings.create(model=model, input="ping")
        return True
    except RateLimitError:
        return False
    except Exception:
        return False


def build_embedding_function(
    persist_dir: Path,
    api_key: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    prefer_openrouter: bool = True,
) -> tuple[EmbeddingFunction, str]:
    existing = read_backend(persist_dir)
    if existing == "default":
        return DefaultEmbeddingFunction(), "default"
    if existing in {"openrouter", "openai"}:
        return (
            OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=OPENROUTER_BASE_URL,
                model_name=embedding_model,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Question Paper Agents",
                },
            ),
            "openrouter",
        )

    if prefer_openrouter and openrouter_embeddings_available(api_key, embedding_model):
        _write_backend(persist_dir, "openrouter")
        return (
            OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=OPENROUTER_BASE_URL,
                model_name=embedding_model,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Question Paper Agents",
                },
            ),
            "openrouter",
        )

    _write_backend(persist_dir, "default")
    return DefaultEmbeddingFunction(), "default"
