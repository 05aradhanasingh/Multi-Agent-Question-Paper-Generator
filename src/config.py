from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
SYLLABUS_DIR = DATA_DIR / "syllabus"
CHROMA_DIR = DATA_DIR / "chroma"
PROMPTS_DIR = ROOT_DIR / "prompts"
EXAMPLES_DIR = ROOT_DIR / "examples"

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"))
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
COLLECTION_NAME = "syllabus"


def get_api_key(override: str | None = None) -> str:
    key = (
        override
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Set it in .env or the Streamlit sidebar."
        )
    os.environ["OPENROUTER_API_KEY"] = key
    return key


def ensure_data_dirs() -> None:
    SYLLABUS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
