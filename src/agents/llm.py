from __future__ import annotations

from crewai import LLM

from src.config import OPENROUTER_BASE_URL


def openrouter_model_slug(model: str) -> str:
    slug = (model or "").strip()
    if slug.startswith("openrouter/"):
        return slug
    if "/" in slug:
        return f"openrouter/{slug}"
    return f"openrouter/openai/{slug}"


def build_llm(model: str, api_key: str) -> LLM:
    return LLM(
        model=openrouter_model_slug(model),
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
    )
