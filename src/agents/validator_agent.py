from __future__ import annotations

from crewai import Agent

from src.agents.llm import build_llm
from src.config import PROMPTS_DIR


def build_validator_agent(model: str, api_key: str, verbose: bool = False) -> Agent:
    backstory = (PROMPTS_DIR / "validator.txt").read_text(encoding="utf-8")
    return Agent(
        role="Academic Auditor and Guardrail",
        goal=(
            "Audit drafted papers for math accuracy, unique MCQ keys, "
            "non-redundancy, and syllabus alignment. Return structured JSON."
        ),
        backstory=backstory,
        llm=build_llm(model, api_key),
        verbose=verbose,
        allow_delegation=False,
    )
