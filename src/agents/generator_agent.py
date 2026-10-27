from __future__ import annotations

from crewai import Agent

from src.agents.llm import build_llm
from src.config import PROMPTS_DIR


def build_generator_agent(model: str, api_key: str, verbose: bool = False) -> Agent:
    backstory = (PROMPTS_DIR / "generator.txt").read_text(encoding="utf-8")
    return Agent(
        role="Question Paper Author",
        goal=(
            "Draft MCQ, Short, and Long questions that match the paper spec, "
            "Bloom levels, marks, and retrieved syllabus context. Output valid JSON."
        ),
        backstory=backstory,
        llm=build_llm(model, api_key),
        verbose=verbose,
        allow_delegation=False,
    )
