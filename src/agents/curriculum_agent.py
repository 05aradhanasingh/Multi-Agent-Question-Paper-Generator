from __future__ import annotations

from crewai import Agent

from src.agents.llm import build_llm
from src.config import PROMPTS_DIR
from src.rag.retriever import SearchSyllabusTool, SyllabusRetriever


def build_curriculum_agent(
    retriever: SyllabusRetriever,
    model: str,
    api_key: str,
    verbose: bool = False,
) -> Agent:
    backstory = (PROMPTS_DIR / "curriculum.txt").read_text(encoding="utf-8")
    tool = SearchSyllabusTool(retriever=retriever)
    return Agent(
        role="Curriculum and Retrieval Specialist",
        goal=(
            "Retrieve cited syllabus chunks for every exam section. "
            "Never hallucinate outside the ingested documents."
        ),
        backstory=backstory,
        tools=[tool],
        llm=build_llm(model, api_key),
        verbose=verbose,
        allow_delegation=False,
    )
