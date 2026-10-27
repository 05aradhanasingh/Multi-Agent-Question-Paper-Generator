from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from crewai import Crew, Process, Task
from openai import RateLimitError

from src.agents.curriculum_agent import build_curriculum_agent
from src.agents.formatter_agent import FormattingAgent, PaperArtifacts
from src.agents.generator_agent import build_generator_agent
from src.agents.validator_agent import build_validator_agent
from src.json_util import extract_json_object
from src.rag.retriever import SyllabusRetriever
from src.schemas.input import PaperSpec
from src.schemas.output import Question, QuestionPaper, ValidationReport
from src.validation.rules import DeterministicAudit, audit_paper

MAX_REVISIONS = 2
COVERAGE_THRESHOLD = 0.35


@dataclass
class GenerationResult:
    paper: QuestionPaper
    artifacts: PaperArtifacts
    syllabus_brief: str
    attempts: int
    agent_log: list[str] = field(default_factory=list)


QUOTA_MESSAGE = (
    "OpenRouter has no remaining credits for this request. "
    "Add credit at https://openrouter.ai/settings/credits "
    "or use a funded OpenRouter key in .env / the sidebar."
)


def _task_output(result) -> str:
    if result is None:
        return ""
    raw = getattr(result, "raw", None)
    if isinstance(raw, str):
        return raw
    return str(result)


def _kickoff(crew) -> str:
    try:
        return _task_output(crew.kickoff())
    except RateLimitError as exc:
        raise RuntimeError(QUOTA_MESSAGE) from exc
    except Exception as exc:
        text = str(exc).lower()
        if "insufficient_quota" in text or "credit_balance_exhausted" in text:
            raise RuntimeError(QUOTA_MESSAGE) from exc
        raise


def _retrieve_programmatic(spec: PaperSpec, retriever: SyllabusRetriever) -> str:
    blocks: list[str] = []
    for section in spec.sections:
        query = (
            f"{spec.subject_name} {section.section_name} "
            f"{section.question_type} {section.blooms_level} {section.difficulty}"
        )
        k = max(6, section.count * 3)
        chunks = retriever.search(query, k=k)
        heading = (
            f"=== {section.section_name} | {section.question_type} | "
            f"{section.difficulty} | {section.blooms_level} | k={k} ==="
        )
        blocks.append(heading)
        blocks.append(retriever.format_chunks(chunks))
    return "\n\n".join(blocks)


def _attach_coverage(paper: QuestionPaper, retriever: SyllabusRetriever) -> None:
    for question in paper.questions:
        query = f"{question.topic} {question.question_text}"
        hits = retriever.search(query, k=1)
        if not hits:
            question.retrieval_score = 0.0
            continue
        question.retrieval_score = hits[0].score
        if not question.source_excerpt:
            question.source_excerpt = hits[0].text[:500]


def _as_text(value, *, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "question", "question_text", "stem", "value"):
            if key in value:
                return _as_text(value[key], fallback=fallback)
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return " ".join(part for part in parts if part).strip() or fallback
    return str(value).strip() or fallback


def _normalize_question(raw: dict, index: int) -> dict | None:
    qtype = _as_text(raw.get("type") or raw.get("question_type"))
    mapping = {
        "MCQ": "MCQ",
        "SHORT": "Short",
        "SHORT ANSWER": "Short",
        "LONG": "Long",
        "LONG ANSWER": "Long",
    }
    qtype = mapping.get(qtype.upper(), qtype)
    if qtype not in {"MCQ", "Short", "Long"}:
        qtype = "Short"

    question_text = _as_text(
        raw.get("question_text")
        or raw.get("question")
        or raw.get("stem")
        or raw.get("prompt")
        or raw.get("text")
    )
    if not question_text:
        return None

    options = raw.get("options")
    if qtype != "MCQ":
        options = None
    elif isinstance(options, list):
        options = [_as_text(item) for item in options if _as_text(item)]
        if len(options) < 4:
            options = None
    else:
        options = None

    difficulty = _as_text(raw.get("difficulty"), fallback="Medium")
    if difficulty not in {"Easy", "Medium", "Hard"}:
        difficulty = "Medium"

    marks_raw = raw.get("marks")
    try:
        marks = int(marks_raw)
    except (TypeError, ValueError):
        marks = 1

    return {
        "id": _as_text(raw.get("id"), fallback=f"Q{index}"),
        "section": _as_text(raw.get("section"), fallback="Section A"),
        "type": qtype,
        "topic": _as_text(raw.get("topic"), fallback="Syllabus topic"),
        "question_text": question_text,
        "options": options,
        "correct_answer": _as_text(raw.get("correct_answer") or raw.get("answer")),
        "step_by_step_solution": _as_text(
            raw.get("step_by_step_solution") or raw.get("solution") or raw.get("explanation")
        ),
        "marks": marks,
        "difficulty": difficulty,
        "blooms_level": _as_text(raw.get("blooms_level") or raw.get("bloom"), fallback="Understand"),
        "source_excerpt": _as_text(raw.get("source_excerpt")) or None,
    }


def _paper_from_generator(payload: dict, spec: PaperSpec) -> QuestionPaper:
    metadata = payload.get("paper_metadata") or {}
    title = metadata.get("title") or f"{spec.subject_name} Question Paper"
    questions_raw = payload.get("questions") or payload.get("question_paper") or []
    if isinstance(questions_raw, dict):
        flattened: list = []
        for value in questions_raw.values():
            if isinstance(value, list):
                flattened.extend(value)
            else:
                flattened.append(value)
        questions_raw = flattened
    questions: list[Question] = []
    for i, item in enumerate(questions_raw, start=1):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_question(item, i)
        if normalized is None:
            continue
        questions.append(Question.model_validate(normalized))
    return QuestionPaper(
        paper_metadata={
            "title": title,
            "total_marks": int(metadata.get("total_marks") or spec.total_marks),
            "time_allowed_mins": int(
                metadata.get("time_allowed_mins") or spec.time_allowed_mins
            ),
        },
        questions=questions,
        validation_report=ValidationReport(
            status="REVISED",
            total_marks_verified=False,
            syllabus_coverage_score=0.0,
            auditor_notes="Pending audit.",
        ),
    )


def _public_paper_dump(paper: QuestionPaper) -> str:
    data = paper.model_dump()
    for question in data["questions"]:
        question.pop("source_excerpt", None)
        question.pop("retrieval_score", None)
    data.pop("validation_report", None)
    return json.dumps(data, indent=2)


def _emit(on_progress: Callable[[str], None] | None, message: str, log: list[str]) -> None:
    log.append(message)
    if on_progress:
        on_progress(message)


def run_paper_pipeline(
    spec: PaperSpec,
    persist_dir: Path,
    api_key: str,
    model: str,
    verbose: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> GenerationResult:
    agent_log: list[str] = []
    retriever = SyllabusRetriever(persist_dir=persist_dir, api_key=api_key)
    if retriever.count() == 0:
        raise RuntimeError("No syllabus chunks found. Ingest a PDF first.")

    curriculum = build_curriculum_agent(retriever, model, api_key, verbose)
    generator = build_generator_agent(model, api_key, verbose)
    validator = build_validator_agent(model, api_key, verbose)
    formatter = FormattingAgent()

    _emit(
        on_progress,
        "Curriculum & Retrieval Agent is searching the ingested syllabus for topic context.",
        agent_log,
    )
    programmatic_context = _retrieve_programmatic(spec, retriever)
    curriculum_task = Task(
        description=(
            f"Search the ingested {spec.subject_name} syllabus for content needed "
            "to write this exam. Cover every section. Quote excerpts with source and page.\n\n"
            f"Paper spec:\n{spec.model_dump_json(indent=2)}\n\n"
            "You MUST call search_syllabus at least once per section."
        ),
        expected_output=(
            "A syllabus briefing with cited excerpts, topics found, and gaps. "
            "No exam questions."
        ),
        agent=curriculum,
    )
    retrieval_crew = Crew(
        agents=[curriculum],
        tasks=[curriculum_task],
        process=Process.sequential,
        verbose=verbose,
    )
    syllabus_brief = _kickoff(retrieval_crew)
    _emit(
        on_progress,
        "Curriculum & Retrieval Agent finished citing syllabus chunks for every section.",
        agent_log,
    )
    combined_context = (
        f"{syllabus_brief}\n\n--- Retrieved chunks ---\n{programmatic_context}"
    )

    revision_request = ""
    paper: QuestionPaper | None = None
    last_audit: DeterministicAudit | None = None
    last_llm_notes = ""
    attempts = 0

    for attempt in range(1 + MAX_REVISIONS):
        attempts = attempt + 1
        revision_block = (
            f"\n\nREVISION REQUEST FROM AUDITOR:\n{revision_request}"
            if revision_request
            else ""
        )
        generator_task = Task(
            description=(
                "Write the full question paper as a single JSON object.\n"
                f"Paper spec:\n{spec.model_dump_json(indent=2)}\n\n"
                f"Syllabus context (use only this):\n{combined_context}"
                f"{revision_block}"
            ),
            expected_output="Valid JSON object matching the question paper schema, no extra commentary.",
            agent=generator,
        )
        gen_crew = Crew(
            agents=[generator],
            tasks=[generator_task],
            process=Process.sequential,
            verbose=verbose,
        )
        if attempt == 0:
            _emit(
                on_progress,
                "Question Generator Agent is drafting MCQ, short, and long questions from the syllabus.",
                agent_log,
            )
        else:
            _emit(
                on_progress,
                f"Question Generator Agent is revising the paper (attempt {attempts} of {1 + MAX_REVISIONS}).",
                agent_log,
            )
        generated = _kickoff(gen_crew)
        payload = extract_json_object(generated)
        paper = _paper_from_generator(payload, spec)
        _attach_coverage(paper, retriever)
        last_audit = audit_paper(paper, spec)

        validator_task = Task(
            description=(
                "Audit this question paper. Deterministic engine notes are included; "
                "you still check MCQ uniqueness, distractor quality, redundancy, "
                "math/formula accuracy, and syllabus backing.\n\n"
                f"Deterministic issues:\n{json.dumps(last_audit.issues, indent=2)}\n\n"
                f"Paper:\n{_public_paper_dump(paper)}\n\n"
                f"Syllabus context:\n{combined_context[:8000]}"
            ),
            expected_output="JSON with keys passed, issues, revision_request, notes.",
            agent=validator,
        )
        val_crew = Crew(
            agents=[validator],
            tasks=[validator_task],
            process=Process.sequential,
            verbose=verbose,
        )
        _emit(
            on_progress,
            "Validator & Auditor Agent is checking marks, MCQ keys, redundancy, and syllabus alignment.",
            agent_log,
        )
        validator_raw = _kickoff(val_crew)
        try:
            validator_json = extract_json_object(validator_raw)
        except ValueError:
            validator_json = {
                "passed": last_audit.ok,
                "issues": last_audit.issues,
                "revision_request": last_audit.revision_request,
                "notes": validator_raw[:1000],
            }
        last_llm_notes = str(validator_json.get("notes") or "")
        llm_passed = bool(validator_json.get("passed")) and last_audit.ok
        llm_revision = str(validator_json.get("revision_request") or "").strip()
        llm_issues = validator_json.get("issues") or []

        if llm_passed:
            _emit(
                on_progress,
                "Validator & Auditor Agent approved the paper.",
                agent_log,
            )
            paper.validation_report = ValidationReport(
                status="APPROVED",
                total_marks_verified=last_audit.total_marks_verified,
                syllabus_coverage_score=last_audit.coverage_score,
                auditor_notes=last_llm_notes or "All checks passed.",
            )
            break
        _emit(
            on_progress,
            "Validator & Auditor Agent requested revisions; sending feedback to the generator.",
            agent_log,
        )

        revision_bits = [last_audit.revision_request]
        if llm_revision:
            revision_bits.append(llm_revision)
        if llm_issues:
            revision_bits.append("LLM auditor issues: " + "; ".join(map(str, llm_issues)))
        revision_request = "\n".join(bit for bit in revision_bits if bit)
        paper.validation_report = ValidationReport(
            status="REVISED",
            total_marks_verified=last_audit.total_marks_verified,
            syllabus_coverage_score=last_audit.coverage_score,
            auditor_notes=revision_request or last_llm_notes,
        )

    assert paper is not None
    if last_audit is not None:
        paper.validation_report.total_marks_verified = last_audit.total_marks_verified
        paper.validation_report.syllabus_coverage_score = last_audit.coverage_score
        if paper.validation_report.status != "APPROVED":
            notes = [last_llm_notes, last_audit.revision_request]
            paper.validation_report.auditor_notes = "\n".join(n for n in notes if n)

    _emit(
        on_progress,
        "Formatting & Layout Agent is building the question paper and answer key as PDF and TXT.",
        agent_log,
    )
    artifacts = formatter.render(paper)
    _emit(on_progress, "Formatting & Layout Agent finished. Downloads are ready.", agent_log)
    return GenerationResult(
        paper=paper,
        artifacts=artifacts,
        syllabus_brief=syllabus_brief,
        attempts=attempts,
        agent_log=agent_log,
    )
