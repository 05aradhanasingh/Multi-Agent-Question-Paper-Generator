from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas.input import PaperSpec
from src.schemas.output import Question, QuestionPaper

TYPE_MAP = {
    "MCQ": "MCQ",
    "Short Answer": "Short",
    "Long Answer": "Long",
}

COVERAGE_THRESHOLD = 0.35


@dataclass
class DeterministicAudit:
    ok: bool
    total_marks_verified: bool
    issues: list[str] = field(default_factory=list)
    coverage_score: float = 0.0
    revision_request: str = ""


def _section_lookup(spec: PaperSpec) -> dict[str, object]:
    return {section.section_name: section for section in spec.sections}


def audit_paper(paper: QuestionPaper, spec: PaperSpec) -> DeterministicAudit:
    issues: list[str] = []
    sections = _section_lookup(spec)

    expected_count = sum(s.count for s in spec.sections)
    if len(paper.questions) != expected_count:
        issues.append(
            f"Expected {expected_count} questions, got {len(paper.questions)}."
        )

    marks_sum = sum(q.marks for q in paper.questions)
    total_marks_verified = marks_sum == spec.total_marks
    if not total_marks_verified:
        issues.append(
            f"Marks sum to {marks_sum}, expected {spec.total_marks}."
        )

    if paper.paper_metadata.total_marks != spec.total_marks:
        issues.append("paper_metadata.total_marks does not match the spec.")
    if paper.paper_metadata.time_allowed_mins != spec.time_allowed_mins:
        issues.append("paper_metadata.time_allowed_mins does not match the spec.")

    ids = [q.id for q in paper.questions]
    if len(ids) != len(set(ids)):
        issues.append("Question IDs are not unique.")
    expected_ids = [f"Q{i}" for i in range(1, len(paper.questions) + 1)]
    if ids != expected_ids:
        issues.append(f"Question IDs must be sequential {expected_ids}.")

    texts = [q.question_text.strip().lower() for q in paper.questions]
    if any(not text for text in texts):
        issues.append("One or more questions have empty question_text; rewrite them as strings.")
    if len([t for t in texts if t]) != len(set(t for t in texts if t)):
        issues.append("Two or more questions have identical wording.")

    counts: dict[str, int] = {}
    for question in paper.questions:
        _audit_question(question, spec, sections, issues)
        counts[question.section] = counts.get(question.section, 0) + 1

    for section in spec.sections:
        actual = counts.get(section.section_name, 0)
        if actual != section.count:
            issues.append(
                f"{section.section_name} should have {section.count} questions, found {actual}."
            )

    covered = 0
    for question in paper.questions:
        score = question.retrieval_score
        excerpt = (question.source_excerpt or "").strip()
        if (score is not None and score >= COVERAGE_THRESHOLD) or excerpt:
            covered += 1
    coverage = covered / len(paper.questions) if paper.questions else 0.0

    if coverage < 0.7:
        issues.append(
            f"Syllabus coverage score {coverage:.2f} is below 0.70. "
            "Ground every question in retrieved source excerpts."
        )

    revision = ""
    if issues:
        numbered = "\n".join(f"- {item}" for item in issues)
        revision = (
            "Revise the question paper. Fix every item below without dropping "
            "the required section counts, marks, difficulty, or Bloom levels.\n"
            + numbered
        )

    return DeterministicAudit(
        ok=not issues,
        total_marks_verified=total_marks_verified,
        issues=issues,
        coverage_score=round(coverage, 2),
        revision_request=revision,
    )


def _audit_question(
    question: Question,
    spec: PaperSpec,
    sections: dict,
    issues: list[str],
) -> None:
    section = sections.get(question.section)
    if section is None:
        issues.append(f"{question.id} uses unknown section '{question.section}'.")
        return

    expected_type = TYPE_MAP[section.question_type]
    if question.type != expected_type:
        issues.append(
            f"{question.id} type is {question.type}, expected {expected_type}."
        )
    if question.marks != section.marks_per_question:
        issues.append(
            f"{question.id} is {question.marks} marks, expected {section.marks_per_question}."
        )
    if question.difficulty != section.difficulty:
        issues.append(
            f"{question.id} difficulty is {question.difficulty}, expected {section.difficulty}."
        )
    if question.blooms_level.strip().lower() != section.blooms_level.strip().lower():
        issues.append(
            f"{question.id} Bloom level is {question.blooms_level}, expected {section.blooms_level}."
        )
    if not question.topic.strip():
        issues.append(f"{question.id} is missing a topic.")
    if not question.question_text.strip():
        issues.append(f"{question.id} has empty question text.")
    if not question.correct_answer.strip():
        issues.append(f"{question.id} is missing a correct answer.")
    if not question.step_by_step_solution.strip():
        issues.append(f"{question.id} is missing a step-by-step solution.")

    if question.type == "MCQ":
        if not question.options or len(question.options) != 4:
            issues.append(f"{question.id} must have exactly 4 options.")
        else:
            prefixes = [opt.strip()[:2] for opt in question.options]
            if prefixes != ["A)", "B)", "C)", "D)"]:
                issues.append(
                    f"{question.id} options must start with A) B) C) D)."
                )
            bodies = [opt.split(")", 1)[-1].strip().lower() for opt in question.options]
            if len(set(bodies)) < 4:
                issues.append(f"{question.id} has duplicate option text.")
        answer_letter = question.correct_answer.strip()[:1].upper()
        if answer_letter not in {"A", "B", "C", "D"}:
            issues.append(
                f"{question.id} correct_answer must start with A, B, C, or D."
            )
    else:
        if question.options not in (None, []):
            issues.append(f"{question.id} is not MCQ so options must be null.")
