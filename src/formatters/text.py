from __future__ import annotations

from collections import defaultdict

from src.schemas.output import Question, QuestionPaper


def _header(paper: QuestionPaper, kind: str) -> str:
    meta = paper.paper_metadata
    return "\n".join(
        [
            meta.title,
            kind,
            f"Maximum Marks: {meta.total_marks}",
            f"Time Allowed: {meta.time_allowed_mins} minutes",
            "",
            "Instructions:",
            "1. Answer all questions.",
            "2. Figures in brackets against a question indicate marks.",
            "3. Write answers clearly. For MCQs, choose exactly one option.",
            "",
            "-" * 72,
            "",
        ]
    )


def _grouped(questions: list[Question]) -> list[tuple[str, list[Question]]]:
    grouped: dict[str, list[Question]] = defaultdict(list)
    order: list[str] = []
    for question in questions:
        if question.section not in grouped:
            order.append(question.section)
        grouped[question.section].append(question)
    return [(name, grouped[name]) for name in order]


def render_question_paper_text(paper: QuestionPaper) -> str:
    lines = [_header(paper, "QUESTION PAPER")]
    for section_name, questions in _grouped(paper.questions):
        section_marks = sum(q.marks for q in questions)
        lines.append(f"{section_name}  ({section_marks} marks)")
        lines.append("")
        for question in questions:
            lines.append(f"{question.id}. {question.question_text}  [{question.marks}]")
            if question.type == "MCQ" and question.options:
                for option in question.options:
                    lines.append(f"    {option}")
            lines.append("")
        lines.append("-" * 72)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_answer_key_text(paper: QuestionPaper) -> str:
    lines = [_header(paper, "ANSWER KEY AND SOLUTIONS")]
    for section_name, questions in _grouped(paper.questions):
        lines.append(f"{section_name}")
        lines.append("")
        for question in questions:
            lines.append(f"{question.id}. {question.question_text}  [{question.marks}]")
            if question.type == "MCQ" and question.options:
                for option in question.options:
                    lines.append(f"    {option}")
            lines.append(f"Correct answer: {question.correct_answer}")
            lines.append("Solution:")
            lines.append(question.step_by_step_solution)
            lines.append("")
        lines.append("-" * 72)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
