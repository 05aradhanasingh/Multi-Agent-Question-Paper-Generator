from __future__ import annotations

from dataclasses import dataclass

from src.formatters.pdf import render_answer_key_pdf, render_question_paper_pdf
from src.formatters.text import render_answer_key_text, render_question_paper_text
from src.schemas.output import QuestionPaper


@dataclass
class PaperArtifacts:
    question_paper_txt: str
    answer_key_txt: str
    question_paper_pdf: bytes
    answer_key_pdf: bytes


class FormattingAgent:
    """Deterministic layout agent: JSON -> printable text PDF and TXT."""

    role = "Formatting and Layout Agent"
    goal = "Convert a validated question paper into exam-style PDF and plain text."

    def render(self, paper: QuestionPaper) -> PaperArtifacts:
        return PaperArtifacts(
            question_paper_txt=render_question_paper_text(paper),
            answer_key_txt=render_answer_key_text(paper),
            question_paper_pdf=render_question_paper_pdf(paper),
            answer_key_pdf=render_answer_key_pdf(paper),
        )
