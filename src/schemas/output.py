from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionKind = Literal["MCQ", "Short", "Long"]
Difficulty = Literal["Easy", "Medium", "Hard"]
ValidationStatus = Literal["APPROVED", "REVISED"]


class PaperMetadata(BaseModel):
    title: str
    total_marks: int
    time_allowed_mins: int


class Question(BaseModel):
    id: str
    section: str
    type: QuestionKind
    topic: str
    question_text: str
    options: list[str] | None = None
    correct_answer: str
    step_by_step_solution: str
    marks: int
    difficulty: Difficulty
    blooms_level: str
    source_excerpt: str | None = None
    retrieval_score: float | None = None


class ValidationReport(BaseModel):
    status: ValidationStatus
    total_marks_verified: bool
    syllabus_coverage_score: float = Field(ge=0.0, le=1.0)
    auditor_notes: str


class QuestionPaper(BaseModel):
    paper_metadata: PaperMetadata
    questions: list[Question]
    validation_report: ValidationReport
