from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["MCQ", "Short Answer", "Long Answer"]
Difficulty = Literal["Easy", "Medium", "Hard"]


class SectionSpec(BaseModel):
    section_name: str
    question_type: QuestionType
    count: int = Field(ge=1)
    marks_per_question: int = Field(ge=1)
    difficulty: Difficulty
    blooms_level: str


class PaperSpec(BaseModel):
    subject_name: str
    total_marks: int = Field(ge=1)
    time_allowed_mins: int = Field(ge=1)
    sections: list[SectionSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def marks_must_match_sections(self) -> "PaperSpec":
        computed = sum(s.count * s.marks_per_question for s in self.sections)
        if computed != self.total_marks:
            raise ValueError(
                f"Section marks ({computed}) do not equal total_marks ({self.total_marks})."
            )
        return self
