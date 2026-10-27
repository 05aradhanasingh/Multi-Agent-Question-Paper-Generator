from __future__ import annotations

from fpdf import FPDF

from src.formatters.text import render_answer_key_text, render_question_paper_text
from src.schemas.output import QuestionPaper


class ExamPDF(FPDF):
    def header(self) -> None:
        return None

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=9)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def _latin(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def render_text_pdf(body: str) -> bytes:
    pdf = ExamPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    for raw_line in body.splitlines():
        line = _latin(raw_line)
        if not line.strip():
            pdf.ln(4)
            continue
        is_title = raw_line.isupper() and len(raw_line.strip()) > 8
        is_section = raw_line.startswith("Section ")
        if is_title:
            pdf.set_font("Helvetica", style="B", size=14)
        elif is_section:
            pdf.set_font("Helvetica", style="B", size=12)
        else:
            pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(usable, 6, line)
        pdf.ln(0.5)
    return bytes(pdf.output())


def render_question_paper_pdf(paper: QuestionPaper) -> bytes:
    return render_text_pdf(render_question_paper_text(paper))


def render_answer_key_pdf(paper: QuestionPaper) -> bytes:
    return render_text_pdf(render_answer_key_text(paper))
