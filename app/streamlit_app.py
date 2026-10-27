from __future__ import annotations

import html
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CHROMA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    SYLLABUS_DIR,
    ensure_data_dirs,
    get_api_key,
)
from src.crews.paper_crew import run_paper_pipeline  # noqa: E402
from src.rag.ingest import ingest_pdfs  # noqa: E402
from src.schemas.input import PaperSpec, SectionSpec  # noqa: E402

st.set_page_config(
    page_title="Multi-agent question paper generation and validation system",
    layout="wide",
    initial_sidebar_state="expanded",
)
ensure_data_dirs()

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stSidebar"],
.block-container, p, h1, h2, h3, h4, label, span, div, input, textarea, button,
.stMarkdown, .stCaption, [data-testid="stHeading"], [data-testid="stWidgetLabel"] {
  font-family: Inter, Helvetica, Arial, sans-serif !important;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: #ffffff !important;
  color: #111 !important;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid #ececec;
}

.block-container {
  max-width: 1080px;
  padding: 2.2rem 2.6rem 4rem !important;
  background: #ffffff;
}

.hero-title {
  font-weight: 500;
  font-size: clamp(2.2rem, 5.6vw, 4.4rem);
  line-height: 1.05;
  letter-spacing: -0.03em;
  color: #111;
  margin: 0 0 1rem 0;
}
.hero-row {
  display: grid;
  grid-template-columns: 1.4fr 0.7fr;
  gap: 2rem;
  align-items: end;
  margin: 0.4rem 0 3rem 0;
}
.hero-lede {
  font-size: 13px;
  line-height: 1.55;
  color: #333;
  max-width: 16rem;
  margin: 0 0 0.4rem auto;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button,
button[kind="primary"],
button[kind="secondary"],
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"],
[data-testid="stFileUploader"] button {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #111 !important;
  border: 1px solid #111 !important;
  border-radius: 2px !important;
  font-family: Inter, Helvetica, Arial, sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em;
  padding: 0.45rem 1.1rem !important;
  box-shadow: none !important;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
  background: #f4f4f4 !important;
  color: #111 !important;
  border-color: #111 !important;
}
div[data-testid="stButton"] > button:disabled,
button[kind="primary"]:disabled {
  background: #ffffff !important;
  color: #999 !important;
  border-color: #ccc !important;
}

.agent-log {
  border-top: 1px solid #ddd;
  border-bottom: 1px solid #ddd;
  padding: 1.1rem 0;
  margin: 0.6rem 0 1.4rem;
}
.agent-step {
  font-size: 14px;
  line-height: 1.7;
  color: #888;
}
.agent-step.current {
  color: #111;
  font-weight: 500;
}
.agent-step.done::before { content: "·  "; color: #bbb; }
.agent-step.current::before { content: "→  "; color: #111; }
"""


def _inject_css() -> None:
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


def _render_agent_log(messages: list[str], *, live: bool = False) -> None:
    if not messages:
        return
    items = []
    last = len(messages) - 1
    for i, message in enumerate(messages):
        klass = "agent-step current" if live and i == last else "agent-step done"
        items.append(f'<div class="{klass}">{html.escape(message)}</div>')
    st.markdown(
        '<div class="agent-log">' + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )


EMPTY_SECTION = {
    "section_name": "",
    "question_type": None,
    "count": None,
    "marks_per_question": None,
    "difficulty": None,
    "blooms_level": None,
}


def _rows_as_dicts(edited) -> list[dict]:
    if edited is None:
        return []
    if hasattr(edited, "to_dict"):
        return edited.to_dict("records")
    return list(edited)


def _is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _section_marks_sum(rows: list[dict]) -> int | None:
    total = 0
    for row in rows:
        count = row.get("count")
        marks = row.get("marks_per_question")
        if count in (None, "") or marks in (None, ""):
            return None
        try:
            total += int(count) * int(marks)
        except (TypeError, ValueError):
            return None
    return total


def _spec_blocking_errors(
    subject: str,
    total_marks,
    time_allowed,
    rows: list[dict],
) -> list[str]:
    errors: list[str] = []
    if not (subject or "").strip():
        errors.append("Subject is required.")
    if total_marks is None:
        errors.append("Total marks is required.")
    elif int(total_marks) < 1:
        errors.append("Total marks must be at least 1.")
    if time_allowed is None:
        errors.append("Time allowed is required.")
    elif int(time_allowed) < 1:
        errors.append("Time allowed must be at least 1 minute.")

    filled_rows = [row for row in rows if any(_is_filled(v) for v in row.values())]
    if not filled_rows:
        errors.append("Add at least one section with question count and marks.")
        return errors

    required = (
        "section_name",
        "question_type",
        "count",
        "marks_per_question",
        "difficulty",
        "blooms_level",
    )
    for i, row in enumerate(filled_rows, start=1):
        missing = [key.replace("_", " ") for key in required if not _is_filled(row.get(key))]
        if missing:
            errors.append(f"Section {i} is incomplete: fill {', '.join(missing)}.")
            continue
        try:
            if int(row["count"]) < 1 or int(row["marks_per_question"]) < 1:
                errors.append(f"Section {i}: count and marks per question must be at least 1.")
        except (TypeError, ValueError):
            errors.append(f"Section {i}: count and marks per question must be numbers.")

    computed = _section_marks_sum(filled_rows)
    if (
        total_marks is not None
        and computed is not None
        and not any("incomplete" in e or "must be" in e for e in errors)
        and int(computed) != int(total_marks)
    ):
        errors.append(
            f"Marks mismatch: sections add up to {computed}, but total marks is {int(total_marks)}."
        )
    return errors


def _persist_dir() -> Path:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return CHROMA_DIR / st.session_state.session_id


def _save_uploads(files) -> list[Path]:
    saved: list[Path] = []
    for uploaded in files:
        dest = SYLLABUS_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        saved.append(dest)
    return saved


def _default_pdf() -> Path | None:
    candidate = SYLLABUS_DIR / "Module-2.pdf"
    return candidate if candidate.exists() else None


_inject_css()
st.markdown(
    """
    <div class="hero-row">
      <h1 class="hero-title">Multi-agent<br>question paper<br>generation and<br>validation system</h1>
      <p class="hero-lede">Retrieve from the syllabus, draft questions, audit them, and export a text PDF. Settings stay in the sidebar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(
        "OpenRouter API key",
        type="password",
        value="",
        help="Leave blank to use OPENROUTER_API_KEY from .env",
    )
    model_name = st.text_input("Chat model", value=DEFAULT_MODEL)
    embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
    verbose = st.checkbox("Verbose agent logs", value=False)
    if st.button("Clear vector store for this session"):
        persist = _persist_dir()
        if persist.exists():
            import shutil

            shutil.rmtree(persist, ignore_errors=True)
        st.session_state.pop("ingest_stats", None)
        st.success("Session vector store cleared.")

try:
    api_key = get_api_key(api_key_input or None)
    st.sidebar.success("API key loaded")
except RuntimeError as exc:
    api_key = ""
    st.sidebar.warning(str(exc))

st.subheader("1. Syllabus")
default_pdf = _default_pdf()
if default_pdf:
    st.info(f"Bundled chapter ready: `{default_pdf.name}`. You can ingest it or upload another PDF.")

uploads = st.file_uploader(
    "Upload syllabus / textbook PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns(2)
with col_a:
    ingest_uploaded = st.button("Ingest uploaded PDFs", disabled=not uploads)
with col_b:
    ingest_default = st.button(
        "Ingest bundled Module-2.pdf",
        disabled=default_pdf is None,
    )

if ingest_uploaded or ingest_default:
    if not api_key:
        st.error("Add an OpenRouter API key first.")
        st.stop()
    try:
        if ingest_uploaded:
            paths = _save_uploads(uploads)
        else:
            paths = [default_pdf]  # type: ignore[list-item]
        with st.spinner("Chunking and embedding syllabus..."):
            stats = ingest_pdfs(
                pdf_paths=paths,
                persist_dir=_persist_dir(),
                api_key=api_key,
                embedding_model=embedding_model,
            )
        st.session_state.ingest_stats = stats
        st.success(
            f"Ingested {stats['files']} file(s), {stats['pages']} pages, {stats['chunks']} chunks."
        )
    except Exception as exc:
        st.exception(exc)

if "ingest_stats" in st.session_state:
    st.caption(f"Last ingest: {st.session_state.ingest_stats}")

st.subheader("2. Paper specification")
subject = st.text_input("Subject", value="", placeholder="e.g. Software Engineering")
c1, c2 = st.columns(2)
with c1:
    total_marks = st.number_input(
        "Total marks",
        min_value=1,
        value=None,
        step=1,
        placeholder="e.g. 50",
    )
with c2:
    time_allowed = st.number_input(
        "Time allowed (minutes)",
        min_value=1,
        value=None,
        step=1,
        placeholder="e.g. 90",
    )

st.markdown("Sections — add rows and set question count and marks per question")
if "section_rows" not in st.session_state:
    st.session_state.section_rows = [EMPTY_SECTION.copy()]

edited = st.data_editor(
    st.session_state.section_rows,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "section_name": st.column_config.TextColumn("Section"),
        "question_type": st.column_config.SelectboxColumn(
            "Type", options=["MCQ", "Short Answer", "Long Answer"]
        ),
        "count": st.column_config.NumberColumn("Count", min_value=1, step=1),
        "marks_per_question": st.column_config.NumberColumn("Marks each", min_value=1, step=1),
        "difficulty": st.column_config.SelectboxColumn(
            "Difficulty", options=["Easy", "Medium", "Hard"]
        ),
        "blooms_level": st.column_config.SelectboxColumn(
            "Bloom level",
            options=["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
        ),
    },
)
rows = _rows_as_dicts(edited)
st.session_state.section_rows = rows

computed_marks = _section_marks_sum(
    [row for row in rows if any(_is_filled(v) for v in row.values())]
)
if computed_marks is not None:
    st.caption(f"Sum of section marks (count × marks each): **{computed_marks}**")

spec_errors = _spec_blocking_errors(subject, total_marks, time_allowed, rows)
if spec_errors:
    for message in spec_errors:
        st.error(message)

can_generate = not spec_errors and bool(api_key)
generate = st.button("Generate paper", disabled=not can_generate)
if not api_key:
    st.caption("Generation is disabled until an OpenRouter API key is available.")

if generate:
    if spec_errors:
        st.stop()
    if not api_key:
        st.error("Add an OpenRouter API key first.")
        st.stop()
    filled_rows = [row for row in rows if any(_is_filled(v) for v in row.values())]
    try:
        spec = PaperSpec(
            subject_name=subject.strip(),
            total_marks=int(total_marks),
            time_allowed_mins=int(time_allowed),
            sections=[SectionSpec.model_validate(row) for row in filled_rows],
        )
    except Exception as exc:
        st.error(f"Invalid paper spec: {exc}")
        st.stop()

    persist = _persist_dir()
    if not persist.exists():
        st.error("Ingest a syllabus PDF before generating.")
        st.stop()

    st.markdown("### Agent progress")
    progress_slot = st.empty()
    live_log: list[str] = []

    def _on_progress(message: str) -> None:
        live_log.append(message)
        with progress_slot.container():
            _render_agent_log(live_log, live=True)

    try:
        result = run_paper_pipeline(
            spec=spec,
            persist_dir=persist,
            api_key=api_key,
            model=model_name,
            verbose=verbose,
            on_progress=_on_progress,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()
    progress_slot.empty()
    st.session_state.result = result
    st.session_state.agent_log = result.agent_log

if "result" in st.session_state:
    result = st.session_state.result
    paper = result.paper
    artifacts = result.artifacts
    report = paper.validation_report

    st.subheader("3. Results")
    if st.session_state.get("agent_log"):
        st.markdown("#### Agent run")
        _render_agent_log(st.session_state.agent_log, live=False)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status", report.status)
    m2.metric("Attempts", result.attempts)
    m3.metric("Marks verified", "Yes" if report.total_marks_verified else "No")
    m4.metric("Coverage", f"{report.syllabus_coverage_score:.2f}")

    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(
        "Download question paper PDF",
        data=artifacts.question_paper_pdf,
        file_name="question_paper.pdf",
        mime="application/pdf",
    )
    d2.download_button(
        "Download question paper TXT",
        data=artifacts.question_paper_txt,
        file_name="question_paper.txt",
        mime="text/plain",
    )
    d3.download_button(
        "Download answer key PDF",
        data=artifacts.answer_key_pdf,
        file_name="answer_key.pdf",
        mime="application/pdf",
    )
    d4.download_button(
        "Download answer key TXT",
        data=artifacts.answer_key_txt,
        file_name="answer_key.txt",
        mime="text/plain",
    )

    tab_paper, tab_key, tab_audit = st.tabs(
        ["Question paper", "Answer key / solutions", "Validation report"]
    )
    with tab_paper:
        st.text(artifacts.question_paper_txt)
    with tab_key:
        st.text(artifacts.answer_key_txt)
    with tab_audit:
        st.write(f"**Status:** {report.status}")
        st.write(f"**Total marks verified:** {report.total_marks_verified}")
        st.write(f"**Syllabus coverage score:** {report.syllabus_coverage_score}")
        st.write("**Auditor notes**")
        st.text(report.auditor_notes)
        with st.expander("Curriculum briefing"):
            st.text(result.syllabus_brief)
