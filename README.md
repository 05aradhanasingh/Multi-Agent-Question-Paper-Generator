# Multi-Agent Question-Paper Generation System

Streamlit app that uses four agents (CrewAI + OpenRouter) to retrieve syllabus content, draft a question paper, audit it, and export **PDF + TXT** only.

Agents:

1. **Curriculum & Retrieval** — searches ingested textbook/syllabus chunks (no off-syllabus invention).
2. **Question Generator** — writes MCQ / Short / Long questions to a JSON schema with answers and solutions.
3. **Validator & Auditor** — deterministic mark/schema checks plus an LLM critic; generator can be asked to revise up to twice.
4. **Formatting & Layout** — exam-style question paper and answer key as selectable-text PDF and plain text.

## What you need

- Python 3.11+
- An OpenRouter API key
- At least one syllabus PDF (a Software Engineering Module-2 chapter is the default demo source)

## Setup

```bash
cd "agentic ai project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your key in `.env`:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
```

Do not commit `.env`.

Copy your chapter PDF to:

```
data/syllabus/Module-2.pdf
```

Or upload it in the app.

## Run

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

1. Ingest the syllabus PDF.
2. Keep or edit the default 50-mark / 90-minute spec (5 MCQs, 4 short, 2 long).
3. Generate.
4. Download `question_paper.pdf` / `.txt` and `answer_key.pdf` / `.txt`.

There is no Markdown or LaTeX export.

## Demo spec

See `examples/sample_spec.json`. Default subject is Software Engineering, 50 marks, 90 minutes.

## Notes

- Chat completions go through OpenRouter (`https://openrouter.ai/api/v1`), not the OpenAI platform directly.
- Vector stores are per browser session under `data/chroma/`.
- `data/` is gitignored so textbooks and embeddings stay local.
- If OpenRouter embeddings are unavailable, ingest falls back to a local embedding model so retrieval still works. Question generation still needs a funded OpenRouter key.
- Keys pasted into chat should be treated as exposed; rotate them after demos.
