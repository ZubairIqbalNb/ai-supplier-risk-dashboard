# AI Supplier Risk Dashboard

Phase 2 multi-source retrieval project for the AI Engineering Masters Program.

**Current status: Module 1 only.** This repository currently contains the
project foundation and configuration layer. No retrieval engines, data stores,
DSPy synthesis, evaluation pipeline, or UI are implemented yet.

## Assignment domain

Supplier Risk Dashboard:

- SQLite: delivery and defect metrics
- FAISS + BM25: contract and complaint text
- Neo4j: subsidiary and ownership graph
- Verdict: `APPROVE` / `FLAG`

The full verified requirements traceability table is in
[`docs/ASSIGNMENT_TRACEABILITY.md`](docs/ASSIGNMENT_TRACEABILITY.md).

## Requirements

- Python 3.12
- `uv` 0.13 or newer
- Docker for Neo4j, beginning in Module 4
- A Gemini API key, beginning in Module 6

## Setup

```bash
uv sync --locked --group dev
```

Copy `.env.example` to `.env` and fill in credentials locally. Never commit
`.env`. The file is ignored by Git and should remain mode `600`.

## Environment variables

| Variable | Required from | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Module 1 (used from Module 6) | Gemini API credential; required for configuration validation |
| `GEMINI_TEACHER_MODEL` | Module 6 | Stronger teacher model |
| `GEMINI_STUDENT_MODEL` | Module 6 | Cheaper student model |
| `NEO4J_URI` | Module 4 | Neo4j Bolt endpoint |
| `NEO4J_USER` | Module 4 | Neo4j user |
| `NEO4J_PASSWORD` | Module 4 | Neo4j password |
| `LOG_LEVEL` | Module 1 | Python logging level |

An existing `GEMINI_MODEL` value in a local `.env` is left untouched but is not
used by the final teacher/student model split.

## Commands

```bash
make install      # install core and development dependencies
make lock         # regenerate uv.lock
make test         # run pytest
make lint         # run Ruff
make typecheck    # run mypy
make check        # lint + typecheck + test
```

## Security

- `.env` is Git-ignored and must remain mode `600`.
- Secret values are loaded through `pydantic.SecretStr` and masked in
  configuration representations and error messages.
- Tests use injected environment mappings and never read real credentials.
- No secret values should appear in logs, documentation, or test output.

## Scope

Module 1 is limited to repository foundation, configuration, and tests.
Later modules add the data stores, retrieval, DSPy synthesis, evaluation, and
UI in separately reviewed steps.

Deployment is deferred until the missing “Vibe Coding Helper Doc” is supplied
or deployment is explicitly waived.

## Limitations

- The 0–100 risk score, evidence-status field, citation schema, and banned-word
  list are provisional engineering choices.
- The dataset split is provisional at 14 train / 4 development / 6 held-out.
- DSPy, Gemini model access, CPU PyTorch/FAISS compatibility, and Neo4j driver
  compatibility have not yet been verified.
