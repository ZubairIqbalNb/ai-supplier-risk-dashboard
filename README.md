# AI Supplier Risk Dashboard

Phase 2 multi-source retrieval project for the AI Engineering Masters Program.

**Current status: Modules 1–2.** The repository contains the project foundation
and configuration layer plus the Module 2 supplier data contracts and
deterministic synthetic dataset. No retrieval engines, data stores, DSPy
synthesis, evaluation pipeline, or UI are implemented yet.

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

## Module 2 dataset

Module 2 defines strict Pydantic contracts for supplier profiles, annual
delivery metrics, contract and complaint text, and ownership links. It also
generates a reproducible 24-supplier dataset with a 14 train / 4 development /
6 held-out split. Supplier-to-partition membership is fixed by ID range and
enforced during validation, so the counts cannot be preserved by swapping IDs
between partitions.

```bash
uv run --locked python -m supplier_risk.data.generate
uv run --locked pytest tests/test_contracts.py tests/test_dataset.py -q
```

The default output, `data/generated/suppliers.json`, is Git-ignored. The dataset
is entirely synthetic. Field definitions, units, validation rules, and
partition/leakage controls are in [`docs/schema.md`](docs/schema.md), which also
records proposed physical schemas for SQLite, FAISS metadata, and Neo4j. Those
store designs are not implemented in Module 2.

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

Modules 1–2 cover repository foundation, configuration, canonical supplier
data contracts, and a deterministic synthetic dataset. Later modules add the
data stores, retrieval, DSPy synthesis, evaluation, and UI in separately
reviewed steps. Module 2 does not populate any store or implement risk scoring.

Deployment is deferred until the missing “Vibe Coding Helper Doc” is supplied
or deployment is explicitly waived.

## Limitations

- The 0–100 risk score, evidence-status field, citation schema, and banned-word
  list are provisional engineering choices and are not implemented yet.
- The 24-supplier dataset and 14 train / 4 development / 6 held-out split are
  implemented but remain provisional design choices. The assignment only
  requires ~20–30 entities and a held-out entity outside training.
- Ownership currently supports one optional parent link per supplier. Multiple
  parents, nested ownership, and traversal depth remain open design decisions
  for the Neo4j module.
- Synthetic data cannot support claims of meaningful model accuracy.
- DSPy, Gemini model access, CPU PyTorch/FAISS compatibility, and Neo4j driver
  compatibility have not yet been verified.
