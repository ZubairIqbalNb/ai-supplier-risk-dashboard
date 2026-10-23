# AI Supplier Risk Dashboard

Phase 2 multi-source retrieval project for the AI Engineering Masters Program.

**Current status: Modules 1–3.** The repository contains the project foundation
and configuration layer, the Module 2 supplier data contracts and deterministic
synthetic dataset, and the Module 3 document-retrieval stack (document corpus,
FAISS dense index, BM25 lexical index, and hybrid Reciprocal Rank Fusion
retrieval). SQLite population, Neo4j graph retrieval, parallel orchestration,
DSPy synthesis, the judge metric, and the UI are **not** implemented yet.

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
- A CPU is sufficient. Dense retrieval uses the CPU build of PyTorch and
  `faiss-cpu`; no CUDA toolkit or GPU is required.
- Network access on first index build to download
  `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) into the Hugging Face
  cache outside the repository. Later builds and queries reuse the cache.
- Docker for Neo4j, beginning in a later module
- A Gemini API key, beginning in Module 6

## Setup

```bash
uv sync --locked --group dev
```

`pyproject.toml` maps `torch` to the PyTorch CPU wheel index
(`https://download.pytorch.org/whl/cpu`, `explicit = true`), so `uv` resolves
CPU-only wheels and the lockfile contains no CUDA or `nvidia-*` packages.

Copy `.env.example` to `.env` and fill in credentials locally. Never commit
`.env`. The file is ignored by Git and should remain mode `600`.
Module 3 does not read any environment variable.

## Environment variables

| Variable | Required from | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Module 1 (used from Module 6) | Gemini API credential; required for configuration validation |
| `GEMINI_TEACHER_MODEL` | Module 6 | Stronger teacher model |
| `GEMINI_STUDENT_MODEL` | Module 6 | Cheaper student model |
| `NEO4J_URI` | Later graph module | Neo4j Bolt endpoint |
| `NEO4J_USER` | Later graph module | Neo4j user |
| `NEO4J_PASSWORD` | Later graph module | Neo4j password |
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
partition/leakage controls are in [`docs/schema.md`](docs/schema.md).

## Module 3 document retrieval

Every contract and complaint record becomes exactly one searchable document
(66 documents for the default seed: 32 contracts, 34 complaints). The document
id is derived from the source record, for example `contract:CON-001-01` or
`complaint:CMP-001-01`, so it is identical across repeated builds. Each document
keeps `supplier_id`, supplier name, partition, source type, source id, the
searchable text, and the domain metadata that exists on the source record.

| Component | Implementation |
|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, CPU, loaded lazily |
| Vector index | FAISS `IndexFlatIP` over L2-normalized vectors (cosine similarity) |
| Lexical index | `rank_bm25.BM25Okapi`, `k1=1.5`, `b=0.75`, tokenizer `module3.v1` |
| Fusion | Reciprocal Rank Fusion, `RRF(d) = Σ 1 / (k + rank_i(d))`, `k = 60`, one-based ranks |
| Filters | exact `supplier_id` and `partition` metadata filters |
| Artifacts | `artifacts/retrieval/` (Git-ignored) |

Build the artifacts (explicit operation; nothing is built during a query):

```bash
uv run --locked python -m supplier_risk.retrieval.build
```

Query with the generic interface or the assignment-facing supplier wrapper:

```python
from supplier_risk.retrieval import HybridRetriever, query_faiss

retriever = HybridRetriever.load("artifacts/retrieval")
hits = retriever.query(
    "quality inspection records", top_k=5, supplier_id="SUP-001", partition="train"
)
for hit in hits:
    print(hit.document.document_id, hit.fusion_score, hit.dense_rank, hit.lexical_rank)

print(query_faiss("Norvend Components Ltd", top_k=3))
```

`query_faiss(entity)` accepts a supplier id (`SUP-001`) or an exact legal name
(case-insensitive) and returns a labeled `[FAISS]` context string covering
contract and complaint documents only. It reports unknown suppliers and
unavailable artifacts as `[FAISS ERROR]` strings instead of raising.

`query_faiss` uses `get_default_retriever(artifacts_dir)`, which caches the
loaded retriever per artifact path for the life of the process. After rebuilding
artifacts in a long-running process, call `reset_default_retriever()` before the
next query, or restart the process.

Supplying your own embedder to `HybridRetriever.load(..., embedder=...)` requires
the model to match the persisted index: the declared model name is always
checked, and an embedder that also exposes `declared_dimension` has its dimension
checked too. Query vectors are still validated against the index dimension for
every embedder.

Artifact layout (`artifacts/retrieval/`):

| File | Contents |
|---|---|
| `corpus.json` | Every document with identity, metadata, text, and a SHA-256 corpus fingerprint |
| `dense.index.faiss` | FAISS `IndexFlatIP`, one vector per document, normalized |
| `dense.metadata.json` | Embedding model, dimension, metric, fingerprint, and vector id → document map |
| `lexical.json` | BM25 parameters, tokenizer version, document ids, and token lists |
| `manifest.json` | Written last; ties the artifact set together and records all compatibility fields |

Loading validates the manifest, corpus fingerprint, document counts, index
dimension, vector/document alignment, the duplicated dense identity fields
(`document_id`, `supplier_id`, `partition`, `source_type`), embedding model,
tokenizer version, BM25 parameters, and each lexical token list against the
tokens the current tokenizer produces from the persisted corpus text. Missing,
stale, or corrupt artifacts raise `RetrievalArtifactError` with an actionable
message; querying never rebuilds an index. Metadata and lexical data are stored
as JSON, never pickle.

### Tests

```bash
uv run --locked pytest -q                       # offline suite; no model download
uv run --locked pytest -q tests/test_retrieval_pipeline.py
SUPPLIER_RISK_REAL_MODEL_TESTS=1 uv run --locked pytest -q -m real_model
```

The real-model smoke tests download and run `all-MiniLM-L6-v2`; they are
skipped unless `SUPPLIER_RISK_REAL_MODEL_TESTS=1` is set. Everything else uses
deterministic stub embedders.

## Commands

```bash
make install          # install core and development dependencies
make lock             # regenerate uv.lock
make index            # build the Module 3 retrieval artifacts
make test             # run pytest
make test-real-model  # opt-in real embedding model smoke tests
make lint             # run Ruff
make typecheck        # run mypy
make check            # lint + typecheck + test
```

## Security

- `.env` is Git-ignored and must remain mode `600`.
- Secret values are loaded through `pydantic.SecretStr` and masked in
  configuration representations and error messages.
- Tests use injected environment mappings and never read real credentials.
- Module 3 performs no external LLM calls and reads no environment variables.
- Retrieved documents are fictional synthetic data; retrieval errors never
  include credentials.
- No secret values should appear in logs, documentation, or test output.

## Scope

Modules 1–2 cover the repository foundation, configuration, canonical supplier
data contracts, and a deterministic synthetic dataset. Module 3 adds the
document corpus, FAISS and BM25 indexes, RRF hybrid retrieval, metadata
filtering, and `query_faiss(entity)`. SQLite population, Neo4j nodes and edges,
parallel multi-engine orchestration, DSPy synthesis, the judge metric, and the
UI remain future work and are not claimed here.

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
  for the graph module.
- Synthetic data cannot support claims of meaningful model accuracy, and the
  retrieved contract and complaint texts come from a small template set.
- Retrieval uses exact FAISS search over 66 documents; recall and latency at
  larger corpus sizes are unmeasured. The `4 × top_k` candidate window and
  `RRF k = 60` are documented defaults, not tuned values.
- A build writes its artifact files one at a time through atomic per-file
  replacement. A crash between files can leave a mixed set; the manifest is
  written last and the fingerprint, model, dimension, and count checks reject
  such a set instead of serving it.
- DSPy, Gemini model access, and Neo4j driver compatibility have not been
  verified.
