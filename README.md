# AI Supplier Risk Dashboard

Phase 2 multi-source retrieval project for the AI Engineering Masters Program.

**Current status: Modules 1–6.** The repository contains the project foundation
and configuration layer, the Module 2 supplier data contracts and deterministic
synthetic dataset, the Module 3 document-retrieval stack (document corpus,
FAISS dense index, BM25 lexical index, and hybrid Reciprocal Rank Fusion
retrieval), the Module 4 Neo4j ownership graph with `query_graph(entity)`, the
Module 5 SQLite delivery/defect store, `query_sql(entity)`, and parallel
three-source orchestration with `query_all_sources(entity)`, and the Module 6
DSPy structured verdict plus deterministic rule-based judge. DSPy optimization
and compilation, held-out evaluation, and the UI are **not** implemented yet.

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
- Docker with Compose for the local Neo4j 5 service, from Module 4
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
Module 3 does not read any environment variable. Module 4 reads only
`NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`; Docker Compose reads the same
three values for the local service. Module 5 adds no new variables: the SQLite
store path defaults to `data/generated/suppliers.sqlite` and can be overridden
per call. Module 6 reads `GEMINI_API_KEY`, `GEMINI_TEACHER_MODEL`, and
`GEMINI_STUDENT_MODEL` only when `load_language_model_settings()` is called; its
tests inject their own mappings and never read `.env`.

## Environment variables

| Variable | Required from | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Module 1 (used from Module 6) | Gemini API credential; required for configuration validation |
| `GEMINI_TEACHER_MODEL` | Module 6 | Stronger teacher model |
| `GEMINI_STUDENT_MODEL` | Module 6 | Cheaper student model |
| `NEO4J_URI` | Module 4 | Neo4j Bolt endpoint (default `bolt://localhost:7687`) |
| `NEO4J_USER` | Module 4 | Neo4j user (default `neo4j`) |
| `NEO4J_PASSWORD` | Module 4 | Neo4j password; required, never committed or logged |
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

## Module 4 supplier ownership graph

The canonical dataset contains 24 suppliers and 24 ownership links to 10 parent
companies. Parent companies are separate `ORG-NNN` entities, never suppliers,
so the faithful graph is:

```cypher
(:Supplier)-[:SUBSIDIARY_OF|:JOINT_VENTURE_WITH {ownership_percentage}]->(:Company)
```

The edge is always stored child supplier → parent company, matching the
`child_supplier_id` and `parent_company_id` contract fields. `SUBSIDIARY_OF`
requires a percentage above 50; `JOINT_VENTURE_WITH` requires at most 50.

### Start and verify Neo4j

The local service uses Neo4j 5 Community, a named data volume, and a health
check. Ports are published only on the loopback interface.

```bash
docker compose up -d neo4j     # or: make neo4j-up
docker compose ps neo4j        # STATUS should become (healthy)
docker compose stop neo4j      # or: make neo4j-down
```

`compose.yaml` reads `NEO4J_USER` and `NEO4J_PASSWORD` from the Git-ignored
`.env`; no credential is stored in the Compose file or the image. The health
check runs `cypher-shell ... "RETURN 1"`.

Changing `NEO4J_PASSWORD` after the data volume has been initialized does not
change the database user's password. Update it with `cypher-shell`'s
`ALTER CURRENT USER SET PASSWORD` or deliberately recreate the volume; do not
delete the volume as a shortcut.

### Ingest the canonical graph

```bash
uv run --locked python -m supplier_risk.graph.ingest   # or: make graph-ingest
```

Ingestion loads and validates the generated dataset, checks that existing
ownership edges for canonical suppliers do not contradict it, creates the two
uniqueness constraints, and writes suppliers, companies, and edges in one
transaction. Re-running it against the same dataset creates nothing. A
duplicate logical edge, a different parent, or a different relationship type is
reported as a conflict instead of being silently overwritten or deleted;
nodes and edges outside the current dataset are left untouched.

### Query the graph

```python
from supplier_risk.graph import query_graph

print(query_graph("SUP-007"))
print(query_graph("Sakurajima Precision KK"))
```

`query_graph(entity)` accepts a supplier id (`SUP-007`, case-insensitive) or an
exact legal name (case-insensitive) and returns a labeled `[GRAPH]` string.
Unknown suppliers, invalid configuration, an unavailable database, timeouts,
and integrity failures return tagged `[GRAPH ERROR]` strings instead of
raising.

Traversal is bounded to **3 ownership hops** and **25 relationships**. Discovery
is undirected so a supplier's parent company and fellow subsidiaries are
reachable, but every returned edge keeps its stored direction
(child supplier → parent company). A valid supplier with no ownership edges
returns a normal `[GRAPH]` context with `ownership_relationships=0`. The
context never asserts legal control, sanctions, financial risk, or an
APPROVE/FLAG verdict.

Queries use a 10-second transaction timeout and a 5-second connection timeout;
there is no unbounded retry loop.

### Tests

```bash
uv run --locked pytest -q                        # offline suite; no Neo4j needed
uv run --locked pytest -q tests/test_graph_schema.py tests/test_graph_pipeline.py
SUPPLIER_RISK_NEO4J_TESTS=1 uv run --locked pytest -q -m neo4j   # or: make test-graph
```

The live tests read `NEO4J_*` from `.env`, use a test-only `SUP-9xx` / `ORG-99x`
namespace, and remove it afterwards. They never wipe the canonical development
graph.

## Module 5 structured retrieval and parallel orchestration

Module 5 adds the third engine and the orchestration layer. The SQLite store
holds only structured supplier facts — identity, partition, and one annual
delivery/defect observation per supplier — in `suppliers` and `delivery_metrics`.
Contract and complaint text stays in the Module 3 artifacts and ownership edges
stay in the Module 4 graph, so no engine duplicates another engine's evidence.

```bash
uv run --locked python -m supplier_risk.sql.build   # or: make sql-build
```

The build writes `data/generated/suppliers.sqlite` (Git-ignored): it validates a
deterministic load plan, inserts every row in one transaction, verifies the row
counts, and replaces the file atomically. Rebuilding is the only supported way
to refresh the store; a query never creates or modifies it.

```python
from supplier_risk.sql import query_sql
from supplier_risk.orchestration import query_all_sources

print(query_sql("SUP-001"))
result = query_all_sources("SUP-007")
print(result.outcome, result.failed_sources)
print(result.sql.context)
print(result.faiss.context)
print(result.graph.context)
```

`query_all_sources(entity)` submits `query_sql`, `query_faiss`, and
`query_graph` to a `ThreadPoolExecutor` before waiting for any of them, then
collects each outcome independently. The result exposes the requested entity,
three `SourceEvidence` sections in fixed `SQL` / `FAISS` / `GRAPH` order,
per-source status (`ok`, `empty`, or `error`), the tagged error string when a
source failed, per-source and overall elapsed time, and an `outcome` of
`complete`, `partial`, or `failed`. `result.context()` joins the labeled sections
without collapsing provenance, and no risk score or verdict is produced.

| Situation | Result |
|---|---|
| All three succeed | `complete`; all three labeled contexts |
| One source fails | `partial`; the other two contexts plus the tagged error |
| Two sources fail | `partial`; the surviving context plus both tagged errors |
| All three fail | `failed`; three tagged errors, no fabricated evidence |
| A source has no evidence | `empty` for that source; still `complete`, never an error |

SQLite opens one short-lived read-only connection per query, the FAISS and BM25
indexes are only read, and Neo4j opens its own store per query, so the three
default sources are safe to run concurrently. `timeout_seconds` bounds how long
the orchestrator waits; it does not terminate a source call that is already
running, so a timed-out source is reported as abandoned rather than killed.

### Tests

```bash
uv run --locked pytest -q tests/test_sql_schema.py tests/test_sql_pipeline.py
uv run --locked pytest -q tests/test_orchestration.py tests/test_orchestration_integration.py
```

The orchestration unit tests inject source callables and use
`threading.Barrier` to prove concurrent submission without relying on sleeps.
The offline integration tests run the real SQL and FAISS/BM25 engines with a
graph store double; the live Neo4j test joins the real graph source and is
opt-in with `SUPPLIER_RISK_NEO4J_TESTS=1`.

## Module 6 structured verdicts and deterministic judge

Module 6 adds the decision layer on top of `query_all_sources`: one DSPy
`ChainOfThought` call and one deterministic business-rule pass. There is no
optimizer, no compiled student, no training set, no retry, no second model call,
no agent loop, and no LLM-as-judge. Those belong to Module 7 or later.

### DSPy Signature

`SupplierRiskSignature` has exactly three evidence inputs and five structured
outputs.

| Direction | Field | Meaning |
|---|---|---|
| input | `sql_context` | Labeled `[SQL]` delivery/defect evidence or a `[SQL ERROR]` marker |
| input | `faiss_context` | Labeled `[FAISS]` contract/complaint evidence or a `[FAISS ERROR]` marker |
| input | `graph_context` | Labeled `[GRAPH]` ownership evidence or a `[GRAPH ERROR]` marker |
| output | `supplier_id` | The `SUP-NNN` identity stated in the contexts; compared with the requested supplier |
| output | `verdict` | `APPROVE` or `FLAG` |
| output | `rationale` | Concise, evidence-grounded explanation (20–800 characters) |
| output | `findings` | One to eight material evidence items naming their source |
| output | `missing_evidence` | Failed, absent, or incomplete sources that materially affect the decision |

The signature instructions require the model to use only the supplied evidence,
separate adverse from favorable and missing evidence, acknowledge conflicts and
failed sources, never treat a failed source as "no risk", and stay conservative
under uncertainty. There is no fourth context: the three-source split is the
architecture, not a convention.

`SupplierRiskVerdict` is the validated contract behind those outputs. It is
strict, frozen, forbids unknown fields, rejects any verdict other than
`APPROVE` / `FLAG`, and rejects a different supplier id. No risk score or
confidence value is invented.

### One-call decision path

```python
import dspy

from supplier_risk.config import load_language_model_settings
from supplier_risk.decision import build_role_language_models, decide_supplier

models = build_role_language_models(load_language_model_settings())
with dspy.context(lm=models.student):
    result = decide_supplier("SUP-007")

print(result.retrieval.outcome)        # complete / partial / failed
print(result.context["sql"])           # raw labeled SQL context
print(result.verdict.verdict)          # model verdict
print(result.final_verdict)            # judge's verdict
print(result.judgment.triggered_rules)
```

`decide_supplier` accepts a canonical supplier id (`SUP-NNN`); legal-name
resolution stays in the three store wrappers so identity comparison is exact. It
either runs `query_all_sources(entity)` itself or accepts an existing
`OrchestrationResult`. Exactly one prediction is made per request. A provider
timeout, provider exception, missing prediction, malformed output, or unknown
verdict raises a project error (`DecisionModelError`, `DecisionResponseError`,
`SupplierMismatchError`) and never becomes an `APPROVE`. Provider error text is
sanitized before it reaches an exception message or a log record.

Partial retrieval keeps its provenance: a failed source reaches the model (and
the result) as its tagged `[SQL ERROR]` / `[FAISS ERROR]` / `[GRAPH ERROR]`
string, while the other sources keep their own labeled contexts.

### Deterministic judge

`judge_verdict(verdict, retrieval, requested_supplier_id=...)` is a pure
business-rule check. It imports no DSPy module, calls no model, and is repeatable
for identical inputs.

| Rule | Meaning |
|---|---|
| `decision_invalid` | No usable validated verdict (missing, malformed, or outside the contract) |
| `supplier_identity_mismatch` | The verdict names a different supplier than requested |
| `unacknowledged_failed_source` | A failed source is not named in `missing_evidence` |
| `retrieval_failed` | All three sources failed, so no verdict is verifiable |
| `no_evidence_for_approval` | `APPROVE` although no source returned evidence |
| `banned_content` | The hard content rule: banned terms appear in the rationale or findings |

The banned-term rule is the Guide's Step 6 example ("at least one hard content
rule (e.g. banned words)") and the list is fixed in
`supplier_risk.decision.judge.BANNED_TERMS`: `sanction`, `blacklist`,
`guarantee`, `no risk`, `zero risk`, `legal control`, and `clean record`. The
three stores cannot establish those claims, so any occurrence fails the verdict.
The judge never upgrades a verdict; a rejected `APPROVE` becomes `FLAG`.

### Teacher and student models

`load_language_model_settings()` reads `GEMINI_API_KEY`, `GEMINI_TEACHER_MODEL`,
and `GEMINI_STUDENT_MODEL` through the existing configuration loader and raises a
`ConfigError` naming any missing variable. `build_role_language_models()` returns
two distinct, validated `dspy.LM` objects; a value without a provider prefix gets
`gemini/`, and a value such as `openai/gpt-4o-mini` is used unchanged. Nothing
calls `dspy.configure`, so the LM is scoped with `dspy.context` and cannot leak
between tests or requests. Module 6 never calls the teacher; the ordinary
decision path uses only the student role.

### Tests

```bash
uv run --locked pytest -q \
  tests/test_decision_signature.py tests/test_decision_module.py \
  tests/test_decision_judge.py tests/test_decision_integration.py \
  tests/test_decision_lm.py
```

All Module 6 tests are offline: a recording fake predictor replaces the LM, a
fake LM factory replaces `dspy.LM`, and `tests/decision_support.py` builds
Module 5 results and fake sources. No Docker, Neo4j, network, or credential is
needed. The integration tests run the real Module 5 orchestrator over those fake
sources.

**Deferred to Module 7:** `BootstrapFewShot` (or another optimizer), teacher
demonstrations, compiled/saved student state, held-out before/after comparison,
and judge-driven optimization. Module 6 provides the signature, the one-call
decision path, the verdict contract, and the judge those steps will use.

## Commands

```bash
make install          # install core and development dependencies
make lock             # regenerate uv.lock
make index            # build the Module 3 retrieval artifacts
make neo4j-up         # start the local Neo4j service
make neo4j-down       # stop the local Neo4j service without deleting data
make graph-ingest     # load the canonical ownership graph
make sql-build        # build the SQLite delivery/defect store
make test             # run pytest
make test-graph       # opt-in live Neo4j integration tests
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
- Module 4 reads `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`; the password
  uses `SecretStr`, is never printed, and is not included in exception messages.
- Compose reads the same credentials from `.env`; no secret is committed.
- Retrieved documents and ownership records are fictional synthetic data;
  retrieval errors never include credentials.
- Module 6 reads no environment variable at import time.
  `load_language_model_settings()` reads the three `GEMINI_*` values, holds the
  key as `SecretStr`, and never prints it. The decision layer sanitizes provider
  exception text and logs only supplier id, statuses, and character counts.
- No secret values should appear in logs, documentation, or test output.

## Scope

Modules 1–2 cover the repository foundation, configuration, canonical supplier
data contracts, and a deterministic synthetic dataset. Module 3 adds the
document corpus, FAISS and BM25 indexes, RRF hybrid retrieval, metadata
filtering, and `query_faiss(entity)`. Module 4 adds the Neo4j ownership graph,
deterministic ingestion, and `query_graph(entity)`. Module 5 adds the SQLite
delivery/defect store, `query_sql(entity)`, and `query_all_sources(entity)`, the
parallel three-source orchestrator with partial-failure tolerance. Module 6 adds
the DSPy `ChainOfThought` verdict module, the strict `SupplierRiskVerdict`
contract, the `decide_supplier` path, teacher/student LM configuration, and the
deterministic rule-based judge. DSPy optimization and compilation, held-out
evaluation, and the UI remain future work and are not claimed here.

Deployment is deferred until the missing “Vibe Coding Helper Doc” is supplied
or deployment is explicitly waived.

## Limitations

- The verdict field names, the banned-term list, and the rationale/findings
  length bounds are provisional engineering choices. The 0–100 risk score,
  evidence-status field, and citation schema remain unimplemented.
- Module 6 enforces the hard content rule as a fixed banned-term list. The
  rationale and findings are not otherwise checked against the contexts: there
  is no citation verifier, and the judge acknowledges a failed source only by
  the source's name or `[LABEL]` tag in `missing_evidence`, without checking
  that the item describes that source accurately.
- ChainOfThought's `reasoning` field is discarded rather than audited; the
  user-facing rationale is the model's summary, not verified internal reasoning.
- The 24-supplier dataset and 14 train / 4 development / 6 held-out split are
  implemented but remain provisional design choices. The assignment only
  requires ~20–30 entities and a held-out entity outside training.
- Ownership currently supports one optional parent link per supplier. The
  canonical dataset has depth 1 (supplier → company); the query is bounded to
  3 hops for future hierarchy data, but no company-to-company edges are
  ingested from the current contract. Multiple parents and nested ownership
  remain open design decisions.
- Synthetic data cannot support claims of meaningful model accuracy, and the
  retrieved contract and complaint texts come from a small template set.
- Retrieval uses exact FAISS search over 66 documents; recall and latency at
  larger corpus sizes are unmeasured. The `4 × top_k` candidate window and
  `RRF k = 60` are documented defaults, not tuned values.
- A build writes its artifact files one at a time through atomic per-file
  replacement. A crash between files can leave a mixed set; the manifest is
  written last and the fingerprint, model, dimension, and count checks reject
  such a set instead of serving it.
- The Neo4j driver is verified against Neo4j 5.26 Community. DSPy 3.4.1 is
  installed and exercised offline; live Gemini model access, optimizer
  compilation, and held-out evaluation have not been verified.
- A configured model name without a provider prefix is completed with `gemini/`.
  Another provider must be named explicitly, for example `openai/gpt-4o-mini`.
- The SQLite store is a snapshot of one dataset build; a store built from a
  different seed is not detected at query time, so rebuild it after regenerating
  the dataset.
- Orchestrator timeouts bound waiting, not execution: an abandoned source call
  may keep running in its worker thread until it returns on its own. The first
  concurrent FAISS query can also load the embedding model more than once
  because the Module 3 embedder loads it lazily without a lock.
- The orchestrator assembles evidence only; cross-engine synthesis and the
  APPROVE / FLAG verdict live in `supplier_risk.decision`, one model call deep.
