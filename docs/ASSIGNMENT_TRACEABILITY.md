# Assignment Traceability

**Project:** AI Supplier Risk Dashboard  
**Phase:** Phase 2 — Weeks 4–6 — Multi-Source Retrieval  
**Authoritative sources:** `Problem_Statement.pdf` and `StepByStep_Guide.pdf`

**Status legend**

- **Verified** — directly supported by the assignment PDFs or observed workspace state.
- **Provisional** — proposed engineering decision; not established by the assignment.
- **Blocked** — cannot be resolved with the currently available evidence.

## Source inventory

| Source | Location | Extraction | Notes |
|---|---|---|---|
| Problem Statement | `Problem_Statement.pdf`, 3 pages | Complete via `pdftotext -layout` | No images or scanned pages |
| Step-by-Step Guide | `StepByStep_Guide.pdf`, 3 pages | Complete via `pdftotext -layout` | No images or scanned pages |
| Reference project | External GitHub README | Read-only review | Supplementary; not authoritative |

Missing: the “Vibe Coding Helper Doc” referenced in Guide Step 10. Deployment is blocked until it is supplied or waived.

## Requirements traceability

| ID | Requirement | Source | Classification | Status | Planned verification |
|---|---|---|---|---|---|
| R-01 | Use Supplier Risk Dashboard: SQL delivery/defect metrics; FAISS contract/complaint text; Neo4j subsidiary/ownership graph; verdict APPROVE / FLAG. | Problem Statement p.2, domain item 7 | Functional / Data | Mandatory | Store smoke tests; UI labels; schema review |
| R-02 | Query SQL, semantic FAISS with hybrid BM25+dense, and Neo4j simultaneously for the same entity. | Problem Statement p.2, “The Assignment” | Architecture / Functional | Mandatory | Orchestrator tests; distinct contexts; latency comparison |
| R-03 | Use a DSPy module compiled with a real optimizer, not a hand-written prompt, to synthesize one structured answer from all three contexts. | Problem Statement p.2 | AI / LLM | Mandatory | Compiled state; before/after evidence |
| R-04 | Demonstrate hybrid search, BM25, RRF, text-to-SQL, Neo4j GraphRAG, DSPy declarative prompting, parallel retrieval, and LLM-as-judge optimization. | Problem Statement p.2, “Topics” | Architecture / AI | Mandatory; text-to-SQL ambiguous | BM25/RRF tests; SQL tests; graph traversal; judge tests |
| R-05 | Deliver a runnable, reproducible synthetic generator populating all three stores. | Problem Statement p.3; Guide p.2 Step 2 | Data / Deliverable | Mandatory | Clean-state generation; repeatable counts |
| R-06 | Use a manifest of approximately 20–30 fake entities. | Guide p.2 Step 2 | Data | Mandatory range | Manifest count assertion |
| R-07 | SQLite tables/rows; FAISS local chunking, embeddings, persisted index, metadata pickle/JSON; Neo4j via Docker with nodes and typed edges; print counts. | Guide p.2 Step 2 | Data / Operational | Mandatory | Per-store count checks; generator output |
| R-08 | Write query_sql, query_faiss, query_graph as separate, individually testable functions returning labeled context strings. | Problem Statement p.3; Guide p.2 Step 3 | Functional / Testing | Mandatory | Unit tests per engine; standalone smoke runs |
| R-09 | FAISS combines exact metadata match with semantic fallback, or real BM25+dense fusion. | Guide p.2 Step 3; Problem Statement p.2 | Functional | Mandatory choice | Hybrid retrieval and RRF tests |
| R-10 | Each engine returns a tagged error string, not a crash, when its store is empty or unreachable. | Guide p.2 Step 3 | Functional / Operational | Mandatory | Fault-injection tests |
| R-11 | Parallel retrieval via ThreadPoolExecutor or async; tolerate any single engine failing. | Problem Statement p.3; Guide p.2 Step 4 | Functional / Architecture | Mandatory | Orchestrator tests; Neo4j-down test |
| R-12 | Show total latency close to the slowest engine, not the sum; stop Neo4j and verify SQL + FAISS + tagged graph error. | Guide p.2 Step 4 | Evaluation | Mandatory | Timed runs; failure demonstration |
| R-13 | DSPy Signature with three context inputs and structured verdict outputs; wrap in ChainOfThought. | Guide p.2 Step 5 | AI / LLM | Mandatory | Signature test; uncompiled smoke |
| R-14 | Configure a stronger teacher model and cheaper student model. | Problem Statement p.3; Guide p.2 Step 5 | AI / LLM | Mandatory | Config test; compile metadata |
| R-15 | Uncompiled module returns syntactically valid structured output for at least one entity. | Guide p.2 Step 5 | Testing | Mandatory | Smoke test; judge parse |
| R-16 | Rule-based judge validates structure, required fields, valid ranges, and at least one hard content rule. | Problem Statement p.3; Guide p.3 Step 6 | Evaluation | Mandatory | Judge unit tests |
| R-17 | Demonstrate the judge rejects a deliberately bad output. | Guide p.3 Step 6; Problem Statement p.3 | Evaluation | Mandatory | Saved rejection evidence |
| R-18 | Build training set from parallel retrieval over the entity manifest and wrap as dspy.Example. | Guide p.3 Step 7 | AI / Data | Mandatory | Train-set builder test |
| R-19 | Compile with BootstrapFewShot or equivalent; save optimized state. | Problem Statement p.3; Guide p.3 Step 7 | AI / LLM | Mandatory | State file and load test |
| R-20 | Show saved optimized state and at least one student reasoning trace. | Guide p.3 Step 7; Problem Statement p.3 | Evaluation / Deliverable | Mandatory | Evidence documentation |
| R-21 | Run compiled module on a held-out entity not in the training manifest; compare uncompiled vs compiled and show measurable improvement. | Guide p.3 Step 8; Problem Statement p.3 | Evaluation | Mandatory | Held-out table; frozen artifact hash |
| R-22 | UI shows all three raw contexts side by side plus the synthesized verdict; reviewer can see each engine’s contribution. | Problem Statement p.3; Guide p.3 Step 9 | UI | Mandatory | AppTest; screenshots; reviewer checklist |
| R-23 | README documents architecture diagram, tech stack, and full setup for all three databases including Neo4j Docker commands. | Problem Statement p.3; Guide p.3 Step 10 | Documentation | Mandatory | README review; fresh setup |
| R-24 | Setup instructions are followable by someone other than the author, end to end. | Guide p.3 Step 10 | Documentation / Operational | Mandatory | Independent reviewer checklist |
| R-25 | Out of scope: LangGraph, tool loops, multi-agent supervisors, production observability, caching, fallback cascades. DSPy synthesis is one compiled module call. | Problem Statement p.3 | Constraint | Mandatory | Dependency and architecture audit |
| R-26 | One-page schema note covering all three stores before domain code. | Guide p.2 Step 1 | Documentation / Data | Mandatory | `docs/schema.md` review |
| R-27 | All three engines return genuinely different information for the same entity, not duplicated content. | Problem Statement p.3 | Evaluation | Mandatory | Diversity report; overlap metric |
| R-28 | Degrade gracefully when one engine is down, for example Neo4j not running. | Problem Statement p.3; Guide p.2 Step 4 | Operational / Evaluation | Mandatory | Stop Neo4j; verify contexts and statuses |
| R-29 | Deploy per Vibe Coding Helper Doc; note local or hosted Neo4j requirement. | Guide p.3 Step 10 | Operational / Documentation | Blocked | Missing helper document |
| R-30 | Do not start DSPy synthesis until all three engines pass their own smoke test. | Guide p.2, “How to Use This Guide” | Testing / Process | Mandatory | Gate before Module 6 |

## Provisional register

These decisions are engineering choices, not verified assignment requirements:

- 24 suppliers and a 14 train / 4 development / 6 held-out split. Implemented
  in Module 2, but still a provisional engineering choice; the assignment only
  requires approximately 20–30 entities and a held-out entity outside training.
- SQLite column names and synthetic metric distributions. The Module 5 store
  implements only `dataset_metadata`, `suppliers`, and `delivery_metrics`; the
  proposed contract, complaint, and ownership tables remain unbuilt because
  Modules 3 and 4 already own that evidence.
- Neo4j ownership schema (one optional parent link per supplier), the
  `Supplier`/`Company` labels, and the 3-hop / 25-relationship query limits.
  Implemented in Module 4 (`src/supplier_risk/graph/`).
- Real BM25 + FAISS dense retrieval + reciprocal rank fusion. Implemented in
  Module 3 (`src/supplier_risk/retrieval/`).
- Pydantic v2 contracts. Implemented in Module 2 (`src/supplier_risk/contracts.py`).
- Streamlit as the UI framework.
- 0–100 risk score, `evidence_status`, and citation schema remain unimplemented
  provisional choices.
- Verdict field names, rationale/findings length bounds, and the banned-term
  list are provisional choices implemented in Module 6.
- Evidence-status behavior for missing or failed retrieval.
- Timeout values and retry policy.
- `all-MiniLM-L6-v2` (384 dimensions, CPU) as the embedding model. Implemented
  in Module 3 and recorded in `artifacts/retrieval/manifest.json`.
- One document per contract/complaint record, `IndexFlatIP` cosine search,
  BM25 `k1=1.5`/`b=0.75`, tokenizer `module3.v1`, `RRF k = 60`, and the
  `artifacts/retrieval/` artifact layout. Implemented in Module 3.
- Local-first Docker Neo4j 5 Community with a named data volume and
  loopback-only published ports. Implemented in `compose.yaml`.

## Resolved conflicts

- **BM25 / RRF:** The problem statement requires hybrid BM25+dense; the guide
  permits real BM25+dense fusion. Implement real BM25 + dense + RRF.
- **Judge:** The deliverables require a rule-based judge. The “LLM-as-judge
  optimization” topic is interpreted as optimization against the deterministic
  judge; no LLM judge is required for the MVP.

## Unresolved ambiguity

- **Text-to-SQL:** The topic list names it, but the concrete deliverable is
  `query_sql(entity)`. Modules 1–5 use parameterized SQL only. A constrained
  NL-to-SQL layer would require clarification and a separate module.

## Module 1 status

Module 1 implements repository foundation, configuration, and tests only. It
does not implement any retrieval, graph, DSPy, evaluation, or UI behavior.

## Module 2 status

Module 2 implements the canonical raw supplier data contracts, a deterministic
synthetic dataset generator, data-integrity tests, and the schema note. The
schema note preserves its one-page store summary and adds proposed physical
schemas for SQLite, FAISS metadata, and Neo4j; none of those stores is populated
in Module 2.

Evidence:

- **R-26:** `docs/schema.md` documents the supplier entity, the structured
  delivery/defect records intended for SQLite, contract/complaint text intended
  for FAISS, and ownership links intended for Neo4j. It labels each area as
  assignment-mandated, implemented, proposed, or open.
- **R-06:** the fixed manifest contains 24 fictional suppliers, inside the
  required 20–30 range. Counts are asserted by tests.
- **Partition integrity:** `PARTITION_ID_RANGES` is the single source of truth
  for supplier-to-partition membership. Validation rejects a swapped assignment
  even when the 14/4/6 counts are unchanged, and the generator uses the same
  mapping.
- **R-21 (preparation):** the dataset provides 6 held-out suppliers with
  identities separate from training and development. Evaluation itself is not
  implemented in Module 2.
- **R-05 / R-07 remain open:** Module 2 generates and validates the source
  dataset only. It does not create SQLite tables, FAISS indexes, or Neo4j nodes
  and edges. Store population belongs to later modules.
- **R-01:** the domain and intended store mapping are recorded; the
  `APPROVE` / `FLAG` verdict contract is deferred to the synthesis module.

Module 2 does not add retrieval, graph persistence, DSPy, Gemini calls, scoring,
evaluation, or UI behavior. The provisional fields (0–100 score, evidence
status, citations) remain unimplemented.

## Module 3 status

Module 3 implements the document half of the semantic store: a deterministic
corpus of contract and complaint documents, a persisted FAISS dense index, a
persisted BM25 lexical index, hybrid ranking with RRF, metadata filtering, and
the assignment-facing `query_faiss(entity)` wrapper. SQLite population, Neo4j,
parallel orchestration, DSPy, the judge, and the UI remain unimplemented.

Evidence:

- **R-09 (implemented):** `reciprocal_rank_fusion()` in
  `src/supplier_risk/retrieval/rrf.py` merges real BM25 and dense rankings with
  `RRF(d) = Σ 1 / (k + rank_i(d))`, one-based ranks, `k = 60`, duplicate-summing,
  deterministic tie-breaking, and `top_k`. Hand-calculated formula tests are in
  `tests/test_retrieval_fusion.py`.
- **R-07 (partially implemented):** contracts and complaints are chunked as one
  document per record, embedded locally with `all-MiniLM-L6-v2`, and persisted
  as `dense.index.faiss` plus JSON metadata under `artifacts/retrieval/`. The build
  prints document, index, and fingerprint counts. Module 4 adds Neo4j nodes and
  typed ownership edges; SQLite tables remain unimplemented.
- **R-08 (partially implemented):** `query_faiss(entity)` in
  `src/supplier_risk/retrieval/pipeline.py` accepts a supplier id or exact legal
  name and returns a labeled `[FAISS]` context string covering only contract and
  complaint documents, with a tagged `[FAISS ERROR]` string for unknown
  suppliers and unavailable artifacts. Module 4 adds the corresponding
  `query_graph(entity)`; `query_sql` does not exist yet.
- **R-04 (partially implemented):** hybrid search, BM25, RRF, and the Neo4j
  ownership graph are implemented; text-to-SQL, DSPy, parallel retrieval, and
  LLM-as-judge are not.
- **R-02 / R-11 / R-12 (not implemented):** each engine is callable
  individually, but retrieval is synchronous. There is no ThreadPoolExecutor,
  no three-engine orchestration, and no latency comparison.
- **R-27 (not demonstrated):** the FAISS wrapper returns document text and the
  graph wrapper returns ownership edges. The cross-engine diversity check still
  requires the SQL context and a side-by-side comparison that do not exist yet.
- **R-30 (partial gate):** the FAISS/BM25 engine passes its own smoke tests
  (offline suite plus opt-in real-model tests), and Module 4 adds the graph
  engine's own offline and live smoke tests. DSPy synthesis must still wait for
  the SQL engine.
- **R-05 (partially satisfied):** the generator plus the Module 3 build populate
  the semantic store; Module 4 populates Neo4j. No store population exists for
  SQLite.
- **Artifact integrity:** loading validates the manifest, corpus fingerprint,
  counts, index dimension/type/vector count, vector-to-document mapping, the
  duplicated dense identity fields, embedding model, tokenizer version, BM25
  parameters, and persisted token lists against the canonical corpus text; an
  injected query embedder must also declare the persisted model (and dimension
  where it is cheap to report). Missing or corrupt artifacts raise
  `RetrievalArtifactError`. Tests are in `tests/test_retrieval_dense.py`,
  `tests/test_retrieval_lexical.py`, `tests/test_retrieval_embeddings.py`, and
  `tests/test_retrieval_pipeline.py`.
- **R-26 (extended):** `docs/schema.md` section 9 documents the implemented
  retrieval artifacts. Module 4 updates section 10 to the implemented graph
  schema; only the SQLite design remains proposed.

Module 3 adds no SQLite, Neo4j, DSPy, Gemini, scoring, evaluation, or UI code.
It reads no environment variables and makes no external LLM calls.

## Module 4 status

Module 4 implements the ownership graph store: Neo4j 5 persistence, a
deterministic and conflict-checked ingestion command, uniqueness constraints,
bounded traversal, and the assignment-facing `query_graph(entity)` wrapper.
SQLite, parallel orchestration, DSPy, the judge, and the UI remain
unimplemented.

Evidence:

- **R-07 (graph portion implemented):** `python -m supplier_risk.graph.ingest`
  loads the canonical dataset, validates every ownership record, writes 24
  `Supplier` nodes, 10 `Company` nodes, and 24 typed ownership edges, and prints
  node/edge counts. Repeated runs create zero nodes and zero relationships.
  Tests: `tests/test_graph_schema.py`, `tests/test_graph_store.py`, and the
  opt-in `tests/test_graph_integration.py`.
- **R-08 (graph portion implemented):** `query_graph(entity)` in
  `src/supplier_risk/graph/pipeline.py` accepts a supplier id or exact legal
  name and returns a labeled `[GRAPH]` context. Unknown suppliers, invalid
  configuration, an unavailable database, timeouts, and integrity failures
  return `[GRAPH ERROR]` strings. `query_sql` remains absent.
- **R-10 (graph portion implemented):** the graph wrapper converts every typed
  graph error into a tagged string instead of crashing; no-evidence queries are
  successful `[GRAPH]` contexts with zero relationships. Tests:
  `tests/test_graph_pipeline.py` and the unavailable/closed-store integration
  tests.
- **R-04 / R-01 (partially implemented):** Neo4j ownership retrieval is real,
  but text-to-SQL, GraphRAG synthesis, DSPy, parallel orchestration, judging,
  scoring, and the APPROVE/FLAG verdict are not implemented.
- **R-26 (extended):** `docs/schema.md` section 10 documents node labels, stable
  keys, relationship type/direction, the two uniqueness constraints, traversal
  semantics (3 hops, 25 relationships, path cap, deterministic ordering),
  ingestion conflict rules, and current limitations.
- **R-23 (partially satisfied):** README documents Neo4j's role, placeholder
  configuration variables, Compose startup/health/stop commands, ingestion, and
  both test modes. Deployment and third-party setup review remain outstanding.
- **R-28 (partially demonstrated):** `query_graph` returns a tagged error for an
  unreachable Neo4j instance, and this is tested. A parallel orchestrator that
  keeps SQL + FAISS running when Neo4j is down does not exist yet.
- **R-02 / R-11 / R-12 (not implemented):** the graph engine is individually
  callable but not fired in parallel with the SQL and FAISS engines.
- **R-30 (partial gate):** the graph engine passes its own offline and live
  smoke tests. DSPy must still wait for the SQL engine.

Module 4 adds no SQLite, DSPy, Gemini, scoring, evaluation, or UI code. It reads
only the three `NEO4J_*` settings, makes no external LLM calls, and never
returns a legal-control or risk verdict from ownership data alone.

## Module 5 status

Module 5 implements the structured SQL source and the parallel orchestration
layer: a SQLite delivery/defect store, the assignment-facing `query_sql(entity)`
wrapper, and `query_all_sources(entity)`, which runs the SQL, FAISS/BM25, and
Neo4j sources concurrently with per-source error isolation. DSPy synthesis, the
judge, scoring, and the UI remain unimplemented.

Evidence:

- **R-08 (implemented):** `query_sql` (`src/supplier_risk/sql/pipeline.py`),
  `query_faiss` (Module 3), and `query_graph` (Module 4) all accept a supplier id
  or exact legal name and return one labeled context string, with tagged
  `[SQL ERROR]` / `[FAISS ERROR]` / `[GRAPH ERROR]` strings for failures. Tests:
  `tests/test_sql_pipeline.py`, plus the existing Module 3 and 4 suites.
- **R-07 (implemented for the canonical stores):**
  `uv run --locked python -m supplier_risk.sql.build` (or `make sql-build`)
  loads the validated dataset, writes 24 `suppliers` rows and 24
  `delivery_metrics` rows in one transaction, verifies the row counts, and
  replaces `data/generated/suppliers.sqlite` atomically. The store deliberately
  does not duplicate contract/complaint text or ownership edges. Tests:
  `tests/test_sql_schema.py`.
- **R-02 / R-11 (implemented):** `query_all_sources(entity)` submits all three
  sources to a `ThreadPoolExecutor` before waiting, collects each outcome
  independently, and returns a provenance-preserving
  `OrchestrationResult`. A `threading.Barrier` test proves concurrent
  submission; single- and multi-source failure tests prove the surviving
  evidence is preserved. Tests: `tests/test_orchestration.py`.
- **R-10 (implemented):** every engine tags its own failures instead of
  crashing, and the orchestrator isolates unexpected exceptions per source and
  redacts credential-like values. Empty evidence is reported as `empty`, not as
  an error.
- **R-12 (implemented; no production latency claim):** the orchestrator reports
  per-source and overall elapsed time, and a test shows three 0.15 s sources
  complete well under the 0.45 s a sequential implementation would need. No
  benchmark at larger scale is claimed.
- **R-28 (implemented):** a failed graph source alongside successful SQL and
  FAISS sources returns a `partial` result carrying the tagged graph error plus
  the two evidence contexts.
- **R-27 (demonstrated):** the offline integration test asserts that the SQL,
  FAISS, and graph contexts for the same supplier contain different facts, and
  that no source claims another's evidence.
- **R-30 (gate satisfied for retrieval):** all three engines pass their own
  offline smoke tests, and `tests/test_orchestration_integration.py` runs the
  real SQL and FAISS/BM25 engines together. The live Neo4j variant is opt-in.
  DSPy synthesis may begin in Module 6.
- **R-04 (partially implemented):** hybrid search, BM25, RRF, Neo4j graph
  retrieval, and parallel retrieval are implemented. Text-to-SQL remains
  parameterized SQL only; DSPy prompting and the judge are not implemented.
- **R-26 (extended):** `docs/schema.md` section 8 documents the implemented
  SQLite subset and section 11 the SQL context, orchestration contract, partial
  failure behavior, concurrency, and timeouts. Sections 11–13 were renumbered to
  12–14.

Module 5 adds no DSPy, Gemini, scoring, evaluation, or UI code, reads no
environment variable, makes no external LLM call, and never produces a risk
score or APPROVE/FLAG verdict.

## Module 6 status

Module 6 implements the DSPy Signature, the one-call ChainOfThought decision
module, the typed verdict contract, the teacher/student LM configuration, and
the deterministic rule-based judge. It does not implement optimization, a
training set, teacher demonstrations, a compiled or saved student program,
held-out comparison, or the UI; those remain Module 7 and later.

Evidence:

- **R-13 (implemented):** `SupplierRiskSignature`
  (`src/supplier_risk/decision/signature.py`) has exactly three evidence inputs —
  `sql_context`, `faiss_context`, and `graph_context` — and the structured
  outputs `supplier_id`, `verdict`, `rationale`, `findings`, and
  `missing_evidence`. `VerdictGenerator` (`decision/module.py`) wraps it in
  `dspy.ChainOfThought` and makes exactly one prediction per request. Tests:
  `tests/test_decision_signature.py`, `tests/test_decision_module.py`.
- **R-14 (implemented for configuration):** `LanguageModelSettings` and
  `load_language_model_settings()` (`src/supplier_risk/config.py`) validate
  `GEMINI_TEACHER_MODEL` and `GEMINI_STUDENT_MODEL` separately;
  `build_role_language_models()` (`decision/lm.py`) returns one `dspy.LM` per
  role without global `dspy.configure`, so configuration cannot leak between
  tests or requests. Tests: `tests/test_decision_lm.py`.
- **R-15 (satisfied offline; live call unverified):** the decision module returns
  contract-valid `APPROVE` and `FLAG` output end to end with a fake predictor.
  No live Gemini call has been made, and no test requires one.
- **R-16 (implemented):** `judge_verdict` and `judge_payload`
  (`decision/judge.py`) validate structure through the verdict contract and
  enforce `decision_invalid`, `supplier_identity_mismatch`,
  `unacknowledged_failed_source`, `retrieval_failed`,
  `no_evidence_for_approval`, and the hard `banned_content` rule
  (`BANNED_TERMS`). Tests: `tests/test_decision_judge.py`.
- **R-17 (demonstrated):** a deliberately bad verdict is rejected and, when it
  was an `APPROVE`, downgraded to `FLAG`: a banned term, an invalid verdict
  value, a mismatched supplier, an unacknowledged failed source, an approval
  without evidence, and a fully failed retrieval all fail. The tests also assert
  that the judge never upgrades a verdict and never invokes an LM.
- **R-01 / R-03 (partially implemented):** the `APPROVE` / `FLAG` verdict and a
  single structured synthesis call exist; the optimizer-compiled program that
  R-03 ultimately requires is Module 7 work.
- **R-30 (gate satisfied):** all three engines pass their own smoke tests, and
  Module 6 consumes their Module 5 `OrchestrationResult` instead of querying any
  store itself.
- **R-26 (extended):** `docs/schema.md` section 15 documents the Signature
  fields, verdict contract, decision result, and judge rules; section 13.3 lists
  the decision-layer errors.

Module 6 adds no optimizer, training-example generation, compiled state, UI,
LLM-as-judge, retrieval-ranking change, graph-schema change, or persistent
storage. Its tests are offline and read no `.env` value.
