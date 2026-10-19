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
| R-26 | One-page schema note covering all three stores before domain code. | Guide p.2 Step 1 | Documentation / Data | Mandatory | docs/DATA_MODEL.md review |
| R-27 | All three engines return genuinely different information for the same entity, not duplicated content. | Problem Statement p.3 | Evaluation | Mandatory | Diversity report; overlap metric |
| R-28 | Degrade gracefully when one engine is down, for example Neo4j not running. | Problem Statement p.3; Guide p.2 Step 4 | Operational / Evaluation | Mandatory | Stop Neo4j; verify contexts and statuses |
| R-29 | Deploy per Vibe Coding Helper Doc; note local or hosted Neo4j requirement. | Guide p.3 Step 10 | Operational / Documentation | Blocked | Missing helper document |
| R-30 | Do not start DSPy synthesis until all three engines pass their own smoke test. | Guide p.2, “How to Use This Guide” | Testing / Process | Mandatory | Gate before Module 6 |

## Provisional register

These decisions are engineering choices, not verified assignment requirements:

- 24 suppliers, with a provisional 14 train / 4 development / 6 held-out split.
- SQLite column names and synthetic metric distributions.
- Neo4j ownership schema and 3-hop traversal limit.
- Real BM25 + FAISS dense retrieval + reciprocal rank fusion.
- Pydantic v2 contracts.
- Streamlit as the UI framework.
- 0–100 risk score, `evidence_status`, citation schema, summary length, and
  banned-word list.
- Evidence-status behavior for missing or failed retrieval.
- Timeout values and retry policy.
- `all-MiniLM-L6-v2` as the embedding model.
- Local-first Docker Neo4j.

## Resolved conflicts

- **BM25 / RRF:** The problem statement requires hybrid BM25+dense; the guide
  permits real BM25+dense fusion. Implement real BM25 + dense + RRF.
- **Judge:** The deliverables require a rule-based judge. The “LLM-as-judge
  optimization” topic is interpreted as optimization against the deterministic
  judge; no LLM judge is required for the MVP.

## Unresolved ambiguity

- **Text-to-SQL:** The topic list names it, but the concrete deliverable is
  `query_sql(entity)`. Module 1 and the MVP use parameterized SQL only. A
  constrained NL-to-SQL layer would require clarification and a separate module.

## Module 1 status

Module 1 implements repository foundation, configuration, and tests only. It
does not implement any retrieval, graph, DSPy, evaluation, or UI behavior.
