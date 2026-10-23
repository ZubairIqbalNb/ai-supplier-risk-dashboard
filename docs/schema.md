# Supplier Data Schema (Modules 2–4)

**Project:** AI Supplier Risk Dashboard  
**Phase:** Phase 2 — Multi-Source Retrieval  
**Versions:** `module2.v1` (source dataset), `module3.v1` (retrieval artifacts), Neo4j graph schema from Module 4

**Status:** Module 2 data foundation, Module 3 document retrieval, and Module 4 ownership graph implemented

This note defines the canonical raw supplier data used by later modules, the
document-retrieval artifacts built from it, and the Neo4j ownership graph.
Module 2 generates and validates the source records. Module 3 builds a
deterministic document corpus, a persisted FAISS dense index, and a persisted
BM25 lexical index, and serves hybrid retrieval with Reciprocal Rank Fusion.
Module 4 validates the ownership records and writes the supplier/company graph
to Neo4j, with bounded `query_graph(entity)` retrieval. SQLite is still **not**
populated, and no scoring, DSPy synthesis, or UI exists.

**Status legend**

- **Mandated** — required by the assignment PDFs.
- **Implemented** — present in Module 2–4 code and covered by tests.
- **Proposed** — future-store design recorded here; not implemented.
- **Open** — requires a design decision or approval before implementation.

## 1. Requirement status

### Confirmed by the assignment

- The domain is **Supplier Risk Dashboard** (Problem Statement, domain item 7).
- Structured facts are delivery/defect metrics for SQL.
- Free text is contract/complaint text for semantic search.
- Relationships are subsidiary/ownership links for the graph store.
- Hybrid retrieval must combine BM25 and dense (FAISS) ranking, or exact
  metadata match with semantic fallback (Problem Statement p.2; Guide p.2
  Step 3). Module 3 implements real BM25 + dense fusion with RRF.
- The semantic store chunks and embeds free text locally and persists the index
  and its metadata (Guide p.2 Step 2). Module 3 implements this for contracts
  and complaints.
- The final verdict vocabulary is `APPROVE` / `FLAG`.
- The generator must be runnable and reproducible (Problem Statement, deliverables).
- The entity manifest must contain approximately **20–30 fake entities**
  (Step-by-Step Guide, Step 2).
- A one-page schema note covering all three stores must exist before domain code
  (Step-by-Step Guide, Step 1).
- Evaluation uses a held-out entity not present in the training manifest
  (Step-by-Step Guide, Step 8).

### Provisional engineering decisions

These are project choices, not assignment-mandated values:

- Exactly **24** synthetic suppliers (inside the required 20–30 range).
- The **14 train / 4 development / 6 held-out** split. The assignment requires
  held-out separation but does not prescribe these counts or a development split.
- Fixed supplier-ID ranges for partition membership (enforced, see section 7).
- Field names, units, numeric ranges, sectors, contract types, complaint
  categories, and the ownership-percentage rule.
- One searchable document per contract or complaint record (66 documents for
  seed `20261019`), with the identity scheme in section 9.
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions), `IndexFlatIP` over
  L2-normalized vectors, BM25 `k1 = 1.5` / `b = 0.75`, RRF `k = 60`, tokenizer
  `module3.v1`, and the artifact paths in section 9.
- Supplier/company identifier formats.
- A single annual (`FY2025`) metrics snapshot per supplier.
- JSON as the generated dataset format and `20261019` as the default seed.
- The `APPROVE` / `FLAG` verdict, 0–100 score, evidence-status field, citation
  schema, and banned-word list are **not implemented in Module 2**; no scoring
  or verdict contract is defined at this stage.

## 2. One-page store schema

| Store | Canonical records | Key fields | Cardinality |
|---|---|---|---|
| SQLite | `SupplierProfile`, `DeliveryMetrics` | `supplier_id`, `reporting_period`, `on_time_delivery_rate`, `defect_rate`, `average_lead_time_days`, `orders_fulfilled` | One profile and one annual metrics row per supplier |
| FAISS + BM25 | `Contract`, `Complaint` | `contract_id` / `complaint_id`, `supplier_id`, dates, `contract_type` / `severity`, free text | One or more contracts and zero or more complaints per supplier |
| Neo4j | `OwnershipLink` | `parent_company_id`, `parent_company_name`, `child_supplier_id`, `ownership_percentage`, `relationship_type` | One optional parent link per supplier |

`SupplierRecord` and `SupplierDataset` are the source-of-truth containers used
to generate all three stores consistently. Sections 3–7 add field-level
validation detail; the table above is the single-page summary. Section 8
records the **proposed** SQLite schema, which remains unimplemented. Section 9
documents the **implemented** Module 3 document-retrieval artifacts, and
section 10 the **implemented** Module 4 Neo4j graph.

## 3. Canonical records

All models are Pydantic v2 models with these shared rules:

- `strict=True`: most cross-type coercion is disabled. A string is not accepted
  for an integer or float field, and a float is not accepted for an integer
  field. Pydantic still applies lossless **int → float widening**: `0`, `1`, or
  `100` are accepted and stored as `0.0`, `1.0`, or `100.0` for float fields.
  This is intentional and documented rather than treated as strict rejection.
- `extra="forbid"`: unknown fields, including premature derived fields, are
  rejected.
- `frozen=True`: records are immutable after validation.
- `allow_inf_nan=False`: `NaN` and infinity are rejected for every float field.
- Payload dates are ISO `YYYY-MM-DD`; numeric units are documented below.

### 3.1 `SupplierProfile`

Stable supplier identity and categorical facts.

| Field | Type | Required | Rules |
|---|---|---|---|
| `supplier_id` | string | yes | `^SUP-\d{3}$`; must be a manifest id `SUP-001`–`SUP-024` |
| `legal_name` | string | yes | 2–120 characters; no leading/trailing whitespace |
| `country_code` | string | yes | Two uppercase letters `^[A-Z]{2}$`; generated values use ISO 3166-1 alpha-2 |
| `sector` | enum | yes | `electronics`, `industrial_components`, `logistics`, `packaging`, `raw_materials` |
| `founded_year` | integer | yes | 1800–2025 |
| `employee_count` | integer | yes | 1–1,000,000 |

### 3.2 `DeliveryMetrics`

One annual structured observation per supplier.

| Field | Type | Required | Rules / unit |
|---|---|---|---|
| `supplier_id` | string | yes | `^SUP-\d{3}$`; must match the enclosing record |
| `reporting_period` | string | yes | `^FY\d{4}$`; generated data uses `FY2025` |
| `on_time_delivery_rate` | float | yes | Fraction in `[0.0, 1.0]`; `0.95` means 95% |
| `defect_rate` | float | yes | Fraction in `[0.0, 1.0]`; `0.02` means 2% of units |
| `average_lead_time_days` | float | yes | Greater than `0`, at most `365` days |
| `orders_fulfilled` | integer | yes | `0`–`10,000,000` |

No score, risk band, or derived flag is stored here.

### 3.3 `Contract`

Synthetic contract metadata and text.

| Field | Type | Required | Rules / unit |
|---|---|---|---|
| `contract_id` | string | yes | `^CON-\d{3}-\d{2}$`; unique in the dataset |
| `supplier_id` | string | yes | `^SUP-\d{3}$`; must match the enclosing record |
| `contract_type` | enum | yes | `MASTER_SUPPLY`, `QUALITY_ASSURANCE`, `LOGISTICS_SERVICES` |
| `effective_date` | date | yes | ISO date |
| `expiry_date` | date | yes | Must be later than `effective_date` |
| `annual_value_usd` | integer | yes | Whole USD, `0`–`1,000,000,000,000` |
| `contract_text` | string | yes | 40–4,000 characters |

Each supplier has at least one contract. Text is fictional and is not legal
advice or a real agreement.

### 3.4 `Complaint`

Synthetic complaint metadata and text.

| Field | Type | Required | Rules |
|---|---|---|---|
| `complaint_id` | string | yes | `^CMP-\d{3}-\d{2}$`; unique in the dataset |
| `supplier_id` | string | yes | `^SUP-\d{3}$`; must match the enclosing record |
| `received_date` | date | yes | ISO date |
| `severity` | enum | yes | `LOW`, `MEDIUM`, `HIGH` |
| `category` | enum | yes | `DELIVERY`, `QUALITY`, `DOCUMENTATION`, `COMPLIANCE` |
| `complaint_text` | string | yes | 20–4,000 characters |

The complaint list is genuinely optional. **An empty list means no complaint
record exists in this synthetic dataset; it is not evidence of low risk.**

### 3.5 `OwnershipLink`

An ownership link between a parent company and a supplier. The contract records
both sides by id; section 10 documents the implemented Neo4j edge direction.

| Field | Type | Required | Rules |
|---|---|---|---|
| `parent_company_id` | string | yes | `^ORG-\d{3}$`; must not collide with a supplier id |
| `parent_company_name` | string | yes | 2–120 characters; consistent for a given id |
| `child_supplier_id` | string | yes | `^SUP-\d{3}$`; must match the enclosing record |
| `ownership_percentage` | float | yes | `(0.0, 100.0]` |
| `relationship_type` | enum | yes | `SUBSIDIARY_OF` or `JOINT_VENTURE_WITH` |

Relationship rule:

- `SUBSIDIARY_OF` requires ownership strictly greater than 50%.
- `JOINT_VENTURE_WITH` requires ownership at most 50%.

The ownership link is optional because the schema does not claim that every
supplier has a known parent.

**Current limitation:** a supplier has **at most one** optional parent link.
The model does not represent multiple parents, parent-of-parent chains, or a
company hierarchy. The assignment names a “subsidiary/ownership graph” but does
not specify graph depth or ownership cardinality. Module 4 implements this
one-parent model faithfully; see section 13 for the expansion decision.

### 3.6 `SupplierRecord`

Bundles one supplier's raw records:

- `profile`
- `partition` (`train`, `dev`, `held_out`)
- `delivery_metrics`
- `contracts` (1–10)
- `complaints` (0–20)
- `ownership` (optional)

Cross-field rules:

- Every nested `supplier_id` or `child_supplier_id` equals
  `profile.supplier_id`.
- Contract and complaint identifiers are unique within the record.

### 3.7 `SupplierDataset`

Top-level JSON document:

- `schema_version`: `module2.v1`, produced from the single `SCHEMA_VERSION`
  constant; the literal type alias `SchemaVersion` is tested against it.
- `seed`: non-negative integer up to `2^32 - 1`
- `suppliers`: the 24 records

Dataset-level rules:

- Exactly 14 `train`, 4 `dev`, and 6 `held_out` records.
- Each supplier id belongs to the fixed manifest (`SUP-001`–`SUP-024`).
- Each supplier's `partition` equals the partition fixed for its id by
  `PARTITION_ID_RANGES` (section 7).
- Supplier identifiers are globally unique.
- Legal names are globally unique, compared case-insensitively.
- Contract and complaint identifiers are globally unique.
- A parent-company id always maps to the same parent-company name.
- No parent-company id may also be a supplier id.

These rules are enforced by `SupplierDataset`, not only by tests.

## 4. Identifier strategy

| Entity | Format | Example | Scope |
|---|---|---|---|
| Supplier | `SUP-NNN` | `SUP-007` | Global, stable across runs |
| Contract | `CON-NNN-SS` | `CON-007-01` | Global unique; `NNN` follows the supplier number for readability |
| Complaint | `CMP-NNN-SS` | `CMP-007-02` | Global unique; `NNN` follows the supplier number for readability |
| Parent company | `ORG-NNN` | `ORG-003` | Shared across suppliers where ownership is shared |

The numerical prefix inside contract and complaint ids is informative only;
uniqueness is enforced globally by id. Supplier ids are assigned by the fixed
manifest and never by random draw, so repeated generation is stable.

## 5. Generated dataset format

Default output: `data/generated/suppliers.json` (Git-ignored).

```json
{
  "schema_version": "module2.v1",
  "seed": 20261019,
  "suppliers": [
    {
      "profile": {
        "supplier_id": "SUP-001",
        "legal_name": "Norvend Components Ltd",
        "country_code": "GB",
        "sector": "electronics",
        "founded_year": 1954,
        "employee_count": 9728
      },
      "partition": "train",
      "delivery_metrics": {
        "supplier_id": "SUP-001",
        "reporting_period": "FY2025",
        "on_time_delivery_rate": 0.809,
        "defect_rate": 0.0291,
        "average_lead_time_days": 13.4,
        "orders_fulfilled": 3970
      },
      "contracts": [{"contract_id": "CON-001-01", "...": "..."}],
      "complaints": [{"complaint_id": "CMP-001-01", "...": "..."}],
      "ownership": {"parent_company_id": "ORG-001", "...": "..."}
    }
  ]
}
```

The example abbreviates nested contract, complaint, and ownership objects; run
the generator for the complete document.

`write_dataset()` writes directly to the destination path. It does **not** use a
temporary file or atomic replace, so an interrupted write can leave a partial or
truncated file. The generator does not read the file back or re-validate it
after writing; validation applies to the in-memory dataset before writing. Use
`load_dataset()` to validate an existing file.

## 6. Generation procedure

```bash
uv run --locked python -m supplier_risk.data.generate
```

Optional arguments:

```bash
uv run --locked python -m supplier_risk.data.generate \
  --seed 20261019 \
  --output data/generated/suppliers.json
```

Determinism rules:

- The 24-supplier manifest is fixed and ordered in code.
- Supplier ids and partition membership come from the manifest and the fixed
  `PARTITION_ID_RANGES` mapping, not from random draws.
- All varying values come from `random.Random(seed)`.
- No timestamps, environment values, network calls, or external APIs are used.
- JSON output uses stable field order, two-space indentation, and a trailing
  newline.
- Two runs with the same seed and output path produce byte-identical files,
  including across processes with different hash seeds.
- A different seed produces different synthetic values while preserving ids,
  partitions, and structural rules.

The generator validates every record by constructing the canonical Pydantic
models, then validates the complete dataset before writing it.

## 7. Partitions and leakage prevention

| Partition | Count | Manifest ids |
|---|---:|---|
| `train` | 14 | `SUP-001` … `SUP-014` |
| `dev` | 4 | `SUP-015` … `SUP-018` |
| `held_out` | 6 | `SUP-019` … `SUP-024` |

Membership is not implied by ordering alone. `PARTITION_ID_RANGES` in
`src/supplier_risk/contracts.py` is the single source of truth:

```python
PARTITION_ID_RANGES = {
    "train": range(1, 15),
    "dev": range(15, 19),
    "held_out": range(19, 25),
}
```

`PARTITION_SIZES` is derived from those ranges, the generator calls
`partition_for_supplier_id()`, and `SupplierDataset` rejects any record whose
`partition` disagrees with the mapping, even when the 14/4/6 totals are
preserved. Supplier ids outside `SUP-001`–`SUP-024` are also rejected.

Leakage controls:

- Supplier identity is the unit of separation: each supplier appears in exactly
  one record and one partition.
- Supplier ids and legal names are globally unique.
- Contracts and complaints are nested inside their supplier record and cannot
  be repartitioned separately from that supplier.
- The held-out partition must not be used for prompt design, tuning, prompt
  iteration, or model selection. Module 2 contains no tuning or evaluation
  logic; this is a constraint on later modules.
- Shared parent companies across partitions are allowed because the ownership
  graph is a legitimate relationship, not a second copy of a supplier record.

## 8. Proposed SQLite schema

**Status:** Proposed. No database is created in Module 2. This design maps the
implemented Pydantic models; it does not add derived risk fields.

Loading assumptions: run inside one transaction, enable
`PRAGMA foreign_keys = ON` for every connection, insert suppliers before child
rows, and abort the transaction on any constraint failure. Identifier formats
are validated by the Pydantic contracts before insert; the “format” notes below
record the expected value. Add `CHECK`/`GLOB` constraints for those formats only
as defense in depth.

### 8.1 `dataset_metadata`

One row describing the generated dataset.

| Column | SQLite type | Constraints |
|---|---|---|
| `singleton_id` | INTEGER | `PRIMARY KEY CHECK (singleton_id = 1)` |
| `schema_version` | TEXT | `NOT NULL`; must equal `module2.v1` |
| `seed` | INTEGER | `NOT NULL CHECK (seed BETWEEN 0 AND 4294967295)` |
| `supplier_count` | INTEGER | `NOT NULL CHECK (supplier_count = 24)` |

This table makes the dataset version and seed queryable. It is a project
convenience, not an assignment requirement.

### 8.2 `suppliers`

| Column | SQLite type | Constraints |
|---|---|---|
| `supplier_id` | TEXT | `PRIMARY KEY`; format `SUP-NNN` |
| `legal_name` | TEXT | `NOT NULL`; unique case-insensitively |
| `country_code` | TEXT | `NOT NULL CHECK (length(country_code) = 2)` |
| `sector` | TEXT | `NOT NULL CHECK` in the five documented sectors |
| `founded_year` | INTEGER | `NOT NULL CHECK (founded_year BETWEEN 1800 AND 2025)` |
| `employee_count` | INTEGER | `NOT NULL CHECK (employee_count BETWEEN 1 AND 1000000)` |
| `partition` | TEXT | `NOT NULL CHECK (partition IN ('train', 'dev', 'held_out'))` |

Supplier identity and partition membership live on this row. Partition is copied
from the validated `SupplierRecord`; it is not recomputed during loading.

Uniqueness: `CREATE UNIQUE INDEX ux_suppliers_legal_name ON suppliers
(legal_name COLLATE NOCASE);`. SQLite `NOCASE` folds ASCII only; generated names
are ASCII, so this matches the Python `casefold()` check for current data.

### 8.3 `delivery_metrics`

| Column | SQLite type | Constraints |
|---|---|---|
| `supplier_id` | TEXT | `NOT NULL REFERENCES suppliers(supplier_id) ON DELETE RESTRICT` |
| `reporting_period` | TEXT | `NOT NULL`; format `FYNNNN` |
| `on_time_delivery_rate` | REAL | `NOT NULL CHECK (BETWEEN 0.0 AND 1.0)` |
| `defect_rate` | REAL | `NOT NULL CHECK (BETWEEN 0.0 AND 1.0)` |
| `average_lead_time_days` | REAL | `NOT NULL CHECK (> 0.0 AND <= 365.0)` |
| `orders_fulfilled` | INTEGER | `NOT NULL CHECK (BETWEEN 0 AND 10000000)` |

Primary key: `(supplier_id, reporting_period)`. The current contract nests exactly
one annual snapshot per supplier; the composite key leaves room for future
periods without changing identity. Rates are stored as fractions (`0.95`, not
`95`).

### 8.4 `contracts`

| Column | SQLite type | Constraints |
|---|---|---|
| `contract_id` | TEXT | `PRIMARY KEY`; format `CON-NNN-NN` |
| `supplier_id` | TEXT | `NOT NULL REFERENCES suppliers(supplier_id) ON DELETE RESTRICT` |
| `contract_type` | TEXT | `NOT NULL CHECK` in the three documented types |
| `effective_date` | TEXT | `NOT NULL`; ISO `YYYY-MM-DD` |
| `expiry_date` | TEXT | `NOT NULL CHECK (expiry_date > effective_date)` |
| `annual_value_usd` | INTEGER | `NOT NULL CHECK (BETWEEN 0 AND 1000000000000)` |
| `contract_text` | TEXT | `NOT NULL CHECK (length(contract_text) BETWEEN 40 AND 4000)` |

SQLite has no native date type; ISO text sorts correctly, so the expiry check is
valid lexicographically. Index `ix_contracts_supplier_id` on `supplier_id`.

### 8.5 `complaints`

| Column | SQLite type | Constraints |
|---|---|---|
| `complaint_id` | TEXT | `PRIMARY KEY`; format `CMP-NNN-NN` |
| `supplier_id` | TEXT | `NOT NULL REFERENCES suppliers(supplier_id) ON DELETE RESTRICT` |
| `received_date` | TEXT | `NOT NULL`; ISO `YYYY-MM-DD` |
| `severity` | TEXT | `NOT NULL CHECK IN ('LOW', 'MEDIUM', 'HIGH')` |
| `category` | TEXT | `NOT NULL CHECK` in the four documented categories |
| `complaint_text` | TEXT | `NOT NULL CHECK (length(complaint_text) BETWEEN 20 AND 4000)` |

Index `ix_complaints_supplier_id` on `supplier_id`. An absent complaint is an
absent row, never a row with placeholder text.

### 8.6 `ownership_links`

| Column | SQLite type | Constraints |
|---|---|---|
| `child_supplier_id` | TEXT | `PRIMARY KEY REFERENCES suppliers(supplier_id) ON DELETE RESTRICT` |
| `parent_company_id` | TEXT | `NOT NULL`; format `ORG-NNN` |
| `parent_company_name` | TEXT | `NOT NULL` |
| `ownership_percentage` | REAL | `NOT NULL CHECK (> 0.0 AND <= 100.0)` |
| `relationship_type` | TEXT | `NOT NULL CHECK IN ('SUBSIDIARY_OF', 'JOINT_VENTURE_WITH')` |

Table-level check: `SUBSIDIARY_OF` requires `ownership_percentage > 50` and
`JOINT_VENTURE_WITH` requires `ownership_percentage <= 50`. The primary key
expresses the current one-optional-parent rule.

Parent companies are not normalized into their own table in this proposal.
Repeated `(ORG-NNN, name)` rows are acceptable because the dataset validator
guarantees id-to-name consistency; normalization is an open decision in section
13. A separate table would also let SQLite express parent/child company
hierarchies that the current contract does not support.

## 9. Implemented document retrieval (Module 3)

**Status:** Implemented. Module 3 builds one searchable document per contract
and complaint record, persists a FAISS dense index and a BM25 lexical index, and
serves hybrid retrieval with RRF. No SQLite or Neo4j code is involved.

### 9.1 Source records and corpus size

The retrieval corpus uses exactly the free-text records already validated in
`SupplierRecord`:

- 32 `Contract` records (one or more per supplier).
- 34 `Complaint` records (zero or more per supplier).
- 66 documents in total for seed `20261019`.

Those counts are properties of the seed, not fixed contract rules. Contract
text runs 393–422 characters and complaint text 112–150 characters.

### 9.2 Document granularity

One document per source record; no chunking. The texts are far below the
embedding model's input limit, one agreement or complaint is a useful retrieval
unit, and a single document id per record keeps the vector-to-source mapping
unambiguous. No splitter or overlap policy is implemented. If a future dataset
contains records that exceed the model input window, chunking must be added
deliberately with deterministic chunk ids and a new artifact version rather than
retrofitted silently.

### 9.3 Document identity and searchable text

- `document_id` is `"<source_type>:<source_id>"`, for example
  `contract:CON-001-01` or `complaint:CMP-001-01`. It is derived from the source
  record, not from a process-random hash, so repeated builds from unchanged
  input produce identical ids. The `contract:` / `complaint:` prefix prevents
  collisions between the two source types.
- Corpus order is deterministic: `supplier_id` ascending, then all contracts,
  then all complaints, then source-record id. This order is shared by the
  corpus, FAISS vector ids, and BM25 document positions.
- The text is a short header followed by the exact source text:
  - Contract: `Contract <id> for supplier <name> (<supplier_id>), contract type
    <TYPE>, annual value USD <value>. <contract_text>`
  - Complaint: `Complaint <id> for supplier <name> (<supplier_id>), category
    <CATEGORY>, severity <SEVERITY>, received <date>. <complaint_text>`
- The header makes supplier names, supplier ids, source ids, contract types,
  severities, and dates searchable as exact terms without stringifying entire
  Pydantic objects or adding serialization noise.

### 9.4 Document metadata fields (implemented)

Persisted in `corpus.json` and validated on reload. `dense.metadata.json`
repeats the filterable identity fields for every vector position.

| Field | Type | Purpose |
|---|---|---|
| `document_id` | string | Stable id, `"<source_type>:<source_id>"` |
| `source_type` | `"contract"` or `"complaint"` | Source record type |
| `source_id` | string | `contract_id` or `complaint_id` |
| `supplier_id` | string | Canonical supplier identity; never omitted |
| `supplier_name` | string | Legal name copied from the validated profile |
| `partition` | `train` / `dev` / `held_out` | Copied from the supplier record |
| `text` | string | Exact searchable and embedded text |
| `contract_type` | string or null | Contract-only categorical metadata |
| `effective_date` | ISO date or null | Contract-only metadata |
| `expiry_date` | ISO date or null | Contract-only metadata |
| `annual_value_usd` | integer or null | Contract-only metadata |
| `received_date` | ISO date or null | Complaint-only metadata |
| `severity` | string or null | Complaint-only categorical metadata |
| `category` | string or null | Complaint-only categorical metadata |

The document model enforces that contract-only fields are present only on
contract documents, that complaint-only fields are present only on complaint
documents, that `document_id` matches the source, that `source_id` matches the
source-type identifier format, and that `partition` equals
`partition_for_supplier_id(supplier_id)`.

### 9.5 Dense retrieval

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, CPU
  only. It is loaded lazily on first use and reused by the retriever; importing
  the package never loads or downloads a model. No hosted embedding API is
  called.
- **Validation:** embedding matrices must match document count and model
  dimension, contain only finite values, and contain no zero-length vector.
  Violations raise `EmbeddingModelError`.
- **Similarity:** vectors are L2-normalized and added to `faiss.IndexFlatIP`, so
  inner product equals cosine similarity. Queries are normalized the same way.
- **Mapping:** FAISS ids are index positions `0..n-1`. `dense.metadata.json`
  stores one entry per vector with `vector_id`, `document_id`, `supplier_id`,
  `partition`, and `source_type`.
- **Lifecycle:** building is an explicit operation. Loading never rebuilds or
  replaces an index, and a missing, corrupt, wrong-type, wrong-dimension, or
  misaligned index raises `RetrievalArtifactError`.

### 9.6 Lexical retrieval

- **Implementation:** `rank_bm25.BM25Okapi` with `k1 = 1.5` and `b = 0.75`.
- **Tokenizer (`module3.v1`):** lowercase, then keep maximal runs of ASCII
  letters and digits, treating `-`, `_`, and `.` as joiners inside a token.
  Identifiers and numbers survive: `con-001-01`, `sup-007`, `lot-8399`,
  `fy2025`, `2023-01-01`. A comma is not a joiner, so `12,500` becomes `12`
  and `500`.
- **Persistence:** `lexical.json` stores the document ids, their token lists,
  the tokenizer version, and the BM25 parameters. BM25 is rebuilt from those
  tokens on load, so no pickle is used and scores are reproducible across
  process restarts.
- **Matching:** only documents that contain at least one query token are
  returned, so a query with no matching term returns no lexical results instead
  of the whole corpus. BM25 scores are not comparable to cosine scores and are
  never mixed directly.

### 9.7 Hybrid ranking

`RRF(d) = Σ 1 / (k + rank_i(d))`, with one-based ranks, the standard default
`k = 60`, and a configurable positive integer constant. Results are
deduplicated by `document_id`; a document present in both lists sums both
contributions. A document absent from a list receives no score from it. Ties
are broken by ascending `document_id`, and `top_k` is applied last. No score
thresholds are applied. Retrieval results keep the original dense rank,
lexical rank, and fused score.

### 9.8 Metadata filtering

The query interface filters on exact `supplier_id` and `partition` values from
the persisted corpus metadata. Filters are applied to both ranked lists before
fusion, so a document outside the requested supplier or partition can never
appear in the result.

- When a filter is active, both engines search the whole corpus. This is the
  documented oversampling strategy: a 66-document corpus makes exhaustive search
  cheap, and it prevents a supplier's documents from being silently missed
  because they fell outside a small unfiltered candidate window.
- When no filter is active, each engine returns
  `min(max(top_k × 4, top_k), document_count)` candidates. `4` is a documented
  default, not a tuned value.
- An invalid filter value raises `RetrievalQueryError`. A valid filter that
  matches no document returns an empty result.

### 9.9 Artifacts, persistence, and reload validation

Artifact directory: `artifacts/retrieval/` (Git-ignored).

| Artifact | Path | Contents |
|---|---|---|
| Corpus | `artifacts/retrieval/corpus.json` | Source dataset version and seed, document count, SHA-256 corpus fingerprint, every document |
| FAISS index | `artifacts/retrieval/dense.index.faiss` | `IndexFlatIP`, one normalized vector per document |
| Dense metadata | `artifacts/retrieval/dense.metadata.json` | Embedding model, dimension, metric, fingerprint, vector id → document map |
| Lexical index | `artifacts/retrieval/lexical.json` | Tokenizer version, BM25 parameters, document ids, token lists |
| Manifest | `artifacts/retrieval/manifest.json` | Written last; artifact version, corpus schema version and fingerprint, document/contract/complaint counts, embedding model and dimension, index type, metric, tokenizer version, BM25 parameters |

`artifacts/` and `*.faiss` are already Git-ignored. All metadata is JSON; no
pickle is read or written.

Reload validation, in order:

1. The manifest exists and validates.
2. The corpus validates, its fingerprint matches its documents, and it matches
   the manifest fingerprint, schema version, document count, and source counts.
3. `index.d` equals the metadata and manifest dimension, `index.ntotal` equals
   the document count, the index type is `IndexFlatIP`, vector ids equal their
   positions, and document ids are unique.
4. The dense metadata matches the manifest fingerprint, embedding model,
   dimension, and document count, and its `document_id`, `supplier_id`,
   `partition`, and `source_type` fields match the corpus document at the same
   position.
5. The lexical tokenizer version, BM25 parameters, fingerprint, and document
   order match the manifest and corpus, and every persisted token list equals
   the tokens the current tokenizer produces from that document's corpus text.
6. An injected query embedder declares the persisted embedding model; when it
   also exposes a cheap `declared_dimension`, that must equal the persisted
   dimension. Lazy embedders keep the query-time vector dimension check.

The default retriever is cached per artifact path, so a process that rebuilds
artifacts must call `reset_default_retriever()` or restart before querying.

Writes go through a temporary sibling file and `os.replace`, so a single file is
never partially written. Replacing the whole artifact set is **not** one
transaction: a crash between files can leave a mixed set. Because the manifest
is written last and records the corpus fingerprint, model, dimension, and
counts, such a set fails validation instead of being served.

### 9.10 Building and querying

```bash
uv run --locked python -m supplier_risk.retrieval.build
uv run --locked python -m supplier_risk.retrieval.build --artifacts-dir /tmp/retrieval
```

The build re-validates `data/generated/suppliers.json`, builds the corpus, both
indexes, and the manifest, and prints document, index, and fingerprint counts.
A missing or invalid dataset raises `CorpusError` with the generator command to
run; it is never replaced with an empty corpus.

Query interface and the assignment-facing wrapper:

```python
from supplier_risk.retrieval import HybridRetriever, query_faiss

retriever = HybridRetriever.load("artifacts/retrieval")
hits = retriever.query(
    "quality inspection records", top_k=5, supplier_id="SUP-001", partition="train"
)
context = query_faiss("SUP-001", retriever=retriever)
```

`query_faiss(entity)` accepts a supplier id or an exact legal name
(case-insensitive), applies an exact `supplier_id` filter, and returns one
labeled `[FAISS]` context string with the supplier identity, partition,
document count, and per-document source identity, ranks, and text. Unknown
suppliers and unavailable or corrupt artifacts are returned as tagged
`[FAISS ERROR]` strings. The string covers contract and complaint documents
only and never implies that SQLite or Neo4j was queried. The generic
`HybridRetriever.query` API raises typed project errors instead.

Error types: `CorpusError` (build input), `EmbeddingModelError` (local model
load or invalid vectors), `RetrievalArtifactError` (missing, stale, or corrupt
artifacts), `RetrievalQueryError` (query, `top_k`, or filter values), and
`UnknownSupplierError` (supplier wrapper resolution).

### 9.11 Deliberate non-claims

Module 3 does **not** populate SQLite, create Neo4j nodes or edges, run the three
engines in parallel, evaluate retrieval quality, or implement DSPy, the judge,
scoring, or the UI. `query_faiss` is one document-retrieval function, not a
multi-engine query.

## 10. Implemented Neo4j graph schema (Module 4)

**Status:** Implemented in `src/supplier_risk/graph/`. Module 4 starts no
container by itself; it connects to the Neo4j 5 service defined in
`compose.yaml`. Ingestion and retrieval are covered by the tests listed in
section 10.7.

### 10.1 Nodes

| Label | Identifying key | Properties | Uniqueness constraint |
|---|---|---|---|
| `Supplier` | `supplier_id` | `supplier_id`, `legal_name`, `country_code`, `sector`, `partition` | `supplier_id_unique`: `REQUIRE s.supplier_id IS UNIQUE` |
| `Company` | `parent_company_id` | `parent_company_id`, `parent_company_name` | `company_id_unique`: `REQUIRE c.parent_company_id IS UNIQUE` |

A supplier maps one-to-one to a `Supplier` node whose key is the same
`supplier_id` used in the SQLite design and FAISS metadata. Parent companies
are `Company` nodes; their `ORG-NNN` identifiers are validated never to overlap
the `SUP-NNN` namespace. Only validated profile fields are copied; contract,
complaint, and metric data stay in their own stores.

### 10.2 Relationships

The contract stores the parent company id/name, the child supplier id, the
percentage, and a `relationship_type`. Module 4 uses that type as the
relationship type and attaches the percentage, with the edge stored child
supplier → parent company:

```cypher
(:Supplier {supplier_id: "SUP-011"})
  -[:SUBSIDIARY_OF {ownership_percentage: 70.0}]->
(:Company {parent_company_id: "ORG-002"})

(:Supplier {supplier_id: "SUP-015"})
  -[:JOINT_VENTURE_WITH {ownership_percentage: 40.0}]->
(:Company {parent_company_id: "ORG-004"})
```

Meaning: the supplier is a subsidiary of, or in a joint venture with, the
company, and `ownership_percentage` is the company's ownership share of the
supplier (0–100, not a 0–1 fraction). Direction follows the contract fields
`child_supplier_id` and `parent_company_id`; the source contract nests the edge
under the child record, so ingestion reverses the storage nesting, not the
meaning.

`SUBSIDIARY_OF` requires `ownership_percentage > 50`; `JOINT_VENTURE_WITH`
requires `ownership_percentage <= 50`. These rules are validated again in the
graph plan even though the Module 2 contract already enforces them.

### 10.3 Traversal semantics and limits

`query_graph(entity)` resolves a supplier id (case-insensitive) or an exact
legal name (case-insensitive), then discovers ownership relationships within a
bounded neighborhood:

- **Maximum 3 relationship hops.** The Cypher pattern is the bounded
  `[:SUBSIDIARY_OF|JOINT_VENTURE_WITH*1..3]`; unbounded `*` traversal is never
  used.
- **At most 200 paths** are read from Neo4j per query. This limits worst-case
  path expansion before Python deduplication.
- **At most 25 ownership relationships** are returned. Relationships are
  deduplicated by `(child_key, parent_key, type)`; duplicate paths keep the
  smallest hop count. Ordering is by hop, child key, parent key, then type, so
  repeated queries produce identical output.
- **Discovery is undirected, direction is preserved.** The traversal is
  undirected so a supplier can reach its parent company (1 hop) and fellow
  subsidiaries of that company (2 hops). Every returned line still renders the
  stored direction, `child supplier -> parent company`; the query does not
  invent reverse edges.
- **A hop is one ownership relationship.** A supplier → company edge is 1 hop;
  supplier → company ← sibling supplier is 2 hops.
- If no relationship is returned, the context reports
  `ownership_relationships=0` and explicitly states that no ownership evidence
  was found. It does not claim a path exists.

With the canonical dataset (24 suppliers, 24 ownership links, 10 parent
companies) every supplier has exactly one edge, so its own evidence is depth 1
and sibling suppliers appear at depth 2. The 3-hop bound is a safety limit for
future hierarchy data; no company-to-company edge is present in the canonical
dataset.

### 10.4 Idempotent ingestion and duplicate prevention

`python -m supplier_risk.graph.ingest` (or `make graph-ingest`) performs one
validated run:

- Loads and re-validates `data/generated/suppliers.json`.
- Builds a deterministic plan of suppliers, companies, and ownership edges.
- Rejects unknown endpoints, self-links, duplicate logical edges, more than one
  ownership edge per supplier, invalid percentages, unsupported relationship
  types, and parent ids that collide with supplier ids.
- Verifies that existing edges for canonical suppliers do not point to a
  different parent or use a different relationship type; such a conflict aborts
  before writes. A changed percentage is updated from the canonical dataset.
- Calls `verify_connectivity`, creates both uniqueness constraints with
  `IF NOT EXISTS`, and writes all nodes and edges in **one transaction**. A
  failure rolls the transaction back and returns a non-zero exit status.
- Uses `MERGE` on the stable keys and parameterized `UNWIND` batches. Re-running
  against the same dataset creates zero nodes and zero relationships.
- Leaves nodes and edges outside the current dataset untouched. It never wipes
  the graph and never deletes an unrelated supplier or company.

### 10.5 Query output and failure contract

A successful query starts with `[GRAPH]`, includes supplier identity, scope,
direction, hop/relationship limits, and one deterministic line per relationship:

```text
[GRAPH] supplier=SUP-001 name=Norvend Components Ltd partition=train
scope=ownership-neighborhood direction=child_supplier->parent_company max_hops=3 max_relationships=25 truncated=false
ownership_relationships=2
1. hop=1 focus=yes SUP-001 "Norvend Components Ltd" -[JOINT_VENTURE_WITH 40.0%]-> ORG-001 "Meridian Industrial Holdings"
2. hop=2 focus=no SUP-014 "Lakeshore Carton Supply" -[JOINT_VENTURE_WITH 50.0%]-> ORG-001 "Meridian Industrial Holdings"
```

`focus=yes` marks the queried supplier's own edge; `focus=no` marks related
suppliers reached through the same company. The string never asserts legal
control, sanctions, financial risk, or an APPROVE/FLAG verdict.

Unknown suppliers, ambiguous identities, invalid configuration, an unavailable
database, query timeouts, and integrity failures return `[GRAPH ERROR]` strings
instead of raising. The error classes distinguish configuration
(`GraphConfigError`), connectivity (`GraphConnectionError`), query/timeout
(`GraphQueryError`), data integrity (`GraphDataIntegrityError`), and unknown or
ambiguous identity (`UnknownGraphEntityError`). The password is never included
in a message.

Queries run with a 10-second transaction timeout, a 5-second connection
timeout, and a 5-second driver retry window. There is no indefinite retry
loop.

### 10.6 Indexes

Only the two uniqueness constraints above are created. The partition is stored
as a `Supplier` property but no partition index is created in Module 4; graph
queries do not currently filter by partition. No full-text index is created:
contract and complaint text belongs to the FAISS/BM25 store.

### 10.7 Tests

Offline (no database): `tests/test_graph_schema.py`,
`tests/test_graph_pipeline.py`, `tests/test_graph_store.py`, and
`tests/test_graph_config.py` for `Neo4jSettings`.

Live (opt-in): `tests/test_graph_integration.py`, enabled with
`SUPPLIER_RISK_NEO4J_TESTS=1` (or `make test-graph`). The live tests cover
connectivity, constraints, canonical ingestion, idempotent re-ingestion, edge
direction, exact-name resolution, unknown and no-evidence behavior, the
3-hop/25-relationship limits, database unavailability, and connection closure.

### 10.8 Current graph limitations

One optional edge per supplier and no company-to-company edges in the
canonical dataset. There are no multiple parents, nested ownership
percentages, or supplier-to-supplier ownership edges. The query is bounded to
3 hops, but a deeper hierarchy is not supported by the current contract or
ingestion. Section 13 records the approval needed before expanding the model.

## 11. Shared identity and partition propagation

`supplier_id` is the **canonical identifier** linking all three stores:

| Store | Location of supplier identity |
|---|---|
| SQLite | `suppliers.supplier_id` primary key; foreign keys from `delivery_metrics`, `contracts`, `complaints`, `ownership_links` |
| Document retrieval | `supplier_id` on every document in `corpus.json` and on every vector entry in `dense.metadata.json`, alongside `source_type` and `source_id`; the BM25 payload uses the same document order |
| Neo4j | `Supplier.supplier_id` unique key, targeted by typed ownership edges |

Partition membership is not a separate lookup table. It is decided once by
`PARTITION_ID_RANGES` in `contracts.py`, embedded in each `SupplierRecord`, and
then copied unchanged by each store:

| Store | Partition representation | Source |
|---|---|---|
| Generated JSON | `SupplierRecord.partition` | Validated against the fixed mapping |
| SQLite | `suppliers.partition` column | Copied from the validated record |
| Document retrieval | `partition` on every document and every vector entry; BM25 positions align with the same document order | Copied from the validated supplier record |
| Neo4j | `Supplier.partition` property | Copied from the validated record |

Consumers should read the stored partition and may assert it equals
`partition_for_supplier_id(supplier_id)`. A mismatch indicates a corrupt or
stale artifact, not a valid repartitioning. Contracts and complaints belong to
their supplier's partition and cannot move independently.

### Status summary

| Area | Assignment | Current status | Remaining work |
|---|---|---|---|
| Supplier/contract/complaint/ownership fields | Domain-mandated categories | Implemented and validated | Must map these contracts |
| 24 suppliers, 14/4/6 split | Only ~20–30 and a held-out entity are mandated | Implemented; provisional counts | — |
| Fixed id-to-partition mapping | Not specified | Implemented and enforced | Must preserve it |
| SQLite tables | Store population mandated | Not implemented; schema proposed in section 8 | Later module |
| FAISS documents, embeddings, index | Store population mandated | Implemented in Module 3 (section 9) | Complete for contracts and complaints |
| BM25 lexical index and RRF hybrid retrieval | Hybrid BM25+dense required | Implemented in Module 3 (section 9) | Complete |
| Neo4j nodes and edges | Store population mandated | Implemented in Module 4 (section 10) | Complete for the canonical ownership graph |
| Multi-parent / nested ownership | Unspecified | Not supported | Open decision |

## 12. Validation errors

`parse_dataset_payload()` and `parse_dataset_json()` raise
`DatasetValidationError` with:

- the field path, such as `suppliers.0.profile.country_code`;
- a reason, such as a pattern or range failure;
- no rejected input values and no full record dumps.

Field-level Pydantic errors are capped at 20 entries, followed by
`... and N more error(s)`. Dataset-level integrity checks are different: all
issues found in one pass are joined into **one** error entry, so that entry can
list more than 20 individual problems (for example, every duplicate supplier id
in a malformed payload). There is no enforced 20-issue cap on that aggregated
message. The output remains input-free.

### 12.1 Retrieval errors (Module 3)

The retrieval stack raises typed project errors; it does not return empty results
for infrastructure failures. Messages never include credentials, rejected input
values, or full record dumps.

| Error | Raised for |
|---|---|
| `CorpusError` | Empty corpus, duplicate document or source id, unknown source record, source/supplier mismatch, invalid tokenizer or BM25 configuration, missing dataset input |
| `EmbeddingModelError` | Model cannot be loaded, inference fails, or embeddings are mis-shaped, non-finite, or zero-length |
| `RetrievalArtifactError` | Missing manifest, corpus, dense index, dense metadata, or lexical payload; unreadable or corrupt files; fingerprint, dimension, count, model, tokenizer, or ordering mismatch |
| `RetrievalQueryError` | Blank query, query with no searchable tokens, invalid `top_k`, invalid RRF `k`, invalid `supplier_id` or `partition` filter |
| `UnknownSupplierError` | A supplier wrapper entity that matches no supplier id or exact legal name |

`query_faiss(entity)` is the only layer that converts known infrastructure
errors (`RetrievalArtifactError`, `EmbeddingModelError`) and
`UnknownSupplierError` into tagged `[FAISS ERROR]` context strings, matching the
engine-level error contract in Guide Step 3. The generic
`HybridRetriever.query` API raises typed errors instead. No broad catch-all
handler is installed, so programming errors still surface.

## 13. Open decisions and limitations

### Open decisions requiring approval

1. **Ownership cardinality and depth.** The assignment names a
   subsidiary/ownership graph but does not specify whether a supplier may have
   multiple parents, whether companies may own companies, or how deep traversal
   must go. The current model supports one optional parent link per supplier and
   Module 4 implements exactly that; the query is bounded to 3 hops for future
   data. Approve any expansion of the contract before adding deeper edges.
2. **Parent-company normalization in SQLite.** Keep repeated parent id/name
   columns in `ownership_links`, or introduce a `parent_companies` table. The
   current proposal keeps the flat representation to avoid implying a hierarchy.
3. **Embedding model.** `all-MiniLM-L6-v2` (384 dimensions, CPU) is implemented
   and recorded in `manifest.json`. It remains a provisional model choice;
   changing it requires rebuilding every artifact.
4. **Dense index type.** `IndexFlatIP` over normalized vectors with JSON
   metadata is implemented. An approximate index is unnecessary at 66 documents
   and is only worth revisiting if the corpus grows substantially.
5. **Chunking fallback.** One document per source record is implemented and
   documented. No >512-token split policy exists; revisit if contract or
   complaint text grows beyond the model input window.
6. **Artifact migration.** `module3.v1` and `manifest.json` now version the
   retrieval artifacts and record the corpus fingerprint, embedding model,
   dimension, tokenizer version, and BM25 parameters. A schema or model change
   still requires rebuilding all artifacts together.
7. **Candidate window and fusion tuning.** The `4 × top_k` unfiltered candidate
   multiplier and `RRF k = 60` are documented defaults, not tuned values.
8. **Partition counts.** 14/4/6 remains provisional even though it is enforced.

### Limitations and assumptions

- Every supplier, company, contract, complaint, and numeric value is
  **fictional**. No claim is made about any real organization.
- The metric distributions are hand-chosen for demonstration; they are not
  calibrated to a real industry and cannot support claims of model accuracy.
- Metrics are one annual snapshot per supplier (`FY2025`), not a time series.
- Contract and complaint text comes from a small template set, so semantic
  diversity is limited.
- Ownership is a single optional parent link per supplier; deeper corporate
  structures are unsupported and must not be inferred from the code.
- Empty complaints mean "no synthetic complaint record", not "no risk".
- Module 4 populates Neo4j with the canonical ownership graph. SQLite remains
  unpopulated, and parallel retrieval, risk scoring, DSPy, the judge, and the UI
  are not implemented.
- Retrieval is exact FAISS search over 66 documents; recall, ranking quality, and
  latency at larger corpus sizes are unmeasured.
- The embedding model, the `4 × top_k` candidate window, and `RRF k = 60` are
  defaults, not tuned values. Hybrid ranking has not been evaluated against
  relevance labels.
- A build replaces its artifact files one at a time: each file is atomic, but the
  set is not. The manifest fingerprint and compatibility checks detect a mixed
  set at load time instead of serving it.
- `SCHEMA_VERSION` in `contracts.py` is the runtime source for
  `schema_version`; the `SchemaVersion` literal alias is checked by a test. Any
  field, range, identifier, or partition change requires a schema-version
  change, documentation update, and test update, and makes existing generated
  artifacts incompatible until regenerated. `ARTIFACT_VERSION` (`module3.v1`)
  plays the same role for retrieval artifacts: any change to document identity,
  text construction, metadata fields, artifact layout, or index configuration
  requires a version bump and a rebuild.
