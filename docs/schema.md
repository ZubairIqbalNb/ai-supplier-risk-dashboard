# Supplier Data Schema (Module 2)

**Project:** AI Supplier Risk Dashboard  
**Phase:** Phase 2 — Multi-Source Retrieval  
**Schema version:** `module2.v1`  
**Status:** Module 2 data foundation; no retrieval, scoring, or AI behavior

This note defines the canonical raw supplier data used by later modules. It
covers the structured, semantic, and graph stores required by the assignment,
but Module 2 only generates the validated source records. It does not populate
SQLite, FAISS, or Neo4j yet.

**Status legend**

- **Mandated** — required by the assignment PDFs.
- **Implemented** — present in Module 2 code and covered by tests.
- **Proposed** — future-store design recorded here; not implemented.
- **Open** — requires a design decision or approval before implementation.

## 1. Requirement status

### Confirmed by the assignment

- The domain is **Supplier Risk Dashboard** (Problem Statement, domain item 7).
- Structured facts are delivery/defect metrics for SQL.
- Free text is contract/complaint text for semantic search.
- Relationships are subsidiary/ownership links for the graph store.
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
validation detail; the table above is the single-page summary. Sections 8–11
record the **proposed** physical schemas for the stores that Module 3+ will
implement.

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
both sides by id; section 10 describes the proposed Neo4j edge direction.

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
not specify graph depth or ownership cardinality; see section 13 for the open
decision that must be resolved before Neo4j implementation.

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

## 9. Proposed FAISS index and metadata

**Status:** Proposed. Module 2 does not chunk, embed, build an index, or persist
vectors. No embedding or FAISS dependency is installed.

### 9.1 Source records

The semantic store uses exactly the free-text records already validated in
`SupplierRecord`:

- 32 `Contract` records (one or more per supplier).
- 34 `Complaint` records (zero or more per supplier).

Those counts are properties of seed `20261019`, not fixed contract rules.
Contract text currently runs 393–422 characters and complaint text 112–150
characters, so the whole corpus is small.

### 9.2 Chunking strategy

- **Proposed initial setting:** one chunk per source record. At current text
  lengths this preserves the whole agreement or complaint and keeps
  `source_id` unambiguous.
- **Proposed fallback:** if a record exceeds a 512-token embedding budget, split
  it on paragraph, then sentence, boundaries with 64 tokens of overlap. Chunk
  order must be deterministic.
- **Implemented:** none. No tokenizer or splitter has been selected.

### 9.3 Embedding model and index type

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384
  dimensions) is recorded as a provisional choice in the traceability register.
  It is **not yet selected by code** and must be confirmed before Module 3.
- **FAISS index:** proposed `IndexFlatIP` over L2-normalized vectors, which is
  exact cosine similarity. Exhaustive search is appropriate for a few dozen
  chunks. An approximate index (for example HNSW) is unnecessary at this size
  and is an open decision only if the corpus grows.
- **Implemented:** none.

### 9.4 Proposed artifact paths

| Artifact | Path | Contents |
|---|---|---|
| FAISS index | `artifacts/faiss/contracts_complaints.faiss` | One vector per chunk |
| Metadata | `artifacts/faiss/metadata.jsonl` | One JSON object per vector, same order as the index |
| Manifest | `artifacts/faiss/manifest.json` | Schema version, embedding model, dimension, metric, chunk count, source counts, generator seed |

`artifacts/` and `*.faiss` are already Git-ignored.

### 9.5 Required metadata fields

| Field | Type | Purpose |
|---|---|---|
| `vector_id` | integer | 0-based position in the FAISS index |
| `chunk_id` | string | Stable id, e.g. `CON-007-01#0` |
| `source_type` | `"contract"` or `"complaint"` | Source record type |
| `source_id` | string | `contract_id` or `complaint_id` |
| `supplier_id` | string | Canonical supplier identity; never omitted |
| `partition` | `train` / `dev` / `held_out` | Copied from the enclosing supplier record |
| `chunk_index` | integer | Chunk position within the source record |
| `text` | string | Exact embedded text |
| `contract_type` | string or null | Contract-only categorical filter |
| `effective_date` | ISO date or null | Contract-only metadata |
| `severity` | string or null | Complaint-only categorical filter |
| `category` | string or null | Complaint-only metadata |
| `received_date` | ISO date or null | Complaint-only metadata |

`supplier_id` is the stable join key. `source_id` remains the identifier of the
individual contract or complaint, so a chunk always names both its source record
and its supplier.

### 9.6 Ordering and reload alignment

1. Build chunks in one deterministic pass ordered by `supplier_id`, then
   contracts before complaints, then `chunk_index`.
2. Embed in that order and add vectors to FAISS without custom ids, so FAISS
   positional ids are `0..n-1`.
3. Write metadata rows in the same order with `vector_id` equal to the row
   index.
4. On reload, assert `index.ntotal == number of metadata rows`, that
   `metadata[i]["vector_id"] == i`, and that the loaded dimension, metric, and
   embedding model match `manifest.json`.

This keeps vector ids and metadata aligned across persistence and reload without
relying on implicit FAISS iteration order. If the corpus is rebuilt with a
different chunking strategy or embedding model, the manifest and artifact set
must be regenerated together.

## 10. Proposed Neo4j graph schema

**Status:** Proposed. Module 2 does not start Neo4j, create constraints, or write
nodes or relationships. No driver is installed.

### 10.1 Nodes

| Label | Identifying key | Properties | Uniqueness constraint |
|---|---|---|---|
| `Supplier` | `supplier_id` | `supplier_id`, `legal_name`, `country_code`, `sector`, `partition` | `REQUIRE s.supplier_id IS UNIQUE` |
| `Company` | `parent_company_id` | `parent_company_id`, `parent_company_name` | `REQUIRE c.parent_company_id IS UNIQUE` |

A supplier maps one-to-one to a `Supplier` node whose key is the same
`supplier_id` used in SQLite and FAISS metadata. Parent companies are `Company`
nodes; they are not suppliers and their ids never overlap the `SUP-` namespace.

### 10.2 Relationships

The current contract stores a parent id and a child supplier id plus a
`relationship_type`. The proposed mapping uses that type as the relationship
type and attaches the percentage:

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
supplier. The direction (supplier → company) is chosen to match the field names
`child_supplier_id` and `parent_company_id`; the source contract stores the same
edge as parent fields pointing at a child, so the loader must reverse the
storage direction rather than the meaning.

`ownership_percentage` is the only relationship property in the current
contract. It is a fraction of 100, not a fraction of 1, consistent with
`OwnershipLink`.

### 10.3 Idempotent writes and duplicate parents

- Load companies before suppliers, or use `MERGE` on the identifying key.
- `MERGE (c:Company {parent_company_id: row.parent_company_id})` creates each
  company once, regardless of how many suppliers share it.
- On an existing company, compare `parent_company_name` with the incoming value
  and abort on mismatch. The dataset validator already rejects an id mapped to
  two names.
- `MERGE (s:Supplier {supplier_id: row.child_supplier_id})`.
- `MERGE (s)-[r:SUBSIDIARY_OF]->(c) SET r.ownership_percentage = ...` (or the
  joint-venture type) makes re-running the loader idempotent.

### 10.4 Indexes

- Unique constraints on `Supplier.supplier_id` and `Company.parent_company_id`
  as above.
- Optional index on `Supplier.partition` for evaluation-time filtering.
- No full-text index is proposed: contract and complaint text belongs to the
  FAISS store, and duplicating it in Neo4j would undermine the requirement that
  each engine return different information.

### 10.5 Current graph limitation

One optional edge per supplier. There are no supplier-to-supplier edges,
company-to-company edges, multiple parents, nested ownership percentages, or
arbitrary-depth traversal in the current contract. Section 13 records the
approval needed before expanding the model.

## 11. Shared identity and partition propagation

`supplier_id` is the **canonical identifier** linking all three stores:

| Store | Location of supplier identity |
|---|---|
| SQLite | `suppliers.supplier_id` primary key; foreign keys from `delivery_metrics`, `contracts`, `complaints`, `ownership_links` |
| FAISS metadata | `supplier_id` on every chunk, alongside `source_type` and `source_id` |
| Neo4j | `Supplier.supplier_id` unique key, targeted by typed ownership edges |

Partition membership is not a separate lookup table. It is decided once by
`PARTITION_ID_RANGES` in `contracts.py`, embedded in each `SupplierRecord`, and
then copied unchanged by each store:

| Store | Partition representation | Source |
|---|---|---|
| Generated JSON | `SupplierRecord.partition` | Validated against the fixed mapping |
| SQLite | `suppliers.partition` column | Copied from the validated record |
| FAISS metadata | `partition` field on every chunk | Copied from the chunk's supplier record |
| Neo4j | `Supplier.partition` property | Copied from the validated record |

Consumers should read the stored partition and may assert it equals
`partition_for_supplier_id(supplier_id)`. A mismatch indicates a corrupt or
stale artifact, not a valid repartitioning. Contracts and complaints belong to
their supplier's partition and cannot move independently.

### Status summary

| Area | Assignment | Module 2 | Later stores |
|---|---|---|---|
| Supplier/contract/complaint/ownership fields | Domain-mandated categories | Implemented and validated | Must map these contracts |
| 24 suppliers, 14/4/6 split | Only ~20–30 and a held-out entity are mandated | Implemented; provisional counts | — |
| Fixed id-to-partition mapping | Not specified | Implemented and enforced | Must preserve it |
| SQLite tables | Store population mandated | Not implemented; schema proposed in section 8 | Module 3 |
| FAISS chunks, embeddings, index | Store population mandated | Not implemented; design proposed in section 9 | Module 3 |
| Neo4j nodes and edges | Store population mandated | Not implemented; design proposed in section 10 | Module 4 |
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

## 13. Open decisions and limitations

### Open decisions requiring approval

1. **Ownership cardinality and depth.** The assignment names a
   subsidiary/ownership graph but does not specify whether a supplier may have
   multiple parents, whether companies may own companies, or how deep traversal
   must go. The current model supports one optional parent link per supplier.
   Approve any expansion before Neo4j implementation.
2. **Parent-company normalization in SQLite.** Keep repeated parent id/name
   columns in `ownership_links`, or introduce a `parent_companies` table. The
   current proposal keeps the flat representation to avoid implying a hierarchy.
3. **Embedding model.** `all-MiniLM-L6-v2` is provisional; no model has been
   selected in code.
4. **FAISS index type and artifact format.** `IndexFlatIP` plus JSONL metadata
   is proposed, not implemented. Confirm before Module 3.
5. **Chunking fallback.** One chunk per record is proposed for the current short
   texts; the >512-token split policy is not implemented.
6. **Schema/artifact migration.** `schema_version` is the only version marker
   for the JSON dataset; separate index/artifact versioning remains to be
   defined.
7. **Partition counts.** 14/4/6 remains provisional even though it is enforced.

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
- Module 2 does not populate SQLite, FAISS, or Neo4j, and does not implement
  retrieval, risk scoring, DSPy, the judge, or the UI.
- `SCHEMA_VERSION` in `contracts.py` is the runtime source for
  `schema_version`; the `SchemaVersion` literal alias is checked by a test. Any
  field, range, identifier, or partition change requires a schema-version
  change, documentation update, and test update, and makes existing generated
  artifacts incompatible until regenerated.
