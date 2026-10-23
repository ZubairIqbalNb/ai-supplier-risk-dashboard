"""Neo4j-backed ownership graph store for Module 4.

The driver is created only by :meth:`Neo4jGraphStore.from_settings`, never at
import time. Every query is parameterized, every session is closed, and driver
failures are translated into typed project errors that never include the
password. Traversal is bounded by :data:`schema.MAX_HOPS` and evidence is
truncated by :func:`schema.build_evidence`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Final, Self, TypeVar, cast

from neo4j import Driver, GraphDatabase, Transaction
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    DriverError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)

from supplier_risk.config import Neo4jSettings
from supplier_risk.contracts import OwnershipType
from supplier_risk.graph.errors import (
    GraphConfigError,
    GraphConnectionError,
    GraphDataIntegrityError,
    GraphError,
    GraphQueryError,
    UnknownGraphEntityError,
)
from supplier_risk.graph.schema import (
    COMPANY_LABEL,
    MAX_HOPS,
    OWNERSHIP_RELATIONSHIP_TYPES,
    SUPPLIER_LABEL,
    GraphPlan,
    OwnershipEvidenceRow,
    OwnershipRow,
    SupplierIdentity,
)

DEFAULT_QUERY_TIMEOUT_SECONDS: Final = 10.0
DEFAULT_CONNECTION_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_MAX_RETRY_TIME_SECONDS: Final = 5.0
DEFAULT_CONNECTION_ACQUISITION_TIMEOUT_SECONDS: Final = 10.0
DEFAULT_PATH_LIMIT: Final = 200

_T = TypeVar("_T")

_CONSTRAINT_STATEMENTS: Final[tuple[tuple[str, str, str], ...]] = (
    (
        SUPPLIER_LABEL,
        "supplier_id",
        (
            "CREATE CONSTRAINT supplier_id_unique IF NOT EXISTS "
            f"FOR (s:{SUPPLIER_LABEL}) REQUIRE s.supplier_id IS UNIQUE"
        ),
    ),
    (
        COMPANY_LABEL,
        "parent_company_id",
        (
            "CREATE CONSTRAINT company_id_unique IF NOT EXISTS "
            f"FOR (c:{COMPANY_LABEL}) REQUIRE c.parent_company_id IS UNIQUE"
        ),
    ),
)

_EXISTING_OWNERSHIP_QUERY: Final = f"""
MATCH (s:{SUPPLIER_LABEL})-[r:SUBSIDIARY_OF|JOINT_VENTURE_WITH]->(c:{COMPANY_LABEL})
WHERE s.supplier_id IN $supplier_ids
RETURN s.supplier_id AS child_supplier_id,
       c.parent_company_id AS parent_company_id,
       type(r) AS relationship_type,
       r.ownership_percentage AS ownership_percentage
ORDER BY child_supplier_id, parent_company_id, relationship_type
"""

_SUPPLIER_UPSERT_QUERY: Final = f"""
UNWIND $rows AS row
MERGE (s:{SUPPLIER_LABEL} {{supplier_id: row.supplier_id}})
SET s.legal_name = row.legal_name,
    s.country_code = row.country_code,
    s.sector = row.sector,
    s.partition = row.partition
"""

_COMPANY_UPSERT_QUERY: Final = f"""
UNWIND $rows AS row
MERGE (c:{COMPANY_LABEL} {{parent_company_id: row.parent_company_id}})
SET c.parent_company_name = row.parent_company_name
"""

def _ownership_upsert_query(relationship_type: OwnershipType) -> str:
    """Return the fixed-type upsert query; the type never comes from user input."""
    return f"""
UNWIND $rows AS row
MATCH (s:{SUPPLIER_LABEL} {{supplier_id: row.child_supplier_id}})
MATCH (c:{COMPANY_LABEL} {{parent_company_id: row.parent_company_id}})
MERGE (s)-[r:{relationship_type}]->(c)
SET r.ownership_percentage = row.ownership_percentage
RETURN count(r) AS merged
"""

_CANONICAL_COUNTS_QUERY: Final = f"""
CALL () {{
    MATCH (s:{SUPPLIER_LABEL})
    WHERE s.supplier_id IN $supplier_ids
    RETURN count(s) AS suppliers
}}
CALL () {{
    MATCH (c:{COMPANY_LABEL})
    WHERE c.parent_company_id IN $company_ids
    RETURN count(c) AS companies
}}
CALL () {{
    MATCH (s:{SUPPLIER_LABEL})-[r:SUBSIDIARY_OF|JOINT_VENTURE_WITH]->(:{COMPANY_LABEL})
    WHERE s.supplier_id IN $supplier_ids
    RETURN count(r) AS relationships
}}
RETURN suppliers, companies, relationships
"""

_RESOLVE_SUPPLIER_QUERY: Final = f"""
MATCH (s:{SUPPLIER_LABEL})
WHERE toUpper(s.supplier_id) = toUpper($entity)
   OR toLower(s.legal_name) = toLower($entity)
RETURN s.supplier_id AS supplier_id,
       s.legal_name AS legal_name,
       s.partition AS partition,
       s.sector AS sector,
       s.country_code AS country_code
ORDER BY CASE WHEN toUpper(s.supplier_id) = toUpper($entity) THEN 0 ELSE 1 END,
         s.supplier_id
LIMIT 2
"""

# Discovery is bounded to three undirected hops from the subject supplier so a
# shared parent company's other subsidiaries are reachable. Each returned edge
# keeps its stored direction (child supplier -> parent company).
_OWNERSHIP_EVIDENCE_QUERY: Final = f"""
MATCH path = (s:{SUPPLIER_LABEL} {{supplier_id: $supplier_id}})
             -[:SUBSIDIARY_OF|JOINT_VENTURE_WITH*1..{MAX_HOPS}]-(other)
WITH path, length(path) AS hops
ORDER BY hops, coalesce(other.supplier_id, other.parent_company_id)
LIMIT $path_limit
UNWIND relationships(path) AS rel
RETURN coalesce(startNode(rel).supplier_id, startNode(rel).parent_company_id) AS child_key,
       coalesce(startNode(rel).legal_name, startNode(rel).parent_company_name) AS child_name,
       type(rel) AS relationship_type,
       rel.ownership_percentage AS ownership_percentage,
       coalesce(endNode(rel).supplier_id, endNode(rel).parent_company_id) AS parent_key,
       coalesce(endNode(rel).legal_name, endNode(rel).parent_company_name) AS parent_name,
       min(hops) AS hops
ORDER BY hops, child_key, parent_key, relationship_type
"""

_CONSTRAINT_NAMES_QUERY: Final = """
SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties
WHERE type = 'UNIQUENESS'
  AND ((labelsOrTypes = ['Supplier'] AND properties = ['supplier_id'])
    OR (labelsOrTypes = ['Company'] AND properties = ['parent_company_id']))
RETURN name
ORDER BY name
"""

_CONSTRAINT_DEFINITIONS_QUERY: Final = """
SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties
WHERE type = 'UNIQUENESS'
RETURN labelsOrTypes AS labels, properties AS properties
"""


@dataclass(frozen=True, slots=True)
class WriteStats:
    """Counts reported by one ingestion transaction."""

    suppliers_created: int
    companies_created: int
    relationships_created: int


@dataclass(frozen=True, slots=True)
class GraphCounts:
    """Canonical graph counts for one ingestion plan."""

    suppliers: int
    companies: int
    relationships: int


class Neo4jGraphStore:
    """Thin, testable owner of one Neo4j driver."""

    def __init__(
        self,
        driver: Driver,
        *,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
        path_limit: int = DEFAULT_PATH_LIMIT,
    ) -> None:
        if query_timeout <= 0:
            raise GraphConfigError("Neo4j query timeout must be positive")
        if isinstance(path_limit, bool) or not isinstance(path_limit, int) or path_limit < 1:
            raise GraphConfigError("Neo4j path limit must be a positive integer")
        self._driver = driver
        self._query_timeout = query_timeout
        self._path_limit = path_limit
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: Neo4jSettings,
        *,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT_SECONDS,
        connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS,
        max_retry_time: float = DEFAULT_MAX_RETRY_TIME_SECONDS,
        connection_acquisition_timeout: float = (
            DEFAULT_CONNECTION_ACQUISITION_TIMEOUT_SECONDS
        ),
    ) -> Neo4jGraphStore:
        """Create a driver from validated settings without opening a connection."""
        try:
            driver = GraphDatabase.driver(
                settings.uri,
                auth=(settings.user, settings.password.get_secret_value()),
                connection_timeout=connection_timeout,
                max_transaction_retry_time=max_retry_time,
                connection_acquisition_timeout=connection_acquisition_timeout,
            )
        except (ConfigurationError, DriverError, ValueError):
            raise GraphConfigError(
                "could not create a Neo4j driver from the configured URI"
            ) from None
        return cls(driver, query_timeout=query_timeout)

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the driver; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._driver.close()

    def verify_connectivity(self) -> None:
        """Raise a typed error when Neo4j cannot be reached or authentication fails."""
        self._ensure_open()
        try:
            self._driver.verify_connectivity()
        except GraphError:
            raise
        except Exception as error:  # noqa: BLE001 - driver boundary
            raise _translate_driver_error("checking connectivity", error) from None

    def ensure_constraints(self) -> None:
        """Create each uniqueness constraint only when its key is not already constrained.

        Checking the definitions rather than the names avoids an
        ``EquivalentSchemaRuleAlreadyExists`` failure when an equivalent
        constraint already exists under a different name.
        """
        existing = self.constraint_definitions()
        for label, property_name, statement in _CONSTRAINT_STATEMENTS:
            if (label, property_name) in existing:
                continue

            def apply_constraint(tx: Transaction, query: str = statement) -> None:
                tx.run(query).consume()

            self._write_transaction("creating constraints", apply_constraint)

    def constraint_definitions(self) -> set[tuple[str, str]]:
        """Return the uniqueness-constrained ``(label, property)`` pairs."""
        rows = self._read("reading constraints", _CONSTRAINT_DEFINITIONS_QUERY, {})
        definitions: set[tuple[str, str]] = set()
        for row in rows:
            labels = row.get("labels") or []
            properties = row.get("properties") or []
            for label in labels:
                for property_name in properties:
                    definitions.add((str(label), str(property_name)))
        return definitions

    def constraint_names(self) -> tuple[str, ...]:
        """Return the names of the ownership uniqueness constraints present."""
        rows = self._read("reading constraints", _CONSTRAINT_NAMES_QUERY, {})
        return tuple(str(row["name"]) for row in rows)

    def fetch_existing_ownership(
        self, supplier_ids: tuple[str, ...]
    ) -> tuple[OwnershipRow, ...]:
        """Return ownership edges already stored for the given supplier ids."""
        if not supplier_ids:
            return ()
        rows = self._read(
            "reading existing ownership",
            _EXISTING_OWNERSHIP_QUERY,
            {"supplier_ids": list(supplier_ids)},
        )
        return tuple(_ownership_row_from_record(row) for row in rows)

    def upsert_plan(self, plan: GraphPlan) -> WriteStats:
        """Write all nodes and ownership edges in one transaction."""

        def work(tx: Transaction) -> WriteStats:
            supplier_summary = tx.run(
                _SUPPLIER_UPSERT_QUERY, {"rows": [asdict(row) for row in plan.suppliers]}
            ).consume()
            company_summary = tx.run(
                _COMPANY_UPSERT_QUERY, {"rows": [asdict(row) for row in plan.companies]}
            ).consume()

            relationships_created = 0
            for relationship_type in OWNERSHIP_RELATIONSHIP_TYPES:
                rows = [
                    row
                    for row in plan.ownership
                    if row.relationship_type == relationship_type
                ]
                if not rows:
                    continue
                result = tx.run(
                    _ownership_upsert_query(relationship_type),
                    {"rows": [asdict(row) for row in rows]},
                )
                records = [dict(record) for record in result]
                summary = result.consume()
                merged = int(records[0]["merged"]) if records else 0
                if merged != len(rows):
                    raise GraphDataIntegrityError(
                        "ownership endpoints disappeared during ingestion; "
                        "the transaction was rolled back"
                    )
                relationships_created += summary.counters.relationships_created

            return WriteStats(
                suppliers_created=supplier_summary.counters.nodes_created,
                companies_created=company_summary.counters.nodes_created,
                relationships_created=relationships_created,
            )

        return self._write_transaction("writing the ownership graph", work)

    def canonical_counts(
        self, *, supplier_ids: tuple[str, ...], company_ids: tuple[str, ...]
    ) -> GraphCounts:
        """Return stored counts restricted to one dataset's identifiers."""
        rows = self._read(
            "reading graph counts",
            _CANONICAL_COUNTS_QUERY,
            {"supplier_ids": list(supplier_ids), "company_ids": list(company_ids)},
        )
        row = rows[0]
        return GraphCounts(
            suppliers=int(row["suppliers"]),
            companies=int(row["companies"]),
            relationships=int(row["relationships"]),
        )

    def resolve_supplier(self, entity: str) -> SupplierIdentity | None:
        """Resolve a supplier id or exact case-insensitive legal name.

        Returns ``None`` when no node matches and raises
        :class:`UnknownGraphEntityError` when more than one node matches.
        """
        self._ensure_open()
        text = entity.strip() if isinstance(entity, str) else ""
        if not text:
            raise UnknownGraphEntityError("supplier identity must not be empty")

        rows = self._read(
            "resolving the supplier", _RESOLVE_SUPPLIER_QUERY, {"entity": text}
        )
        if not rows:
            return None
        if len(rows) > 1:
            raise UnknownGraphEntityError(f"supplier identity {text!r} is ambiguous")
        row = rows[0]
        identity = SupplierIdentity(
            supplier_id=str(row["supplier_id"]),
            legal_name=str(row["legal_name"]),
            partition=str(row["partition"]),
            sector=str(row["sector"]),
            country_code=str(row["country_code"]),
        )
        return identity

    def fetch_ownership_evidence(
        self, supplier_id: str
    ) -> tuple[OwnershipEvidenceRow, ...]:
        """Return bounded, deduplicated ownership rows for one supplier."""
        rows = self._read(
            "reading ownership evidence",
            _OWNERSHIP_EVIDENCE_QUERY,
            {"supplier_id": supplier_id, "path_limit": self._path_limit},
        )
        return tuple(_evidence_row_from_record(row) for row in rows)

    def _ensure_open(self) -> None:
        if self._closed:
            raise GraphConnectionError("the Neo4j graph store is already closed")

    def _read(
        self, operation: str, query: str, parameters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        try:
            with self._driver.session() as session:
                tx = session.begin_transaction(timeout=self._query_timeout)
                try:
                    result = tx.run(query, parameters)
                    rows = [dict(record) for record in result]
                    tx.commit()
                except BaseException:
                    try:
                        tx.rollback()
                    except Exception:  # noqa: BLE001, S110 - best-effort rollback
                        pass
                    raise
        except GraphError:
            raise
        except Exception as error:  # noqa: BLE001 - driver boundary
            raise _translate_driver_error(operation, error) from None
        return rows

    def _write_transaction(self, operation: str, work: Callable[[Transaction], _T]) -> _T:
        self._ensure_open()
        try:
            with self._driver.session() as session:
                tx = session.begin_transaction(timeout=self._query_timeout)
                try:
                    value = work(tx)
                    tx.commit()
                except BaseException:
                    try:
                        tx.rollback()
                    except Exception:  # noqa: BLE001, S110 - best-effort rollback
                        pass
                    raise
        except GraphError:
            raise
        except Exception as error:  # noqa: BLE001 - driver boundary
            raise _translate_driver_error(operation, error) from None
        return value


def _ownership_row_from_record(record: dict[str, Any]) -> OwnershipRow:
    relationship_type = record["relationship_type"]
    percentage = record["ownership_percentage"]
    if relationship_type not in OWNERSHIP_RELATIONSHIP_TYPES:
        raise GraphDataIntegrityError(
            f"graph contains an unsupported ownership relationship type: {relationship_type!r}"
        )
    if not isinstance(percentage, (int, float)) or isinstance(percentage, bool):
        raise GraphDataIntegrityError(
            f"graph ownership edge {record['child_supplier_id']} has no valid percentage"
        )
    return OwnershipRow(
        child_supplier_id=str(record["child_supplier_id"]),
        parent_company_id=str(record["parent_company_id"]),
        relationship_type=cast(OwnershipType, relationship_type),
        ownership_percentage=float(percentage),
    )


def _evidence_row_from_record(record: dict[str, Any]) -> OwnershipEvidenceRow:
    relationship_type = record["relationship_type"]
    percentage = record["ownership_percentage"]
    hops = record["hops"]
    if relationship_type not in OWNERSHIP_RELATIONSHIP_TYPES:
        raise GraphDataIntegrityError(
            f"graph contains an unsupported ownership relationship type: {relationship_type!r}"
        )
    if not isinstance(percentage, (int, float)) or isinstance(percentage, bool):
        raise GraphDataIntegrityError("graph ownership edge has no valid percentage")
    if isinstance(hops, bool) or not isinstance(hops, int):
        raise GraphDataIntegrityError("graph ownership evidence has an invalid hop count")
    return OwnershipEvidenceRow(
        child_key=str(record["child_key"]),
        child_name=str(record["child_name"]),
        parent_key=str(record["parent_key"]),
        parent_name=str(record["parent_name"]),
        relationship_type=cast(OwnershipType, relationship_type),
        ownership_percentage=float(percentage),
        hops=hops,
    )


def _translate_driver_error(operation: str, error: Exception) -> GraphError:
    """Map a driver exception to a project error without exposing credentials."""
    if isinstance(error, AuthError):
        return GraphConfigError(
            "Neo4j authentication failed; check NEO4J_USER and NEO4J_PASSWORD."
        )
    if isinstance(error, (ServiceUnavailable, SessionExpired)):
        return GraphConnectionError(
            f"Neo4j is unavailable at the configured URI while {operation}."
        )
    if isinstance(error, Neo4jError):
        code = error.code or ""
        if "TimedOut" in code or "Timeout" in code:
            return GraphQueryError(f"Neo4j timed out while {operation}.")
        return GraphQueryError(
            f"Neo4j rejected the request while {operation} ({type(error).__name__})."
        )
    if isinstance(error, ConfigurationError):
        return GraphConfigError("the configured Neo4j URI or driver options are invalid")
    if isinstance(error, DriverError):
        return GraphConnectionError(
            f"could not communicate with Neo4j while {operation}."
        )
    return GraphQueryError(
        f"unexpected Neo4j failure while {operation} ({type(error).__name__})."
    )
