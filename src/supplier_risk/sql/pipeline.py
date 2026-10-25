"""Assignment-facing SQL retrieval interface for Module 5.

``query_sql(entity)`` resolves a supplier id or exact legal name against the
SQLite structured store and returns a labeled ``[SQL]`` context covering only
delivery and defect metrics. Unknown suppliers, an unavailable database, and
integrity failures are converted into tagged ``[SQL ERROR]`` strings, matching
the engine-level contract used by ``query_faiss`` and ``query_graph`` so the
Module 5 orchestrator can treat a failed source separately from a successful
query with no evidence.

The store is opened per call (one short-lived read-only connection), so this
function is safe to call from several orchestrator threads at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from supplier_risk.sql.errors import (
    SqlConfigError,
    SqlDataIntegrityError,
    SqlError,
    SqlQueryError,
    SqlStoreUnavailableError,
    UnknownSqlEntityError,
)
from supplier_risk.sql.schema import (
    DeliveryEvidence,
    DeliveryMetricsRow,
    SupplierRow,
    build_delivery_evidence,
)
from supplier_risk.sql.store import DEFAULT_DATABASE_PATH, SqliteStore


class SqlQueryStore(Protocol):
    """Minimal store surface needed by :func:`query_sql`."""

    def resolve_supplier(self, entity: str) -> SupplierRow | None: ...

    def fetch_delivery_metrics(self, supplier_id: str) -> tuple[DeliveryMetricsRow, ...]: ...


def query_sql(
    entity: str,
    *,
    store: SqlQueryStore | None = None,
    database_path: Path | str = DEFAULT_DATABASE_PATH,
) -> str:
    """Return labeled delivery and defect evidence for ``entity``.

    ``entity`` is a supplier id such as ``SUP-007`` or an exact supplier legal
    name (case-insensitive). Matching is exact; there is no fuzzy fallback.

    The context covers SQLite structured metrics only. A supplier that exists
    without a stored observation returns a normal ``[SQL]`` context with
    ``delivery_metrics=0``; that no-evidence result is not an error. Unknown
    suppliers, a missing or unreadable database, and data-integrity failures are
    returned as ``[SQL ERROR]`` strings rather than raised to the caller.
    """
    if not isinstance(entity, str) or not entity.strip():
        return "[SQL ERROR] supplier identity must not be empty"

    active_store: SqlQueryStore = store if store is not None else SqliteStore(database_path)
    return _query_with_store(active_store, entity)


def _query_with_store(store: SqlQueryStore, entity: str) -> str:
    """Run one query and normalize every project error into a tagged string."""
    try:
        supplier = store.resolve_supplier(entity)
        if supplier is None:
            return (
                f"[SQL ERROR] unknown supplier identity {entity!r}; pass a supplier id "
                "such as SUP-007 or an exact supplier legal name"
            )
        rows = store.fetch_delivery_metrics(supplier.supplier_id)
        return format_sql_context(build_delivery_evidence(supplier, rows))
    except SqlError as error:
        return _error_context(error)


def _error_context(error: SqlError) -> str:
    if isinstance(error, UnknownSqlEntityError):
        return f"[SQL ERROR] {error}"
    if isinstance(error, SqlConfigError):
        return f"[SQL ERROR] SQL configuration is invalid: {error}"
    if isinstance(error, SqlStoreUnavailableError):
        return f"[SQL ERROR] SQL retrieval unavailable: {error}"
    if isinstance(error, SqlQueryError):
        return f"[SQL ERROR] SQL query failed: {error}"
    if isinstance(error, SqlDataIntegrityError):
        return f"[SQL ERROR] SQL data integrity failure: {error}"
    return f"[SQL ERROR] SQL retrieval failed: {error}"


def format_sql_context(evidence: DeliveryEvidence) -> str:
    """Format delivery evidence deterministically, preserving stored values."""
    supplier = evidence.supplier
    lines = [
        (
            f"[SQL] supplier={supplier.supplier_id} name={supplier.legal_name} "
            f"partition={supplier.partition}"
        ),
        "scope=delivery+defect-metrics source=sqlite table=delivery_metrics",
        f"delivery_metrics={len(evidence.metrics)}",
    ]

    if not evidence.metrics:
        lines.append("No delivery or defect metrics were found for this supplier.")
        return "\n".join(lines)

    for rank, row in enumerate(evidence.metrics, start=1):
        lines.append(
            f"{rank}. period={row.reporting_period} "
            f"on_time_delivery_rate={row.on_time_delivery_rate} "
            f"defect_rate={row.defect_rate} "
            f"average_lead_time_days={row.average_lead_time_days} "
            f"orders_fulfilled={row.orders_fulfilled}"
        )
    return "\n".join(lines)
