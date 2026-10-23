"""Assignment-facing graph retrieval interface for Module 4.

``query_graph(entity)`` resolves a supplier id or exact legal name, returns a
labeled ``[GRAPH]`` ownership-evidence string, and converts every failure into a
tagged ``[GRAPH ERROR]`` string so a future orchestrator can treat a failed
source separately from a successful query with no evidence.

The default call creates a short-lived store from the validated Neo4j settings
and closes it before returning. Tests and callers that already own a store can
pass one explicitly; no connection is opened at import time.
"""

from __future__ import annotations

from typing import Protocol

from supplier_risk.config import ConfigError, load_neo4j_settings
from supplier_risk.graph.errors import (
    GraphConfigError,
    GraphConnectionError,
    GraphDataIntegrityError,
    GraphError,
    GraphQueryError,
    UnknownGraphEntityError,
)
from supplier_risk.graph.schema import (
    MAX_HOPS,
    MAX_OWNERSHIP_RELATIONSHIPS,
    OwnershipEvidence,
    OwnershipEvidenceRow,
    SupplierIdentity,
    build_evidence,
)
from supplier_risk.graph.store import Neo4jGraphStore


class GraphQueryStore(Protocol):
    """Minimal store surface needed by :func:`query_graph`."""

    def resolve_supplier(self, entity: str) -> SupplierIdentity | None: ...

    def fetch_ownership_evidence(self, supplier_id: str) -> tuple[OwnershipEvidenceRow, ...]: ...


def query_graph(entity: str, *, store: GraphQueryStore | None = None) -> str:
    """Return labeled ownership evidence for ``entity``.

    ``entity`` is a supplier id such as ``SUP-007`` or an exact supplier legal
    name (case-insensitive). Matching is exact; there is no fuzzy fallback.

    Evidence is limited to :data:`MAX_HOPS` ownership hops and
    :data:`MAX_OWNERSHIP_RELATIONSHIPS` relationships, unordered paths are
    deduplicated, and each relationship keeps its stored direction
    (child supplier -> parent company).

    Unknown suppliers, invalid configuration, unavailable databases, timeouts,
    and integrity failures are returned as ``[GRAPH ERROR]`` strings rather than
    raised to the caller.
    """
    if not isinstance(entity, str) or not entity.strip():
        return "[GRAPH ERROR] supplier identity must not be empty"

    if store is not None:
        return _query_with_store(store, entity)

    try:
        settings = load_neo4j_settings()
    except ConfigError as error:
        return f"[GRAPH ERROR] graph configuration is invalid: {error}"

    try:
        with Neo4jGraphStore.from_settings(settings) as active_store:
            return _query_with_store(active_store, entity)
    except GraphError as error:
        return _error_context(error)


def _query_with_store(store: GraphQueryStore, entity: str) -> str:
    """Run one query and normalize every project error into a tagged string."""
    try:
        identity = store.resolve_supplier(entity)
        if identity is None:
            return (
                f"[GRAPH ERROR] unknown supplier identity {entity!r}; pass a supplier id "
                "such as SUP-007 or an exact supplier legal name"
            )
        rows = store.fetch_ownership_evidence(identity.supplier_id)
        return format_graph_context(build_evidence(identity, rows))
    except GraphError as error:
        return _error_context(error)


def _error_context(error: GraphError) -> str:
    if isinstance(error, UnknownGraphEntityError):
        return f"[GRAPH ERROR] {error}"
    if isinstance(error, GraphConfigError):
        return f"[GRAPH ERROR] graph configuration is invalid: {error}"
    if isinstance(error, GraphConnectionError):
        return f"[GRAPH ERROR] graph retrieval unavailable: {error}"
    if isinstance(error, GraphQueryError):
        return f"[GRAPH ERROR] graph query failed: {error}"
    if isinstance(error, GraphDataIntegrityError):
        return f"[GRAPH ERROR] graph data integrity failure: {error}"
    return f"[GRAPH ERROR] graph retrieval failed: {error}"


def format_graph_context(evidence: OwnershipEvidence) -> str:
    """Format evidence deterministically, preserving relationship direction."""
    supplier = evidence.supplier
    truncated = "true" if evidence.truncated else "false"
    lines = [
        (
            f"[GRAPH] supplier={supplier.supplier_id} name={supplier.legal_name} "
            f"partition={supplier.partition}"
        ),
        (
            "scope=ownership-neighborhood direction=child_supplier->parent_company "
            f"max_hops={MAX_HOPS} max_relationships={MAX_OWNERSHIP_RELATIONSHIPS} "
            f"truncated={truncated}"
        ),
        f"ownership_relationships={len(evidence.relationships)}",
    ]

    if not evidence.relationships:
        lines.append(
            f"No ownership or subsidiary relationships were found within {MAX_HOPS} hops."
        )
        return "\n".join(lines)

    for rank, row in enumerate(evidence.relationships, start=1):
        focus = "yes" if row.child_key == supplier.supplier_id else "no"
        lines.append(
            f"{rank}. hop={row.hops} focus={focus} "
            f'{row.child_key} "{row.child_name}" '
            f"-[{row.relationship_type} {row.ownership_percentage:.1f}%]-> "
            f'{row.parent_key} "{row.parent_name}"'
        )
    return "\n".join(lines)
