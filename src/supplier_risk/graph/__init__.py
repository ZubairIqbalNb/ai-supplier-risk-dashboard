"""Module 4 supplier ownership graph: Neo4j persistence and retrieval.

Importing this package creates no driver and opens no connection. The
assignment-facing entry point is :func:`query_graph`; ingestion is an explicit
command (``python -m supplier_risk.graph.ingest``).
"""

from supplier_risk.graph.errors import (
    GraphConfigError,
    GraphConnectionError,
    GraphDataIntegrityError,
    GraphError,
    GraphQueryError,
    UnknownGraphEntityError,
)
from supplier_risk.graph.pipeline import query_graph
from supplier_risk.graph.schema import (
    MAX_HOPS,
    MAX_OWNERSHIP_RELATIONSHIPS,
    CompanyRow,
    GraphPlan,
    OwnershipEvidence,
    OwnershipEvidenceRow,
    OwnershipRow,
    SupplierIdentity,
    SupplierRow,
    build_evidence,
    build_graph_plan,
)
from supplier_risk.graph.store import GraphCounts, Neo4jGraphStore, WriteStats

__all__ = [
    "MAX_HOPS",
    "MAX_OWNERSHIP_RELATIONSHIPS",
    "CompanyRow",
    "GraphConfigError",
    "GraphConnectionError",
    "GraphCounts",
    "GraphDataIntegrityError",
    "GraphError",
    "GraphPlan",
    "GraphQueryError",
    "Neo4jGraphStore",
    "OwnershipEvidence",
    "OwnershipEvidenceRow",
    "OwnershipRow",
    "SupplierIdentity",
    "SupplierRow",
    "UnknownGraphEntityError",
    "WriteStats",
    "build_evidence",
    "build_graph_plan",
    "query_graph",
]
