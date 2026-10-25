"""Module 5 parallel multi-source retrieval orchestration.

Importing this package imports the three engine wrappers but opens no database,
creates no driver, and loads no embedding model. The assignment-facing entry
point is :func:`query_all_sources`, which runs ``query_sql``, ``query_faiss``,
and ``query_graph`` concurrently and returns one provenance-preserving result.
"""

from supplier_risk.orchestration.pipeline import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_SOURCES,
    OrchestrationError,
    SourceCallable,
    query_all_sources,
)
from supplier_risk.orchestration.result import (
    SOURCE_LABELS,
    SOURCE_ORDER,
    OrchestrationResult,
    Outcome,
    SourceEvidence,
    SourceName,
    SourceStatus,
)

__all__ = [
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_SOURCES",
    "SOURCE_LABELS",
    "SOURCE_ORDER",
    "OrchestrationError",
    "OrchestrationResult",
    "Outcome",
    "SourceCallable",
    "SourceEvidence",
    "SourceName",
    "SourceStatus",
    "query_all_sources",
]
