"""Project-level errors for the Module 3 retrieval stack.

Every failure that callers are expected to handle is expressed as one of these
types. Messages are actionable and never include credentials or raw record
dumps. Unexpected programming errors are intentionally left uncaught.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base class for retrieval build, persistence, and query failures."""


class CorpusError(RetrievalError, ValueError):
    """Raised when the document corpus cannot be built or validated."""


class RetrievalArtifactError(RetrievalError):
    """Raised when persisted retrieval artifacts are missing, stale, or corrupt."""


class RetrievalQueryError(RetrievalError, ValueError):
    """Raised when a query, ``top_k``, or metadata filter is invalid."""


class EmbeddingModelError(RetrievalError, RuntimeError):
    """Raised when the local embedding model cannot be loaded or returns bad vectors."""


class UnknownSupplierError(RetrievalError, LookupError):
    """Raised when a supplier-facing query names an unknown supplier identity."""
