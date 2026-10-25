"""Typed errors for the Module 5 SQLite structured-retrieval stack.

The store raises project errors rather than leaking ``sqlite3`` exceptions, and
the assignment-facing wrapper converts them into tagged ``[SQL ERROR]`` strings.
Messages name variables, operations, and paths only; they never include record
payloads or credentials.
"""

from __future__ import annotations


class SqlError(Exception):
    """Base class for SQL configuration, availability, query, and data errors."""


class SqlConfigError(SqlError):
    """Raised when the SQLite store or orchestrator settings are invalid."""


class SqlStoreUnavailableError(SqlError):
    """Raised when the SQLite database file is missing, unreadable, or unusable."""


class SqlQueryError(SqlError):
    """Raised when SQLite rejects a statement or a query cannot complete."""


class SqlDataIntegrityError(SqlError):
    """Raised when a load plan or a stored row contradicts the dataset contract."""


class UnknownSqlEntityError(SqlError):
    """Raised when a supplier identity is unknown or ambiguous in the SQL store."""
