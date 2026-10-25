"""Module 5 SQLite structured retrieval: delivery and defect metrics.

Importing this package opens no database and creates no file. The
assignment-facing entry point is :func:`query_sql`; building the store is an
explicit command (``python -m supplier_risk.sql.build``), and the database file
under ``data/generated/`` is Git-ignored.
"""

from supplier_risk.sql.errors import (
    SqlConfigError,
    SqlDataIntegrityError,
    SqlError,
    SqlQueryError,
    SqlStoreUnavailableError,
    UnknownSqlEntityError,
)
from supplier_risk.sql.pipeline import format_sql_context, query_sql
from supplier_risk.sql.schema import (
    SCHEMA_STATEMENTS,
    SQL_SCHEMA_VERSION,
    DeliveryEvidence,
    DeliveryMetricsRow,
    SqlPlan,
    SupplierRow,
    build_delivery_evidence,
    build_sql_plan,
    validate_sql_plan,
)
from supplier_risk.sql.store import (
    BUILD_COMMAND,
    DEFAULT_DATABASE_PATH,
    SqlBuildSummary,
    SqliteStore,
    build_database,
)

__all__ = [
    "BUILD_COMMAND",
    "DEFAULT_DATABASE_PATH",
    "SCHEMA_STATEMENTS",
    "SQL_SCHEMA_VERSION",
    "DeliveryEvidence",
    "DeliveryMetricsRow",
    "SqlBuildSummary",
    "SqlConfigError",
    "SqlDataIntegrityError",
    "SqlError",
    "SqlPlan",
    "SqlQueryError",
    "SqlStoreUnavailableError",
    "SqliteStore",
    "SupplierRow",
    "UnknownSqlEntityError",
    "build_database",
    "build_delivery_evidence",
    "build_sql_plan",
    "format_sql_context",
    "query_sql",
    "validate_sql_plan",
]
