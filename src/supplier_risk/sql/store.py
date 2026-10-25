"""SQLite persistence for the Module 5 structured supplier store.

Every operation opens its own short-lived connection, so concurrent retrieval
threads never share a SQLite connection (connections must not be used across
threads). Queries open the database read-only; only :func:`build_database`
writes, and it replaces the database file atomically, mirroring the Module 3
artifact convention.

The database file is Git-ignored and is created explicitly::

    uv run --locked python -m supplier_risk.sql.build

Querying never creates or rebuilds the store. A missing file is reported as a
tagged error by :func:`supplier_risk.sql.pipeline.query_sql`.
"""

from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import quote

from supplier_risk.contracts import Partition
from supplier_risk.sql.errors import (
    SqlConfigError,
    SqlDataIntegrityError,
    SqlError,
    SqlQueryError,
    SqlStoreUnavailableError,
    UnknownSqlEntityError,
)
from supplier_risk.sql.schema import (
    SCHEMA_STATEMENTS,
    DeliveryMetricsRow,
    SqlPlan,
    SupplierRow,
    validate_sql_plan,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH: Final = PROJECT_ROOT / "data" / "generated" / "suppliers.sqlite"
BUILD_COMMAND: Final = "uv run --locked python -m supplier_risk.sql.build"
DEFAULT_BUSY_TIMEOUT_SECONDS: Final = 5.0

_PARTITIONS: Final = frozenset({"train", "dev", "held_out"})
_MAX_ORDERS_FULFILLED: Final = 10_000_000

_RESOLVE_SUPPLIER_QUERY: Final = """
SELECT supplier_id, legal_name, country_code, sector, partition
FROM suppliers
WHERE supplier_id COLLATE NOCASE = :entity
   OR legal_name COLLATE NOCASE = :entity
ORDER BY CASE WHEN supplier_id COLLATE NOCASE = :entity THEN 0 ELSE 1 END,
         supplier_id
LIMIT 2
"""

_DELIVERY_METRICS_QUERY: Final = """
SELECT supplier_id, reporting_period, on_time_delivery_rate, defect_rate,
       average_lead_time_days, orders_fulfilled
FROM delivery_metrics
WHERE supplier_id = :supplier_id
ORDER BY reporting_period
"""

_INSERT_SUPPLIER_QUERY: Final = """
INSERT INTO suppliers (supplier_id, legal_name, country_code, sector, partition)
VALUES (?, ?, ?, ?, ?)
"""

_INSERT_DELIVERY_METRICS_QUERY: Final = """
INSERT INTO delivery_metrics (
    supplier_id, reporting_period, on_time_delivery_rate, defect_rate,
    average_lead_time_days, orders_fulfilled
)
VALUES (?, ?, ?, ?, ?, ?)
"""

_INSERT_DATASET_METADATA_QUERY: Final = """
INSERT INTO dataset_metadata (singleton_id, schema_version, seed, supplier_count)
VALUES (1, ?, ?, ?)
"""


@dataclass(frozen=True, slots=True)
class SqlBuildSummary:
    """Counts verified after one build transaction committed."""

    database_path: Path
    suppliers: int
    delivery_metrics: int


class SqliteStore:
    """Thread-safe owner of one SQLite database path.

    The store holds no connection and no mutable state, so one instance can be
    shared by the orchestrator's worker threads; each call opens and closes its
    own read-only connection.
    """

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT_SECONDS,
    ) -> None:
        if busy_timeout <= 0:
            raise SqlConfigError("SQLite busy timeout must be positive")
        self._database_path = Path(database_path)
        self._busy_timeout = busy_timeout

    @property
    def database_path(self) -> Path:
        return self._database_path

    def resolve_supplier(self, entity: str) -> SupplierRow | None:
        """Resolve a supplier id or exact case-insensitive legal name.

        Returns ``None`` when no row matches and raises
        :class:`UnknownSqlEntityError` when more than one row matches.
        """
        text = entity.strip() if isinstance(entity, str) else ""
        if not text:
            raise UnknownSqlEntityError("supplier identity must not be empty")

        rows = self._select(
            "resolving the supplier", _RESOLVE_SUPPLIER_QUERY, {"entity": text}
        )
        if not rows:
            return None
        if len(rows) > 1:
            raise UnknownSqlEntityError(f"supplier identity {text!r} is ambiguous")
        return _supplier_row_from_record(rows[0])

    def fetch_delivery_metrics(self, supplier_id: str) -> tuple[DeliveryMetricsRow, ...]:
        """Return the stored annual delivery/defect rows for one supplier."""
        rows = self._select(
            "reading delivery metrics",
            _DELIVERY_METRICS_QUERY,
            {"supplier_id": supplier_id},
        )
        return tuple(_metrics_row_from_record(row) for row in rows)

    def _select(
        self, operation: str, query: str, parameters: dict[str, Any]
    ) -> list[sqlite3.Row]:
        try:
            with self._connection() as connection:
                return list(connection.execute(query, parameters))
        except SqlError:
            raise
        except sqlite3.Error as error:  # database/schema boundary
            raise SqlQueryError(
                f"SQLite rejected the request while {operation} ({type(error).__name__})."
            ) from None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a read-only connection for one operation and always close it."""
        path = self._database_path
        if not path.is_file():
            raise SqlStoreUnavailableError(
                f"SQLite store not found at {path}; build it with `{BUILD_COMMAND}`."
            )

        uri = f"file:{quote(path.as_posix(), safe='/')}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=self._busy_timeout)
        except sqlite3.Error as error:
            raise SqlStoreUnavailableError(
                f"the SQLite store at {path} cannot be opened "
                f"({type(error).__name__}); rebuild it with `{BUILD_COMMAND}`."
            ) from None

        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
        finally:
            connection.close()


def build_database(
    plan: SqlPlan,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    busy_timeout: float = DEFAULT_BUSY_TIMEOUT_SECONDS,
) -> SqlBuildSummary:
    """Write ``plan`` to a fresh SQLite database, replacing any existing file.

    The plan is validated first, every statement runs in one transaction, and
    the row counts are verified before the transaction commits. The finished
    file is then moved into place with :func:`os.replace`, so a failed build
    cannot leave a half-populated database at the final path.

    Raises:
        SqlDataIntegrityError: if the plan or the written row counts are invalid.
        SqlConfigError: if the destination path is not a usable file path.
        SqlStoreUnavailableError: if the file cannot be created or replaced.
    """
    validate_sql_plan(plan)
    target = Path(database_path)
    if target.exists() and not target.is_file():
        raise SqlConfigError(f"SQLite destination is not a regular file: {target}")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SqlStoreUnavailableError(
            f"the SQLite destination directory for {target} cannot be created"
        ) from None

    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)

    try:
        connection = sqlite3.connect(temporary, isolation_level=None, timeout=busy_timeout)
    except sqlite3.Error as error:
        raise SqlStoreUnavailableError(
            f"the SQLite store at {target} cannot be created ({type(error).__name__})"
        ) from None

    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                _INSERT_DATASET_METADATA_QUERY,
                (plan.schema_version, plan.seed, len(plan.suppliers)),
            )
            connection.executemany(
                _INSERT_SUPPLIER_QUERY,
                [
                    (
                        row.supplier_id,
                        row.legal_name,
                        row.country_code,
                        row.sector,
                        row.partition,
                    )
                    for row in plan.suppliers
                ],
            )
            connection.executemany(
                _INSERT_DELIVERY_METRICS_QUERY,
                [
                    (
                        row.supplier_id,
                        row.reporting_period,
                        row.on_time_delivery_rate,
                        row.defect_rate,
                        row.average_lead_time_days,
                        row.orders_fulfilled,
                    )
                    for row in plan.delivery_metrics
                ],
            )
            suppliers = _count_rows(connection, "suppliers")
            delivery_metrics = _count_rows(connection, "delivery_metrics")
            if (suppliers, delivery_metrics) != (
                len(plan.suppliers),
                len(plan.delivery_metrics),
            ):
                raise SqlDataIntegrityError(
                    "row counts after loading do not match the validated plan"
                )
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:  # best-effort rollback of a failed build
                pass
            raise
    except sqlite3.Error as error:  # database/schema boundary
        raise SqlDataIntegrityError(
            f"SQLite rejected the load plan ({type(error).__name__}); "
            "the transaction was rolled back."
        ) from None
    finally:
        connection.close()

    try:
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise SqlStoreUnavailableError(
            f"the SQLite store at {target} could not be moved into place"
        ) from None

    return SqlBuildSummary(
        database_path=target,
        suppliers=suppliers,
        delivery_metrics=delivery_metrics,
    )


def _count_rows(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _supplier_row_from_record(record: sqlite3.Row) -> SupplierRow:
    partition = record["partition"]
    if partition not in _PARTITIONS:
        raise SqlDataIntegrityError(
            f"stored supplier {record['supplier_id']} has an invalid partition"
        )
    return SupplierRow(
        supplier_id=str(record["supplier_id"]),
        legal_name=str(record["legal_name"]),
        partition=cast(Partition, partition),
        sector=str(record["sector"]),
        country_code=str(record["country_code"]),
    )


def _metrics_row_from_record(record: sqlite3.Row) -> DeliveryMetricsRow:
    on_time = record["on_time_delivery_rate"]
    defect = record["defect_rate"]
    lead_time = record["average_lead_time_days"]
    orders = record["orders_fulfilled"]

    for label, value in (("on_time_delivery_rate", on_time), ("defect_rate", defect)):
        if not _is_number(value) or not 0.0 <= float(value) <= 1.0:
            raise SqlDataIntegrityError(
                f"stored delivery metrics for {record['supplier_id']} have an "
                f"invalid {label}"
            )
    if not _is_number(lead_time) or not 0.0 < float(lead_time) <= 365.0:
        raise SqlDataIntegrityError(
            f"stored delivery metrics for {record['supplier_id']} have an "
            "invalid average_lead_time_days"
        )
    if (
        isinstance(orders, bool)
        or not isinstance(orders, int)
        or not 0 <= orders <= _MAX_ORDERS_FULFILLED
    ):
        raise SqlDataIntegrityError(
            f"stored delivery metrics for {record['supplier_id']} have an "
            "invalid orders_fulfilled"
        )

    return DeliveryMetricsRow(
        supplier_id=str(record["supplier_id"]),
        reporting_period=str(record["reporting_period"]),
        on_time_delivery_rate=float(on_time),
        defect_rate=float(defect),
        average_lead_time_days=float(lead_time),
        orders_fulfilled=orders,
    )


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
