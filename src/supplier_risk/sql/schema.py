"""SQLite schema, deterministic load plan, and evidence shaping for Module 5.

Only structured supplier facts live in SQLite: supplier identity, its partition,
and one annual delivery/defect observation per supplier. Contract and complaint
text stays in the Module 3 retrieval artifacts and ownership edges stay in the
Module 4 Neo4j graph, so this store never duplicates another source's evidence.

The physical schema follows ``docs/schema.md`` section 8 for the tables this
store actually needs. Rates are stored as fractions (``0.95``, not ``95``), dates
are not needed by this subset, and no derived risk field is stored.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from supplier_risk.contracts import Partition, SupplierDataset, partition_for_supplier_id
from supplier_risk.sql.errors import SqlDataIntegrityError

SQL_SCHEMA_VERSION: Final = "module5.v1"

_SUPPLIER_ID_PATTERN: Final = re.compile(r"^SUP-\d{3}$")
_REPORTING_PERIOD_PATTERN: Final = re.compile(r"^FY\d{4}$")
_PARTITIONS: Final = frozenset({"train", "dev", "held_out"})

MAX_ORDERS_FULFILLED: Final = 10_000_000

SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS dataset_metadata (
        singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
        schema_version TEXT NOT NULL,
        seed INTEGER NOT NULL CHECK (seed BETWEEN 0 AND 4294967295),
        supplier_count INTEGER NOT NULL CHECK (supplier_count = 24)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS suppliers (
        supplier_id TEXT PRIMARY KEY,
        legal_name TEXT NOT NULL,
        country_code TEXT NOT NULL CHECK (length(country_code) = 2),
        sector TEXT NOT NULL,
        partition TEXT NOT NULL CHECK (partition IN ('train', 'dev', 'held_out'))
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_suppliers_legal_name
        ON suppliers (legal_name COLLATE NOCASE)
    """,
    """
    CREATE TABLE IF NOT EXISTS delivery_metrics (
        supplier_id TEXT NOT NULL REFERENCES suppliers(supplier_id) ON DELETE RESTRICT,
        reporting_period TEXT NOT NULL,
        on_time_delivery_rate REAL NOT NULL
            CHECK (on_time_delivery_rate BETWEEN 0.0 AND 1.0),
        defect_rate REAL NOT NULL CHECK (defect_rate BETWEEN 0.0 AND 1.0),
        average_lead_time_days REAL NOT NULL
            CHECK (average_lead_time_days > 0.0 AND average_lead_time_days <= 365.0),
        orders_fulfilled INTEGER NOT NULL
            CHECK (orders_fulfilled BETWEEN 0 AND 10000000),
        PRIMARY KEY (supplier_id, reporting_period)
    )
    """,
)


@dataclass(frozen=True, slots=True)
class SupplierRow:
    """One supplier identity row, using only validated profile fields."""

    supplier_id: str
    legal_name: str
    partition: Partition
    sector: str
    country_code: str


@dataclass(frozen=True, slots=True)
class DeliveryMetricsRow:
    """One annual delivery/defect observation for one supplier."""

    supplier_id: str
    reporting_period: str
    on_time_delivery_rate: float
    defect_rate: float
    average_lead_time_days: float
    orders_fulfilled: int


@dataclass(frozen=True, slots=True)
class SqlPlan:
    """Everything one build writes to SQLite, in deterministic order."""

    schema_version: str
    seed: int
    suppliers: tuple[SupplierRow, ...]
    delivery_metrics: tuple[DeliveryMetricsRow, ...]

    @property
    def supplier_ids(self) -> tuple[str, ...]:
        return tuple(row.supplier_id for row in self.suppliers)


@dataclass(frozen=True, slots=True)
class DeliveryEvidence:
    """Deterministic, validated delivery evidence for one supplier."""

    supplier: SupplierRow
    metrics: tuple[DeliveryMetricsRow, ...]

    @property
    def has_metrics(self) -> bool:
        return bool(self.metrics)


def build_sql_plan(dataset: SupplierDataset) -> SqlPlan:
    """Build and validate the deterministic SQLite load plan for ``dataset``.

    Raises:
        SqlDataIntegrityError: if supplier identity or delivery metrics are
            malformed, duplicated, or inconsistent with the manifest.
    """
    ordered = sorted(dataset.suppliers, key=lambda item: item.profile.supplier_id)
    suppliers = tuple(
        SupplierRow(
            supplier_id=record.profile.supplier_id,
            legal_name=record.profile.legal_name,
            partition=record.partition,
            sector=record.profile.sector,
            country_code=record.profile.country_code,
        )
        for record in ordered
    )
    metrics = tuple(
        DeliveryMetricsRow(
            supplier_id=record.delivery_metrics.supplier_id,
            reporting_period=record.delivery_metrics.reporting_period,
            on_time_delivery_rate=record.delivery_metrics.on_time_delivery_rate,
            defect_rate=record.delivery_metrics.defect_rate,
            average_lead_time_days=record.delivery_metrics.average_lead_time_days,
            orders_fulfilled=record.delivery_metrics.orders_fulfilled,
        )
        for record in ordered
    )
    plan = SqlPlan(
        schema_version=SQL_SCHEMA_VERSION,
        seed=dataset.seed,
        suppliers=suppliers,
        delivery_metrics=metrics,
    )
    validate_sql_plan(plan)
    return plan


def validate_sql_plan(plan: SqlPlan) -> None:
    """Validate supplier keys, partitions, and delivery metric ranges.

    The plan rules are checked again at load time on purpose: ingestion must not
    depend on a dataset having been validated somewhere else, and an invalid row
    must never become plausible-looking evidence.

    Raises:
        SqlDataIntegrityError: with every problem found, joined into one message.
    """
    issues: list[str] = []

    if not plan.suppliers:
        issues.append("SQL plan contains no suppliers")
    if plan.schema_version != SQL_SCHEMA_VERSION:
        issues.append(f"unsupported SQL schema version: {plan.schema_version!r}")

    supplier_ids: set[str] = set()
    legal_names: set[str] = set()
    for supplier in plan.suppliers:
        if _SUPPLIER_ID_PATTERN.fullmatch(supplier.supplier_id) is None:
            issues.append(f"invalid supplier_id: {supplier.supplier_id!r}")
        if supplier.supplier_id in supplier_ids:
            issues.append(f"duplicate supplier_id: {supplier.supplier_id}")
        supplier_ids.add(supplier.supplier_id)

        if not supplier.legal_name.strip():
            issues.append(f"blank legal_name for {supplier.supplier_id}")
        name_key = supplier.legal_name.casefold()
        if name_key in legal_names:
            issues.append(f"duplicate supplier legal_name for {supplier.supplier_id}")
        legal_names.add(name_key)

        if supplier.partition not in _PARTITIONS:
            issues.append(
                f"invalid partition for {supplier.supplier_id}: {supplier.partition!r}"
            )
            continue
        try:
            expected_partition = partition_for_supplier_id(supplier.supplier_id)
        except ValueError:
            continue
        if supplier.partition != expected_partition:
            issues.append(
                f"partition for {supplier.supplier_id} must be {expected_partition}; "
                f"found {supplier.partition}"
            )

    seen_periods: set[tuple[str, str]] = set()
    for row in plan.delivery_metrics:
        if row.supplier_id not in supplier_ids:
            issues.append(f"delivery metrics reference an unknown supplier: {row.supplier_id}")
        key = (row.supplier_id, row.reporting_period)
        if key in seen_periods:
            issues.append(
                f"duplicate delivery metrics for {row.supplier_id} {row.reporting_period}"
            )
        seen_periods.add(key)

        if _REPORTING_PERIOD_PATTERN.fullmatch(row.reporting_period) is None:
            issues.append(
                f"invalid reporting_period for {row.supplier_id}: {row.reporting_period!r}"
            )
        if not _is_number(row.on_time_delivery_rate) or not (
            0.0 <= row.on_time_delivery_rate <= 1.0
        ):
            issues.append(f"on_time_delivery_rate for {row.supplier_id} must be in [0, 1]")
        if not _is_number(row.defect_rate) or not (0.0 <= row.defect_rate <= 1.0):
            issues.append(f"defect_rate for {row.supplier_id} must be in [0, 1]")
        if not _is_number(row.average_lead_time_days) or not (
            0.0 < row.average_lead_time_days <= 365.0
        ):
            issues.append(
                f"average_lead_time_days for {row.supplier_id} must be in (0, 365]"
            )
        if (
            isinstance(row.orders_fulfilled, bool)
            or not isinstance(row.orders_fulfilled, int)
            or not 0 <= row.orders_fulfilled <= MAX_ORDERS_FULFILLED
        ):
            issues.append(
                f"orders_fulfilled for {row.supplier_id} must be an integer in "
                f"[0, {MAX_ORDERS_FULFILLED}]"
            )

    if issues:
        raise SqlDataIntegrityError("; ".join(issues))


def build_delivery_evidence(
    supplier: SupplierRow,
    rows: Sequence[DeliveryMetricsRow],
) -> DeliveryEvidence:
    """Validate, deduplicate, and order stored delivery rows for one supplier.

    Duplicate ``(supplier, period)`` rows are a primary-key violation in SQLite,
    so seeing one here means the database was modified outside this project;
    conflicting values are rejected rather than silently merged.

    Raises:
        SqlDataIntegrityError: if a row belongs to another supplier, repeats a
            period with different values, or holds an impossible observation.
    """
    merged: dict[str, DeliveryMetricsRow] = {}
    for row in rows:
        if row.supplier_id != supplier.supplier_id:
            raise SqlDataIntegrityError(
                f"delivery metrics for {row.supplier_id} were returned for "
                f"supplier {supplier.supplier_id}"
            )
        _validate_metrics_row(row)
        existing = merged.get(row.reporting_period)
        if existing is None:
            merged[row.reporting_period] = row
            continue
        if existing != row:
            raise SqlDataIntegrityError(
                f"delivery metrics for {row.supplier_id} {row.reporting_period} "
                "report conflicting values"
            )
    return DeliveryEvidence(
        supplier=supplier,
        metrics=tuple(merged[period] for period in sorted(merged)),
    )


def _validate_metrics_row(row: DeliveryMetricsRow) -> None:
    if _REPORTING_PERIOD_PATTERN.fullmatch(row.reporting_period) is None:
        raise SqlDataIntegrityError(
            f"delivery metrics for {row.supplier_id} have an invalid reporting period"
        )
    if not _is_number(row.on_time_delivery_rate) or not (
        0.0 <= row.on_time_delivery_rate <= 1.0
    ):
        raise SqlDataIntegrityError(
            f"delivery metrics for {row.supplier_id} have an invalid on_time_delivery_rate"
        )
    if not _is_number(row.defect_rate) or not (0.0 <= row.defect_rate <= 1.0):
        raise SqlDataIntegrityError(
            f"delivery metrics for {row.supplier_id} have an invalid defect_rate"
        )
    if not _is_number(row.average_lead_time_days) or not (
        0.0 < row.average_lead_time_days <= 365.0
    ):
        raise SqlDataIntegrityError(
            f"delivery metrics for {row.supplier_id} have an invalid average_lead_time_days"
        )
    if (
        isinstance(row.orders_fulfilled, bool)
        or not isinstance(row.orders_fulfilled, int)
        or not 0 <= row.orders_fulfilled <= MAX_ORDERS_FULFILLED
    ):
        raise SqlDataIntegrityError(
            f"delivery metrics for {row.supplier_id} have an invalid orders_fulfilled"
        )


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
