"""Tests for the Module 5 SQLite schema, load plan, and build."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from supplier_risk.contracts import SupplierDataset
from supplier_risk.sql.errors import SqlDataIntegrityError
from supplier_risk.sql.schema import (
    SQL_SCHEMA_VERSION,
    DeliveryMetricsRow,
    SqlPlan,
    SupplierRow,
    build_delivery_evidence,
    build_sql_plan,
    validate_sql_plan,
)
from supplier_risk.sql.store import build_database


def test_plan_covers_every_supplier_and_metric(module2_dataset: SupplierDataset) -> None:
    plan = build_sql_plan(module2_dataset)

    assert plan.schema_version == SQL_SCHEMA_VERSION
    assert plan.seed == module2_dataset.seed
    assert len(plan.suppliers) == 24
    assert len(plan.delivery_metrics) == 24
    assert plan.supplier_ids == tuple(sorted(plan.supplier_ids))
    assert {row.supplier_id for row in plan.delivery_metrics} == set(plan.supplier_ids)


def test_plan_is_deterministic(module2_dataset: SupplierDataset) -> None:
    assert build_sql_plan(module2_dataset) == build_sql_plan(module2_dataset)


def test_plan_rejects_a_duplicate_supplier(module2_dataset: SupplierDataset) -> None:
    plan = build_sql_plan(module2_dataset)
    duplicated = replace(plan, suppliers=plan.suppliers + (plan.suppliers[0],))

    with pytest.raises(SqlDataIntegrityError, match="duplicate supplier_id"):
        validate_sql_plan(duplicated)


def test_plan_rejects_metrics_for_an_unknown_supplier(
    module2_dataset: SupplierDataset,
) -> None:
    plan = build_sql_plan(module2_dataset)
    unknown = replace(plan.delivery_metrics[0], supplier_id="SUP-999")
    corrupted = replace(plan, delivery_metrics=(unknown,))

    with pytest.raises(SqlDataIntegrityError, match="unknown supplier"):
        validate_sql_plan(corrupted)


def test_plan_rejects_a_partition_that_contradicts_the_manifest(
    module2_dataset: SupplierDataset,
) -> None:
    plan = build_sql_plan(module2_dataset)
    swapped = replace(plan.suppliers[0], partition="dev")
    corrupted = replace(plan, suppliers=(swapped, *plan.suppliers[1:]))

    with pytest.raises(SqlDataIntegrityError, match="must be train"):
        validate_sql_plan(corrupted)


def test_plan_rejects_an_impossible_metric(module2_dataset: SupplierDataset) -> None:
    plan = build_sql_plan(module2_dataset)
    impossible = replace(plan.delivery_metrics[0], defect_rate=1.5)
    corrupted = replace(plan, delivery_metrics=(impossible,))

    with pytest.raises(SqlDataIntegrityError, match="defect_rate"):
        validate_sql_plan(corrupted)


def test_build_writes_the_expected_rows(
    module2_dataset: SupplierDataset, tmp_path: Path
) -> None:
    plan = build_sql_plan(module2_dataset)
    database = tmp_path / "suppliers.sqlite"

    summary = build_database(plan, database)

    assert summary.suppliers == 24
    assert summary.delivery_metrics == 24
    with sqlite3.connect(database) as connection:
        suppliers = connection.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]
        metrics = connection.execute("SELECT COUNT(*) FROM delivery_metrics").fetchone()[0]
        metadata = connection.execute(
            "SELECT schema_version, seed, supplier_count FROM dataset_metadata"
        ).fetchone()
        stored = connection.execute(
            "SELECT on_time_delivery_rate, defect_rate FROM delivery_metrics "
            "WHERE supplier_id = 'SUP-001'"
        ).fetchone()

    source = next(
        record for record in module2_dataset.suppliers if record.profile.supplier_id == "SUP-001"
    )
    assert (suppliers, metrics) == (24, 24)
    assert metadata == (SQL_SCHEMA_VERSION, module2_dataset.seed, 24)
    assert stored == (
        source.delivery_metrics.on_time_delivery_rate,
        source.delivery_metrics.defect_rate,
    )


def test_rebuilding_replaces_the_file_without_leaving_temporary_files(
    module2_dataset: SupplierDataset, tmp_path: Path
) -> None:
    plan = build_sql_plan(module2_dataset)
    database = tmp_path / "suppliers.sqlite"

    first = build_database(plan, database)
    second = build_database(plan, database)

    assert first == second
    assert not (tmp_path / ".suppliers.sqlite.tmp").exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0] == 24


def test_schema_checks_reject_an_invalid_metric(module2_dataset: SupplierDataset, tmp_path: Path) -> None:
    database = tmp_path / "suppliers.sqlite"
    build_database(build_sql_plan(module2_dataset), database)

    with sqlite3.connect(database) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO delivery_metrics (supplier_id, reporting_period, "
            "on_time_delivery_rate, defect_rate, average_lead_time_days, orders_fulfilled) "
            "VALUES ('SUP-001', 'FY2024', 0.9, 2.0, 10.0, 100)"
        )


def test_evidence_is_ordered_by_period_and_deduplicated() -> None:
    supplier = SupplierRow(
        supplier_id="SUP-001",
        legal_name="Norvend Components Ltd",
        partition="train",
        sector="electronics",
        country_code="GB",
    )
    newer = DeliveryMetricsRow("SUP-001", "FY2025", 0.95, 0.01, 12.0, 300)
    older = DeliveryMetricsRow("SUP-001", "FY2024", 0.90, 0.02, 14.0, 250)

    evidence = build_delivery_evidence(supplier, (newer, older, older))

    assert [row.reporting_period for row in evidence.metrics] == ["FY2024", "FY2025"]
    assert evidence.has_metrics is True


def test_evidence_rejects_rows_for_another_supplier() -> None:
    supplier = SupplierRow(
        supplier_id="SUP-001",
        legal_name="Norvend Components Ltd",
        partition="train",
        sector="electronics",
        country_code="GB",
    )
    other = DeliveryMetricsRow("SUP-002", "FY2025", 0.95, 0.01, 12.0, 300)

    with pytest.raises(SqlDataIntegrityError, match="returned for supplier"):
        build_delivery_evidence(supplier, (other,))


def test_evidence_rejects_conflicting_duplicate_periods() -> None:
    supplier = SupplierRow(
        supplier_id="SUP-001",
        legal_name="Norvend Components Ltd",
        partition="train",
        sector="electronics",
        country_code="GB",
    )
    first = DeliveryMetricsRow("SUP-001", "FY2025", 0.95, 0.01, 12.0, 300)
    conflicting = DeliveryMetricsRow("SUP-001", "FY2025", 0.80, 0.01, 12.0, 300)

    with pytest.raises(SqlDataIntegrityError, match="conflicting values"):
        build_delivery_evidence(supplier, (first, conflicting))


def test_empty_evidence_is_not_an_error() -> None:
    supplier = SupplierRow(
        supplier_id="SUP-001",
        legal_name="Norvend Components Ltd",
        partition="train",
        sector="electronics",
        country_code="GB",
    )

    evidence = build_delivery_evidence(supplier, ())

    assert evidence.metrics == ()
    assert evidence.has_metrics is False


def test_plan_type_round_trips_its_public_fields(module2_dataset: SupplierDataset) -> None:
    plan: SqlPlan = build_sql_plan(module2_dataset)

    assert plan.schema_version == "module5.v1"
    assert isinstance(plan.suppliers[0], SupplierRow)
    assert isinstance(plan.delivery_metrics[0], DeliveryMetricsRow)
