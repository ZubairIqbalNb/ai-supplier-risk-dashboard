"""Tests for the Module 5 ``query_sql(entity)`` wrapper and SQLite store."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import supplier_risk.sql.store as store_module
from supplier_risk.contracts import SupplierDataset, SupplierRecord
from supplier_risk.data.generate import build_dataset, write_dataset
from supplier_risk.sql.build import build_from_dataset
from supplier_risk.sql.errors import (
    SqlConfigError,
    SqlDataIntegrityError,
    SqlError,
    SqlQueryError,
    SqlStoreUnavailableError,
    UnknownSqlEntityError,
)
from supplier_risk.sql.pipeline import query_sql
from supplier_risk.sql.schema import DeliveryMetricsRow, SupplierRow


@pytest.fixture(scope="module")
def sql_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("sql") / "suppliers.sqlite"
    return build_from_dataset(build_dataset(), path).database_path


def _record(module2_dataset: SupplierDataset, supplier_id: str) -> SupplierRecord:
    return next(
        record
        for record in module2_dataset.suppliers
        if record.profile.supplier_id == supplier_id
    )


def test_query_sql_returns_labeled_delivery_metrics(
    sql_database: Path, module2_dataset: SupplierDataset
) -> None:
    record = _record(module2_dataset, "SUP-001")
    metrics = record.delivery_metrics

    result = query_sql("SUP-001", database_path=sql_database)

    assert result.startswith(
        "[SQL] supplier=SUP-001 name=Norvend Components Ltd partition=train"
    )
    assert "scope=delivery+defect-metrics source=sqlite table=delivery_metrics" in result
    assert "delivery_metrics=1" in result
    assert f"period={metrics.reporting_period}" in result
    assert f"on_time_delivery_rate={metrics.on_time_delivery_rate}" in result
    assert f"defect_rate={metrics.defect_rate}" in result
    assert f"average_lead_time_days={metrics.average_lead_time_days}" in result
    assert f"orders_fulfilled={metrics.orders_fulfilled}" in result
    assert "ownership" not in result.lower()
    assert "document=" not in result


def test_query_sql_accepts_ids_and_exact_legal_names(sql_database: Path) -> None:
    by_id = query_sql("SUP-007", database_path=sql_database)
    by_lowercase = query_sql("sup-007", database_path=sql_database)
    by_name = query_sql("Sakurajima Precision KK", database_path=sql_database)
    by_lowercase_name = query_sql("sakurajima precision kk", database_path=sql_database)

    assert by_id == by_lowercase == by_name == by_lowercase_name
    assert by_id.startswith("[SQL] supplier=SUP-007")


def test_query_sql_is_stable_across_repeated_calls(sql_database: Path) -> None:
    assert query_sql("SUP-012", database_path=sql_database) == query_sql(
        "SUP-012", database_path=sql_database
    )


def test_unknown_supplier_is_tagged(sql_database: Path) -> None:
    result = query_sql("SUP-999", database_path=sql_database)

    assert result.startswith("[SQL ERROR] unknown supplier identity 'SUP-999'")
    assert "exact supplier legal name" in result


@pytest.mark.parametrize("entity", ["", "   ", None])
def test_empty_identity_is_tagged_without_opening_a_database(
    entity: object, tmp_path: Path
) -> None:
    missing = tmp_path / "never-created.sqlite"

    result = query_sql(entity, database_path=missing)  # type: ignore[arg-type]

    assert result == "[SQL ERROR] supplier identity must not be empty"
    assert not missing.exists()


def test_missing_database_is_tagged_with_a_build_hint(tmp_path: Path) -> None:
    result = query_sql("SUP-001", database_path=tmp_path / "absent.sqlite")

    assert result.startswith("[SQL ERROR] SQL retrieval unavailable")
    assert "uv run --locked python -m supplier_risk.sql.build" in result


def test_corrupt_database_is_tagged_not_raised(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_bytes(b"this is not a sqlite database")

    result = query_sql("SUP-001", database_path=corrupt)

    assert result.startswith("[SQL ERROR] SQL query failed")
    assert "SQLite" in result


def test_database_without_the_expected_schema_is_tagged(
    module2_dataset: SupplierDataset, tmp_path: Path
) -> None:
    database = tmp_path / "empty.sqlite"
    with sqlite3.connect(database):
        pass

    result = query_sql("SUP-001", database_path=database)

    assert result.startswith("[SQL ERROR] SQL query failed")


def test_valid_supplier_without_metrics_is_no_evidence_not_an_error(
    module2_dataset: SupplierDataset, tmp_path: Path
) -> None:
    database = tmp_path / "no-metrics.sqlite"
    build_from_dataset(module2_dataset, database)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM delivery_metrics WHERE supplier_id = 'SUP-002'")

    result = query_sql("SUP-002", database_path=database)

    assert result.startswith("[SQL] supplier=SUP-002")
    assert "delivery_metrics=0" in result
    assert "No delivery or defect metrics were found for this supplier." in result
    assert "[SQL ERROR]" not in result


class FakeStore:
    """Query-store double that raises a chosen project error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def resolve_supplier(self, entity: str) -> SupplierRow | None:
        raise self.error

    def fetch_delivery_metrics(self, supplier_id: str) -> tuple[DeliveryMetricsRow, ...]:
        raise AssertionError("must not be called when resolution fails")


@pytest.mark.parametrize(
    ("error", "expected_prefix"),
    [
        (
            UnknownSqlEntityError("supplier identity 'SUP-999' is ambiguous"),
            "[SQL ERROR] supplier identity 'SUP-999' is ambiguous",
        ),
        (SqlConfigError("SQLite busy timeout must be positive"), "[SQL ERROR] SQL configuration"),
        (SqlStoreUnavailableError("database is locked"), "[SQL ERROR] SQL retrieval unavailable"),
        (SqlQueryError("statement failed"), "[SQL ERROR] SQL query failed"),
        (SqlDataIntegrityError("row is malformed"), "[SQL ERROR] SQL data integrity failure"),
        (SqlError("something else"), "[SQL ERROR] SQL retrieval failed"),
    ],
)
def test_project_errors_are_tagged_by_category(
    error: Exception, expected_prefix: str
) -> None:
    result = query_sql("SUP-001", store=FakeStore(error))

    assert result.startswith(expected_prefix)


def test_store_rejects_a_non_positive_busy_timeout() -> None:
    with pytest.raises(SqlConfigError, match="busy timeout"):
        store_module.SqliteStore(busy_timeout=0)


def test_ambiguous_stored_identity_is_tagged(tmp_path: Path) -> None:
    database = tmp_path / "ambiguous.sqlite"
    build_from_dataset(build_dataset(), database)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX ux_suppliers_legal_name")
        connection.execute(
            "INSERT INTO suppliers (supplier_id, legal_name, country_code, sector, partition) "
            "VALUES ('SUP-900', 'Norvend Components Ltd', 'GB', 'electronics', 'train')"
        )

    result = query_sql("Norvend Components Ltd", database_path=database)

    assert result == "[SQL ERROR] supplier identity 'Norvend Components Ltd' is ambiguous"


def test_each_query_closes_its_connection(
    sql_database: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def recording_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        captured.append(connection)
        return connection

    monkeypatch.setattr(store_module.sqlite3, "connect", recording_connect)

    query_sql("SUP-001", database_path=sql_database)

    assert len(captured) >= 1
    for connection in captured:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_concurrent_queries_match_sequential_results(
    sql_database: Path, module2_dataset: SupplierDataset
) -> None:
    supplier_ids = [
        record.profile.supplier_id for record in module2_dataset.suppliers
    ]
    expected = {
        supplier_id: query_sql(supplier_id, database_path=sql_database)
        for supplier_id in supplier_ids
    }

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = dict(
            zip(
                supplier_ids,
                executor.map(
                    lambda supplier_id: query_sql(supplier_id, database_path=sql_database),
                    supplier_ids,
                ),
                strict=True,
            )
        )

    assert results == expected


def test_build_cli_reports_success(
    module2_dataset: SupplierDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from supplier_risk.sql.build import main

    dataset_path = tmp_path / "suppliers.json"
    write_dataset(module2_dataset, dataset_path)
    database = tmp_path / "built.sqlite"

    exit_code = main(["--dataset", str(dataset_path), "--database", str(database)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Suppliers: 24" in output
    assert "Delivery metrics rows: 24" in output
    assert database.is_file()


def test_build_cli_reports_a_missing_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from supplier_risk.sql.build import main

    exit_code = main(
        ["--dataset", str(tmp_path / "absent.json"), "--database", str(tmp_path / "db.sqlite")]
    )

    assert exit_code == 1
    assert "supplier dataset not found" in capsys.readouterr().err
