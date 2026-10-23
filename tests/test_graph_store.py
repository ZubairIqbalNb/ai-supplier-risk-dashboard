from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Self

import pytest
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from pydantic import SecretStr

import supplier_risk.graph.store as store_module
from supplier_risk.config import Neo4jSettings
from supplier_risk.graph.errors import (
    GraphConfigError,
    GraphConnectionError,
    GraphDataIntegrityError,
    GraphQueryError,
    UnknownGraphEntityError,
)
from supplier_risk.graph.schema import CompanyRow, GraphPlan, OwnershipRow, SupplierRow
from supplier_risk.graph.store import DEFAULT_PATH_LIMIT, Neo4jGraphStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeCounters:
    def __init__(self, *, nodes_created: int = 0, relationships_created: int = 0) -> None:
        self.nodes_created = nodes_created
        self.relationships_created = relationships_created


class FakeSummary:
    def __init__(self, counters: FakeCounters) -> None:
        self.counters = counters


class FakeResult:
    def __init__(
        self,
        records: list[dict[str, Any]] | None = None,
        *,
        nodes_created: int = 0,
        relationships_created: int = 0,
    ) -> None:
        self._records = records or []
        self._summary = FakeSummary(
            FakeCounters(
                nodes_created=nodes_created,
                relationships_created=relationships_created,
            )
        )

    def __iter__(self) -> Any:
        return iter(self._records)

    def consume(self) -> FakeSummary:
        return self._summary


class FakeTransaction:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> FakeResult:
        self.driver.queries.append(query)
        self.driver.parameters.append(parameters or {})
        if self.driver.run_error is not None:
            raise self.driver.run_error
        if not self.driver.results:
            return FakeResult()
        return self.driver.results.pop(0)

    def commit(self) -> None:
        self.driver.commit_calls += 1

    def rollback(self) -> None:
        self.driver.rollback_calls += 1


class FakeSession:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def begin_transaction(self, *, timeout: float | None = None) -> FakeTransaction:
        self.driver.timeouts.append(timeout)
        return FakeTransaction(self.driver)


class FakeDriver:
    def __init__(
        self,
        *,
        results: list[FakeResult] | None = None,
        connect_error: Exception | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.results = list(results or [])
        self.connect_error = connect_error
        self.run_error = run_error
        self.queries: list[str] = []
        self.parameters: list[dict[str, Any]] = []
        self.timeouts: list[float | None] = []
        self.verify_calls = 0
        self.close_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0

    def session(self) -> FakeSession:
        return FakeSession(self)

    def verify_connectivity(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.verify_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class TimeoutNeo4jError(Neo4jError):
    """Neo4j error carrying the transaction-timeout code for classification tests."""

    def __init__(self) -> None:
        super().__init__("transaction timed out")
        self._neo4j_code = "Neo.ClientError.Transaction.TransactionTimedOut"


def _plan() -> GraphPlan:
    return GraphPlan(
        suppliers=(
            SupplierRow(
                supplier_id="SUP-001",
                legal_name="Norvend Components Ltd",
                country_code="GB",
                sector="electronics",
                partition="train",
            ),
        ),
        companies=(CompanyRow("ORG-001", "Meridian Industrial Holdings"),),
        ownership=(
            OwnershipRow(
                child_supplier_id="SUP-001",
                parent_company_id="ORG-001",
                relationship_type="JOINT_VENTURE_WITH",
                ownership_percentage=40.0,
            ),
        ),
    )


def test_close_is_idempotent() -> None:
    driver = FakeDriver()
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    store.close()
    store.close()

    assert driver.close_calls == 1


def test_verify_connectivity_translates_service_unavailable() -> None:
    store = Neo4jGraphStore(
        FakeDriver(connect_error=ServiceUnavailable("down"))  # type: ignore[arg-type]
    )

    with pytest.raises(GraphConnectionError, match="unavailable"):
        store.verify_connectivity()


def test_authentication_failure_is_a_config_error_without_the_password() -> None:
    store = Neo4jGraphStore(
        FakeDriver(connect_error=AuthError("bad password supersecret"))  # type: ignore[arg-type]
    )

    with pytest.raises(GraphConfigError) as info:
        store.verify_connectivity()

    assert "supersecret" not in str(info.value)
    assert "NEO4J_USER" in str(info.value)


def test_resolve_supplier_uses_parameters_and_a_bounded_timeout() -> None:
    driver = FakeDriver(
        results=[
            FakeResult(
                [
                    {
                        "supplier_id": "SUP-001",
                        "legal_name": "Norvend Components Ltd",
                        "partition": "train",
                        "sector": "electronics",
                        "country_code": "GB",
                    }
                ]
            )
        ]
    )
    store = Neo4jGraphStore(driver, query_timeout=3.5)  # type: ignore[arg-type]

    identity = store.resolve_supplier("SUP-001")

    assert identity is not None
    assert identity.supplier_id == "SUP-001"
    assert "$entity" in driver.queries[0]
    assert "toUpper(s.supplier_id)" in driver.queries[0]
    assert "SUP-001" not in driver.queries[0]
    assert driver.parameters[0] == {"entity": "SUP-001"}
    assert driver.timeouts == [3.5]


def test_resolve_supplier_detects_ambiguity() -> None:
    rows = [
        {
            "supplier_id": f"SUP-00{index}",
            "legal_name": "Same Name Ltd",
            "partition": "train",
            "sector": "electronics",
            "country_code": "GB",
        }
        for index in range(1, 3)
    ]
    store = Neo4jGraphStore(FakeDriver(results=[FakeResult(rows)]))  # type: ignore[arg-type]

    with pytest.raises(UnknownGraphEntityError, match="ambiguous"):
        store.resolve_supplier("Same Name Ltd")


def test_resolve_supplier_returns_none_for_no_match() -> None:
    store = Neo4jGraphStore(FakeDriver(results=[FakeResult([])]))  # type: ignore[arg-type]

    assert store.resolve_supplier("SUP-999") is None


def test_evidence_query_is_bounded_parameterized_and_mapped() -> None:
    driver = FakeDriver(
        results=[
            FakeResult(
                [
                    {
                        "child_key": "SUP-001",
                        "child_name": "Norvend Components Ltd",
                        "relationship_type": "JOINT_VENTURE_WITH",
                        "ownership_percentage": 40.0,
                        "parent_key": "ORG-001",
                        "parent_name": "Meridian Industrial Holdings",
                        "hops": 2,
                    }
                ]
            )
        ]
    )
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    rows = store.fetch_ownership_evidence("SUP-001")

    assert rows[0].parent_key == "ORG-001"
    assert rows[0].hops == 2
    assert "SUP-001" not in driver.queries[0]
    assert "*1..3" in driver.queries[0]
    assert driver.parameters[0]["supplier_id"] == "SUP-001"
    assert driver.parameters[0]["path_limit"] == DEFAULT_PATH_LIMIT


def test_evidence_with_an_unknown_type_is_integrity_failure() -> None:
    driver = FakeDriver(
        results=[
            FakeResult(
                [
                    {
                        "child_key": "SUP-001",
                        "child_name": "Norvend Components Ltd",
                        "relationship_type": "OWNS",
                        "ownership_percentage": 40.0,
                        "parent_key": "ORG-001",
                        "parent_name": "Meridian Industrial Holdings",
                        "hops": 1,
                    }
                ]
            )
        ]
    )
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    with pytest.raises(GraphDataIntegrityError, match="unsupported ownership"):
        store.fetch_ownership_evidence("SUP-001")


def test_constraints_are_created_idempotently() -> None:
    driver = FakeDriver(results=[FakeResult(), FakeResult(), FakeResult()])
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    store.ensure_constraints()

    assert len(driver.queries) == 3
    assert "SHOW CONSTRAINTS" in driver.queries[0]
    assert all("CREATE CONSTRAINT" in query for query in driver.queries[1:])
    assert all("IF NOT EXISTS" in query for query in driver.queries[1:])


def test_equivalent_constraints_are_verified_not_recreated() -> None:
    definitions = FakeResult(
        [
            {"labels": ["Supplier"], "properties": ["supplier_id"]},
            {"labels": ["Company"], "properties": ["parent_company_id"]},
        ]
    )
    driver = FakeDriver(results=[definitions])
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    store.ensure_constraints()

    assert len(driver.queries) == 1
    assert "SHOW CONSTRAINTS" in driver.queries[0]


def test_constraint_names_are_returned_in_order() -> None:
    driver = FakeDriver(
        results=[FakeResult([{"name": "company_id_unique"}, {"name": "supplier_id_unique"}])]
    )
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    assert store.constraint_names() == ("company_id_unique", "supplier_id_unique")


def test_upsert_reports_created_counts_and_parameterizes_rows() -> None:
    driver = FakeDriver(
        results=[
            FakeResult(nodes_created=1),
            FakeResult(nodes_created=1),
            FakeResult([{"merged": 1}], relationships_created=1),
        ]
    )
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    stats = store.upsert_plan(_plan())

    assert (stats.suppliers_created, stats.companies_created, stats.relationships_created) == (
        1,
        1,
        1,
    )
    assert len(driver.queries) == 3
    assert all("MERGE" in query for query in driver.queries)
    assert "SUP-001" not in " ".join(driver.queries)
    assert driver.parameters[0]["rows"][0]["supplier_id"] == "SUP-001"
    assert driver.commit_calls == 1


def test_upsert_rolls_back_and_does_not_report_success_on_failure() -> None:
    driver = FakeDriver(run_error=ServiceUnavailable("down"))
    store = Neo4jGraphStore(driver)  # type: ignore[arg-type]

    with pytest.raises(GraphConnectionError):
        store.upsert_plan(_plan())

    assert driver.rollback_calls == 1
    assert driver.commit_calls == 0


def test_query_timeout_is_classified_separately() -> None:
    store = Neo4jGraphStore(FakeDriver(run_error=TimeoutNeo4jError()))  # type: ignore[arg-type]

    with pytest.raises(GraphQueryError, match="timed out"):
        store.fetch_ownership_evidence("SUP-001")


def test_from_settings_creates_a_driver_without_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = FakeDriver()
    monkeypatch.setattr(
        store_module.GraphDatabase,
        "driver",
        lambda *args, **kwargs: driver,
    )
    settings = Neo4jSettings(
        uri="bolt://example:7687",
        user="graph-user",
        password=SecretStr("local-test-password"),
    )

    store = Neo4jGraphStore.from_settings(settings)

    assert driver.verify_calls == 0
    assert driver.close_calls == 0
    store.close()
    assert driver.close_calls == 1


def test_operations_after_close_fail_with_a_typed_error() -> None:
    store = Neo4jGraphStore(FakeDriver())  # type: ignore[arg-type]
    store.close()

    with pytest.raises(GraphConnectionError, match="already closed"):
        store.resolve_supplier("SUP-001")


def test_importing_the_graph_package_opens_no_connection() -> None:
    code = (
        "import neo4j\n"
        "def fail(*args, **kwargs):\n"
        "    raise AssertionError('driver created at import time')\n"
        "neo4j.GraphDatabase.driver = fail\n"
        "import supplier_risk.graph\n"
        "import supplier_risk.graph.store\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
