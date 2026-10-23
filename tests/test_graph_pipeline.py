from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Self

import pytest

import supplier_risk.graph.pipeline as pipeline_module
from supplier_risk.config import ConfigError
from supplier_risk.graph.errors import (
    GraphConfigError,
    GraphConnectionError,
    GraphDataIntegrityError,
    GraphQueryError,
    UnknownGraphEntityError,
)
from supplier_risk.graph.pipeline import format_graph_context, query_graph
from supplier_risk.graph.schema import (
    MAX_OWNERSHIP_RELATIONSHIPS,
    OwnershipEvidenceRow,
    SupplierIdentity,
    build_evidence,
)


def _identity() -> SupplierIdentity:
    return SupplierIdentity(
        supplier_id="SUP-001",
        legal_name="Norvend Components Ltd",
        partition="train",
        sector="electronics",
        country_code="GB",
    )


def _row(**overrides: object) -> OwnershipEvidenceRow:
    values: dict[str, object] = {
        "child_key": "SUP-001",
        "child_name": "Norvend Components Ltd",
        "parent_key": "ORG-001",
        "parent_name": "Meridian Industrial Holdings",
        "relationship_type": "JOINT_VENTURE_WITH",
        "ownership_percentage": 40.0,
        "hops": 1,
    }
    values.update(overrides)
    return OwnershipEvidenceRow(**values)  # type: ignore[arg-type]


DEFAULT_IDENTITY = _identity()


class FakeStore:
    def __init__(
        self,
        *,
        identity: SupplierIdentity | None = DEFAULT_IDENTITY,
        rows: Sequence[OwnershipEvidenceRow] = (),
        resolve_error: Exception | None = None,
        fetch_error: Exception | None = None,
    ) -> None:
        self.identity = identity
        self.rows = tuple(rows)
        self.resolve_error = resolve_error
        self.fetch_error = fetch_error

    def resolve_supplier(self, entity: str) -> SupplierIdentity | None:
        assert entity
        if self.resolve_error is not None:
            raise self.resolve_error
        return self.identity

    def fetch_ownership_evidence(self, supplier_id: str) -> tuple[OwnershipEvidenceRow, ...]:
        assert supplier_id
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.rows


class ManagedFakeStore(FakeStore):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True


def test_query_graph_returns_labeled_evidence() -> None:
    result = query_graph("SUP-001", store=FakeStore(rows=(_row(),)))

    assert result.startswith("[GRAPH] supplier=SUP-001 name=Norvend Components Ltd partition=train")
    assert "ownership_relationships=1" in result
    assert (
        '1. hop=1 focus=yes SUP-001 "Norvend Components Ltd" '
        '-[JOINT_VENTURE_WITH 40.0%]-> ORG-001 "Meridian Industrial Holdings"'
    ) in result
    assert "sql" not in result.lower()
    assert "neo4j" not in result.lower()


def test_related_supplier_edges_are_marked_as_not_focused() -> None:
    related = _row(
        child_key="SUP-014",
        child_name="Lakeshore Carton Supply",
        hops=2,
    )

    result = query_graph("SUP-001", store=FakeStore(rows=(_row(), related)))

    assert "focus=yes SUP-001" in result
    assert "focus=no SUP-014" in result
    assert result.index("SUP-001") < result.index("SUP-014")


def test_query_graph_no_evidence_is_a_successful_context() -> None:
    result = query_graph("SUP-001", store=FakeStore(rows=()))

    assert result.startswith("[GRAPH] ")
    assert "ownership_relationships=0" in result
    assert "truncated=false" in result
    assert "No ownership or subsidiary relationships were found within 3 hops." in result


def test_query_graph_unknown_supplier_is_tagged() -> None:
    result = query_graph("SUP-999", store=FakeStore(identity=None))

    assert result.startswith("[GRAPH ERROR] unknown supplier identity")
    assert "exact supplier legal name" in result


def test_query_graph_ambiguous_supplier_is_tagged() -> None:
    result = query_graph(
        "Shared Name",
        store=FakeStore(
            resolve_error=UnknownGraphEntityError("supplier identity 'Shared Name' is ambiguous")
        ),
    )

    assert result == "[GRAPH ERROR] supplier identity 'Shared Name' is ambiguous"


def test_query_graph_empty_identity_is_tagged_without_opening_a_store() -> None:
    assert query_graph("   ") == "[GRAPH ERROR] supplier identity must not be empty"
    assert query_graph(None) == "[GRAPH ERROR] supplier identity must not be empty"  # type: ignore[arg-type]


def test_query_graph_connection_failure_is_distinguished_from_no_evidence() -> None:
    result = query_graph(
        "SUP-001",
        store=FakeStore(fetch_error=GraphConnectionError("Neo4j is unavailable")),
    )

    assert result == "[GRAPH ERROR] graph retrieval unavailable: Neo4j is unavailable"


def test_query_graph_timeout_is_tagged_as_a_query_failure() -> None:
    result = query_graph(
        "SUP-001",
        store=FakeStore(fetch_error=GraphQueryError("Neo4j timed out while reading")),
    )

    assert result == "[GRAPH ERROR] graph query failed: Neo4j timed out while reading"


def test_query_graph_configuration_failure_is_tagged() -> None:
    result = query_graph(
        "SUP-001",
        store=FakeStore(resolve_error=GraphConfigError("NEO4J_PASSWORD is missing")),
    )

    assert result == "[GRAPH ERROR] graph configuration is invalid: NEO4J_PASSWORD is missing"


def test_query_graph_integrity_failure_is_tagged() -> None:
    result = query_graph(
        "SUP-001",
        store=FakeStore(fetch_error=GraphDataIntegrityError("conflicting ownership percentages")),
    )

    assert result == (
        "[GRAPH ERROR] graph data integrity failure: conflicting ownership percentages"
    )


def test_query_graph_missing_default_configuration_is_tagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> object:
        raise ConfigError("Missing or blank required environment variable(s): NEO4J_PASSWORD")

    monkeypatch.setattr(pipeline_module, "load_neo4j_settings", fail)
    monkeypatch.setattr(
        pipeline_module.Neo4jGraphStore,
        "from_settings",
        classmethod(lambda cls, settings: pytest.fail("must not create a store")),
    )

    result = query_graph("SUP-001")

    assert result == (
        "[GRAPH ERROR] graph configuration is invalid: Missing or blank required "
        "environment variable(s): NEO4J_PASSWORD"
    )


def test_default_store_is_closed_after_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ManagedFakeStore(rows=(_row(),))
    monkeypatch.setattr(pipeline_module, "load_neo4j_settings", lambda: object())
    monkeypatch.setattr(
        pipeline_module.Neo4jGraphStore,
        "from_settings",
        classmethod(lambda cls, settings: fake),
    )

    result = query_graph("SUP-001")

    assert result.startswith("[GRAPH] ")
    assert fake.closed is True


def test_format_applies_the_twenty_five_relationship_limit() -> None:
    rows = tuple(
        _row(child_key=f"SUP-{index:03d}", hops=1) for index in range(1, 31)
    )
    evidence = build_evidence(_identity(), rows)
    result = format_graph_context(evidence)

    assert "truncated=true" in result
    assert f"ownership_relationships={MAX_OWNERSHIP_RELATIONSHIPS}" in result
    assert "SUP-025" in result
    assert "SUP-026" not in result


def test_format_keeps_direction_and_percentage() -> None:
    evidence = build_evidence(
        _identity(),
        (_row(relationship_type="SUBSIDIARY_OF", ownership_percentage=80.0),),
    )
    result = format_graph_context(evidence)

    assert "-[SUBSIDIARY_OF 80.0%]-> ORG-001" in result
    assert "direction=child_supplier->parent_company" in result


def test_format_does_not_claim_a_relationship_that_was_not_returned() -> None:
    evidence = build_evidence(_identity(), ())
    result = format_graph_context(evidence)

    assert "No ownership or subsidiary relationships" in result
    assert "]->" not in result


def test_store_rows_still_work_when_replaced_for_deduplication() -> None:
    first = _row(hops=2)
    nearest = replace(first, hops=1)
    evidence = build_evidence(_identity(), (first, nearest))

    assert len(evidence.relationships) == 1
    assert evidence.relationships[0].hops == 1
