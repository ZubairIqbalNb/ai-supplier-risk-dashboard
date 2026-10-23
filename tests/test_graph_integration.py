"""Live Neo4j integration tests for Module 4.

These tests require the local Compose service and are skipped unless
``SUPPLIER_RISK_NEO4J_TESTS=1``. They use a test-only ``SUP-9xx`` / ``ORG-99x``
namespace and clean it up, so the canonical development graph is never wiped.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from neo4j import Driver, GraphDatabase
from pydantic import SecretStr

from supplier_risk.config import Neo4jSettings, load_neo4j_settings
from supplier_risk.contracts import SupplierDataset
from supplier_risk.graph.ingest import IngestSummary, ingest_dataset
from supplier_risk.graph.pipeline import query_graph
from supplier_risk.graph.schema import build_graph_plan
from supplier_risk.graph.store import GraphCounts, Neo4jGraphStore

DATASET_PATH = Path("data/generated/suppliers.json")
TEST_SUPPLIER_PREFIX = "SUP-9"
TEST_COMPANY_PREFIX = "ORG-99"

pytestmark = [
    pytest.mark.neo4j,
    pytest.mark.skipif(
        os.environ.get("SUPPLIER_RISK_NEO4J_TESTS") != "1",
        reason="set SUPPLIER_RISK_NEO4J_TESTS=1 to run against the local Neo4j service",
    ),
]


@pytest.fixture(scope="module")
def settings() -> Neo4jSettings:
    return load_neo4j_settings()


@pytest.fixture(scope="module")
def raw_driver(settings: Neo4jSettings) -> Iterator[Driver]:
    driver = GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password.get_secret_value()),
    )
    try:
        yield driver
    finally:
        driver.close()


def _clean_test_namespace(driver: Driver) -> None:
    with driver.session() as session:
        session.run(
            "MATCH (n) WHERE n.supplier_id STARTS WITH $supplier_prefix "
            "OR n.parent_company_id STARTS WITH $company_prefix "
            "DETACH DELETE n",
            supplier_prefix=TEST_SUPPLIER_PREFIX,
            company_prefix=TEST_COMPANY_PREFIX,
        ).consume()


@pytest.fixture(scope="module", autouse=True)
def clean_test_namespace(raw_driver: Driver) -> Iterator[None]:
    _clean_test_namespace(raw_driver)
    yield
    _clean_test_namespace(raw_driver)


@pytest.fixture(scope="module")
def store(settings: Neo4jSettings) -> Iterator[Neo4jGraphStore]:
    with Neo4jGraphStore.from_settings(settings) as graph_store:
        graph_store.verify_connectivity()
        yield graph_store


@pytest.fixture(scope="module")
def canonical_summary(
    store: Neo4jGraphStore, module2_dataset: SupplierDataset
) -> IngestSummary:
    return ingest_dataset(store, module2_dataset, dataset_path=DATASET_PATH)


def _create_supplier(driver: Driver, supplier_id: str, legal_name: str) -> None:
    with driver.session() as session:
        session.run(
            "CREATE (s:Supplier {supplier_id: $supplier_id, legal_name: $legal_name, "
            "country_code: 'GB', sector: 'packaging', partition: 'train'})",
            supplier_id=supplier_id,
            legal_name=legal_name,
        ).consume()


def test_live_connection_and_constraints(
    store: Neo4jGraphStore, canonical_summary: IngestSummary, module2_dataset: SupplierDataset
) -> None:
    names = store.constraint_names()
    assert "supplier_id_unique" in names
    assert "company_id_unique" in names

    plan = build_graph_plan(module2_dataset)
    assert canonical_summary.supplier_count == len(plan.suppliers) == 24
    assert canonical_summary.company_count == len(plan.companies) == 10
    assert canonical_summary.ownership_count == len(plan.ownership) == 24
    assert canonical_summary.graph_counts == GraphCounts(24, 10, 24)
    assert set(canonical_summary.constraints) == {
        "company_id_unique",
        "supplier_id_unique",
    }


def test_repeated_ingestion_is_idempotent(
    store: Neo4jGraphStore, module2_dataset: SupplierDataset, canonical_summary: IngestSummary
) -> None:
    second = ingest_dataset(store, module2_dataset, dataset_path=DATASET_PATH)

    assert second.suppliers_created == 0
    assert second.companies_created == 0
    assert second.relationships_created == 0
    assert second.graph_counts == canonical_summary.graph_counts


def test_each_supplier_has_at_most_one_logical_ownership_edge(
    raw_driver: Driver, module2_dataset: SupplierDataset
) -> None:
    plan = build_graph_plan(module2_dataset)
    with raw_driver.session() as session:
        rows = session.run(
            "MATCH (s:Supplier)-[r:SUBSIDIARY_OF|JOINT_VENTURE_WITH]->(c:Company) "
            "WHERE s.supplier_id IN $supplier_ids "
            "RETURN s.supplier_id AS supplier_id, count(r) AS edge_count",
            supplier_ids=list(plan.supplier_ids),
        ).data()

    assert len(rows) == 24
    assert all(row["edge_count"] == 1 for row in rows)


def test_query_keeps_ownership_direction(
    store: Neo4jGraphStore, module2_dataset: SupplierDataset, canonical_summary: IngestSummary
) -> None:
    plan = build_graph_plan(module2_dataset)
    edge = next(row for row in plan.ownership if row.child_supplier_id == "SUP-007")

    context = query_graph("SUP-007", store=store)

    assert context.startswith("[GRAPH] supplier=SUP-007 name=Sakurajima Precision KK")
    assert (
        f"-[{edge.relationship_type} {edge.ownership_percentage:.1f}%]-> "
        f"{edge.parent_company_id} "
    ) in context
    assert "direction=child_supplier->parent_company" in context


def test_query_by_exact_legal_name_and_case_insensitive_id(
    store: Neo4jGraphStore, canonical_summary: IngestSummary
) -> None:
    by_id = query_graph("SUP-007", store=store)
    by_name = query_graph("sakurajima precision kk", store=store)
    by_lowercase_id = query_graph("sup-007", store=store)

    assert by_id == by_name == by_lowercase_id


def test_unknown_supplier_is_tagged(store: Neo4jGraphStore, canonical_summary: IngestSummary) -> None:
    context = query_graph("SUP-999", store=store)

    assert context.startswith("[GRAPH ERROR] unknown supplier identity")


def test_valid_supplier_without_edges_returns_no_evidence(
    store: Neo4jGraphStore, raw_driver: Driver, canonical_summary: IngestSummary
) -> None:
    _create_supplier(raw_driver, "SUP-990", "Ownerless Test Supplier")

    context = query_graph("SUP-990", store=store)

    assert context.startswith("[GRAPH] supplier=SUP-990")
    assert "ownership_relationships=0" in context
    assert "No ownership or subsidiary relationships" in context
    assert "[GRAPH ERROR]" not in context


def test_output_is_limited_to_twenty_five_relationships(
    store: Neo4jGraphStore, raw_driver: Driver, canonical_summary: IngestSummary
) -> None:
    with raw_driver.session() as session:
        session.run(
            "CREATE (c:Company {parent_company_id: 'ORG-990', "
            "parent_company_name: 'Limit Test Holdings'})"
        ).consume()
        for index in range(950, 980):
            session.run(
                "MATCH (c:Company {parent_company_id: 'ORG-990'}) "
                "CREATE (s:Supplier {supplier_id: $supplier_id, legal_name: $legal_name, "
                "country_code: 'GB', sector: 'packaging', partition: 'train'}) "
                "CREATE (s)-[:JOINT_VENTURE_WITH {ownership_percentage: 40.0}]->(c)",
                supplier_id=f"SUP-{index}",
                legal_name=f"Limit Test Supplier {index}",
            ).consume()

    context = query_graph("SUP-950", store=store)

    assert "ownership_relationships=25" in context
    assert "truncated=true" in context
    assert len([line for line in context.splitlines() if line and line[0].isdigit()]) == 25


def test_traversal_stops_after_three_hops(
    store: Neo4jGraphStore, raw_driver: Driver, canonical_summary: IngestSummary
) -> None:
    with raw_driver.session() as session:
        session.run(
            "CREATE (s:Supplier {supplier_id: 'SUP-991', legal_name: 'Chain Test Supplier', "
            "country_code: 'GB', sector: 'packaging', partition: 'train'}), "
            "(a:Company {parent_company_id: 'ORG-991', parent_company_name: 'Chain One'}), "
            "(b:Company {parent_company_id: 'ORG-992', parent_company_name: 'Chain Two'}), "
            "(c:Company {parent_company_id: 'ORG-993', parent_company_name: 'Chain Three'}), "
            "(d:Company {parent_company_id: 'ORG-994', parent_company_name: 'Chain Four'}), "
            "(s)-[:SUBSIDIARY_OF {ownership_percentage: 60.0}]->(a), "
            "(a)-[:SUBSIDIARY_OF {ownership_percentage: 60.0}]->(b), "
            "(b)-[:SUBSIDIARY_OF {ownership_percentage: 60.0}]->(c), "
            "(c)-[:SUBSIDIARY_OF {ownership_percentage: 60.0}]->(d)"
        ).consume()

    context = query_graph("SUP-991", store=store)

    assert "hop=1" in context
    assert "hop=2" in context
    assert "hop=3" in context
    assert "ORG-993" in context
    assert "ORG-994" not in context


def test_database_unavailable_is_tagged() -> None:
    unavailable = Neo4jSettings(
        uri="bolt://127.0.0.1:1",
        user="neo4j",
        password=SecretStr("not-a-real-secret"),
    )
    with Neo4jGraphStore.from_settings(
        unavailable,
        connection_timeout=0.5,
        max_retry_time=0.5,
        query_timeout=1.0,
    ) as store:
        context = query_graph("SUP-001", store=store)

    assert context.startswith("[GRAPH ERROR] graph retrieval unavailable")


def test_closed_store_is_tagged_not_crashed(settings: Neo4jSettings) -> None:
    store = Neo4jGraphStore.from_settings(settings)
    store.close()

    context = query_graph("SUP-001", store=store)

    assert context == (
        "[GRAPH ERROR] graph retrieval unavailable: the Neo4j graph store is already closed"
    )
