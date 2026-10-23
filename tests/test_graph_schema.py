from __future__ import annotations

from dataclasses import replace

import pytest

from supplier_risk.contracts import SupplierDataset
from supplier_risk.graph.errors import GraphDataIntegrityError
from supplier_risk.graph.ingest import validate_existing_graph
from supplier_risk.graph.schema import (
    MAX_OWNERSHIP_RELATIONSHIPS,
    CompanyRow,
    GraphPlan,
    OwnershipEvidenceRow,
    OwnershipRow,
    SupplierIdentity,
    SupplierRow,
    build_evidence,
    build_graph_plan,
    validate_graph_plan,
)


def test_graph_plan_is_deterministic_and_complete(module2_dataset: SupplierDataset) -> None:
    plan = build_graph_plan(module2_dataset)
    again = build_graph_plan(module2_dataset)

    assert plan == again
    assert len(plan.suppliers) == 24
    assert len(plan.companies) == 10
    assert len(plan.ownership) == 24
    assert plan.supplier_ids == tuple(sorted(plan.supplier_ids))
    assert plan.company_ids == tuple(sorted(plan.company_ids))
    assert len(set(plan.supplier_ids)) == 24
    assert len(set(plan.company_ids)) == 10
    assert all(company.startswith("ORG-") for company in plan.company_ids)
    assert all(edge.child_supplier_id in plan.supplier_ids for edge in plan.ownership)
    assert all(edge.parent_company_id in plan.company_ids for edge in plan.ownership)
    assert {edge.relationship_type for edge in plan.ownership} == {
        "SUBSIDIARY_OF",
        "JOINT_VENTURE_WITH",
    }


def test_plan_without_ownership_edges_is_valid(module2_dataset: SupplierDataset) -> None:
    plan = build_graph_plan(module2_dataset)
    validate_graph_plan(GraphPlan(plan.suppliers, plan.companies, ()))


def test_ownership_nested_under_the_wrong_supplier_is_rejected(
    module2_dataset: SupplierDataset,
) -> None:
    record = module2_dataset.suppliers[0]
    assert record.ownership is not None
    bad_link = record.ownership.model_copy(update={"child_supplier_id": "SUP-002"})
    bad_record = record.model_copy(update={"ownership": bad_link})
    bad_dataset = module2_dataset.model_copy(
        update={"suppliers": (bad_record, *module2_dataset.suppliers[1:])}
    )

    with pytest.raises(GraphDataIntegrityError, match="nested under"):
        build_graph_plan(bad_dataset)


def _base_plan() -> GraphPlan:
    supplier = SupplierRow(
        supplier_id="SUP-001",
        legal_name="Test Supplier Ltd",
        country_code="GB",
        sector="electronics",
        partition="train",
    )
    company = CompanyRow(parent_company_id="ORG-001", parent_company_name="Test Holdings")
    edge = OwnershipRow(
        child_supplier_id="SUP-001",
        parent_company_id="ORG-001",
        relationship_type="JOINT_VENTURE_WITH",
        ownership_percentage=40.0,
    )
    return GraphPlan((supplier,), (company,), (edge,))


def test_duplicate_logical_edge_is_rejected() -> None:
    plan = _base_plan()
    duplicated = GraphPlan(plan.suppliers, plan.companies, plan.ownership * 2)

    with pytest.raises(GraphDataIntegrityError, match="duplicate logical ownership edge"):
        validate_graph_plan(duplicated)


def test_multiple_ownership_edges_for_one_supplier_are_rejected() -> None:
    plan = _base_plan()
    second = OwnershipRow(
        child_supplier_id="SUP-001",
        parent_company_id="ORG-002",
        relationship_type="SUBSIDIARY_OF",
        ownership_percentage=70.0,
    )
    companies = plan.companies + (CompanyRow("ORG-002", "Second Holdings"),)

    with pytest.raises(GraphDataIntegrityError, match="multiple ownership edges"):
        validate_graph_plan(GraphPlan(plan.suppliers, companies, plan.ownership + (second,)))


def test_self_link_is_rejected() -> None:
    plan = _base_plan()
    self_edge = replace(plan.ownership[0], parent_company_id="SUP-001")
    companies = plan.companies + (CompanyRow("SUP-001", "Impossible"),)

    with pytest.raises(GraphDataIntegrityError, match="self-referential ownership edge"):
        validate_graph_plan(GraphPlan(plan.suppliers, companies, (self_edge,)))


def test_unknown_child_endpoint_is_rejected() -> None:
    plan = _base_plan()
    edge = replace(plan.ownership[0], child_supplier_id="SUP-999")

    with pytest.raises(GraphDataIntegrityError, match="unknown child supplier endpoint"):
        validate_graph_plan(GraphPlan(plan.suppliers, plan.companies, (edge,)))


def test_unknown_parent_endpoint_is_rejected() -> None:
    plan = _base_plan()
    edge = replace(plan.ownership[0], parent_company_id="ORG-999")

    with pytest.raises(GraphDataIntegrityError, match="unknown parent company endpoint"):
        validate_graph_plan(GraphPlan(plan.suppliers, plan.companies, (edge,)))


def test_invalid_ownership_percentage_is_rejected() -> None:
    plan = _base_plan()
    edge = replace(plan.ownership[0], ownership_percentage=0.0)

    with pytest.raises(GraphDataIntegrityError, match="must be in \\(0, 100\\]"):
        validate_graph_plan(GraphPlan(plan.suppliers, plan.companies, (edge,)))


def test_subsidiary_rule_requires_a_majority() -> None:
    plan = _base_plan()
    edge = replace(
        plan.ownership[0],
        relationship_type="SUBSIDIARY_OF",
        ownership_percentage=50.0,
    )

    with pytest.raises(GraphDataIntegrityError, match="requires ownership_percentage greater than 50"):
        validate_graph_plan(GraphPlan(plan.suppliers, plan.companies, (edge,)))


def test_joint_venture_rule_rejects_a_majority() -> None:
    plan = _base_plan()
    edge = replace(plan.ownership[0], ownership_percentage=75.0)

    with pytest.raises(GraphDataIntegrityError, match="requires ownership_percentage at most 50"):
        validate_graph_plan(GraphPlan(plan.suppliers, plan.companies, (edge,)))


def test_partition_mismatch_is_rejected() -> None:
    plan = _base_plan()
    supplier = replace(plan.suppliers[0], partition="dev")

    with pytest.raises(GraphDataIntegrityError, match="must be train"):
        validate_graph_plan(GraphPlan((supplier,), plan.companies, plan.ownership))


def test_duplicate_company_key_is_rejected() -> None:
    plan = _base_plan()
    companies = plan.companies + (plan.companies[0],)

    with pytest.raises(GraphDataIntegrityError, match="duplicate parent_company_id"):
        validate_graph_plan(GraphPlan(plan.suppliers, companies, plan.ownership))


def test_company_id_colliding_with_a_supplier_is_rejected() -> None:
    plan = _base_plan()
    companies = (CompanyRow("SUP-001", "Colliding Company"),)

    with pytest.raises(GraphDataIntegrityError, match="collides with a supplier_id"):
        validate_graph_plan(GraphPlan(plan.suppliers, companies, ()))


def _evidence_row(**overrides: object) -> OwnershipEvidenceRow:
    values: dict[str, object] = {
        "child_key": "SUP-001",
        "child_name": "Test Supplier Ltd",
        "parent_key": "ORG-001",
        "parent_name": "Test Holdings",
        "relationship_type": "JOINT_VENTURE_WITH",
        "ownership_percentage": 40.0,
        "hops": 1,
    }
    values.update(overrides)
    return OwnershipEvidenceRow(**values)  # type: ignore[arg-type]


def _identity() -> SupplierIdentity:
    return SupplierIdentity(
        supplier_id="SUP-001",
        legal_name="Test Supplier Ltd",
        partition="train",
        sector="electronics",
        country_code="GB",
    )


def test_evidence_deduplicates_by_relationship_and_keeps_minimum_hop() -> None:
    rows = (_evidence_row(hops=2), _evidence_row(hops=1))
    evidence = build_evidence(_identity(), rows)

    assert len(evidence.relationships) == 1
    assert evidence.relationships[0].hops == 1
    assert evidence.truncated is False


def test_evidence_conflicting_percentages_are_rejected() -> None:
    rows = (_evidence_row(), _evidence_row(ownership_percentage=45.0))

    with pytest.raises(GraphDataIntegrityError, match="conflicting ownership percentages"):
        build_evidence(_identity(), rows)


def test_evidence_ordering_is_deterministic() -> None:
    rows = (
        _evidence_row(child_key="SUP-003", hops=2),
        _evidence_row(child_key="SUP-002", hops=1),
        _evidence_row(child_key="SUP-001", hops=1),
    )
    evidence = build_evidence(_identity(), rows)

    assert [row.child_key for row in evidence.relationships] == [
        "SUP-001",
        "SUP-002",
        "SUP-003",
    ]


def test_evidence_is_limited_to_twenty_five_relationships() -> None:
    rows = tuple(
        _evidence_row(child_key=f"SUP-{index:03d}", hops=1) for index in range(1, 31)
    )
    evidence = build_evidence(_identity(), rows)

    assert len(evidence.relationships) == MAX_OWNERSHIP_RELATIONSHIPS
    assert evidence.truncated is True


def test_evidence_rejects_a_hop_beyond_the_limit() -> None:
    with pytest.raises(GraphDataIntegrityError, match="invalid hop count"):
        build_evidence(_identity(), (_evidence_row(hops=4),))


def test_existing_graph_edge_with_a_different_parent_is_a_conflict() -> None:
    plan = _base_plan()
    existing = (replace(plan.ownership[0], parent_company_id="ORG-002"),)

    with pytest.raises(GraphDataIntegrityError, match="dataset does not contain"):
        validate_existing_graph(existing, plan)


def test_existing_graph_edge_with_a_different_type_is_a_conflict() -> None:
    plan = _base_plan()
    existing = (replace(plan.ownership[0], relationship_type="SUBSIDIARY_OF"),)

    with pytest.raises(GraphDataIntegrityError, match="dataset requires"):
        validate_existing_graph(existing, plan)


def test_existing_graph_duplicate_edges_are_a_conflict() -> None:
    plan = _base_plan()
    existing = plan.ownership * 2

    with pytest.raises(GraphDataIntegrityError, match="ownership edges for supplier"):
        validate_existing_graph(existing, plan)


def test_existing_graph_percentage_change_is_allowed() -> None:
    plan = _base_plan()
    existing = (replace(plan.ownership[0], ownership_percentage=45.0),)

    validate_existing_graph(existing, plan)


def test_existing_graph_data_outside_the_dataset_is_ignored() -> None:
    plan = _base_plan()
    unrelated = replace(
        plan.ownership[0], child_supplier_id="SUP-999", parent_company_id="ORG-999"
    )

    validate_existing_graph((unrelated,), plan)
