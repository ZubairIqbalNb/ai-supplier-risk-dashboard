"""Ownership graph schema, validation, and evidence shaping for Module 4.

The Module 2 contract stores each supplier's optional parent company
(``ORG-NNN``). Parent companies are not suppliers and their identifiers never
overlap the ``SUP-NNN`` namespace, so the faithful graph is:

``(:Supplier)-[:SUBSIDIARY_OF|:JOINT_VENTURE_WITH {ownership_percentage}]->(:Company)``

The edge is stored child -> parent, matching the contract fields
``child_supplier_id`` and ``parent_company_id``. Nothing in this module infers
edges from names or free text, and the current dataset contains at most one
ownership edge per supplier.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from supplier_risk.contracts import (
    OwnershipType,
    Partition,
    SupplierDataset,
    partition_for_supplier_id,
)
from supplier_risk.graph.errors import GraphDataIntegrityError

MAX_HOPS: Final = 3
MAX_OWNERSHIP_RELATIONSHIPS: Final = 25

SUPPLIER_LABEL: Final = "Supplier"
COMPANY_LABEL: Final = "Company"
OWNERSHIP_RELATIONSHIP_TYPES: Final[tuple[OwnershipType, ...]] = (
    "SUBSIDIARY_OF",
    "JOINT_VENTURE_WITH",
)

_SUPPLIER_ID_PATTERN: Final = re.compile(r"^SUP-\d{3}$")
_COMPANY_ID_PATTERN: Final = re.compile(r"^ORG-\d{3}$")
_PARTITIONS: Final = frozenset({"train", "dev", "held_out"})


@dataclass(frozen=True, slots=True)
class SupplierRow:
    """One supplier node, using only validated profile fields."""

    supplier_id: str
    legal_name: str
    country_code: str
    sector: str
    partition: Partition


@dataclass(frozen=True, slots=True)
class CompanyRow:
    """One parent-company node; companies are not suppliers."""

    parent_company_id: str
    parent_company_name: str


@dataclass(frozen=True, slots=True)
class OwnershipRow:
    """One validated ownership edge, stored child supplier -> parent company."""

    child_supplier_id: str
    parent_company_id: str
    relationship_type: OwnershipType
    ownership_percentage: float


@dataclass(frozen=True, slots=True)
class GraphPlan:
    """Everything one ingestion run will write, in deterministic order."""

    suppliers: tuple[SupplierRow, ...]
    companies: tuple[CompanyRow, ...]
    ownership: tuple[OwnershipRow, ...]

    @property
    def supplier_ids(self) -> tuple[str, ...]:
        return tuple(row.supplier_id for row in self.suppliers)

    @property
    def company_ids(self) -> tuple[str, ...]:
        return tuple(row.parent_company_id for row in self.companies)


@dataclass(frozen=True, slots=True)
class SupplierIdentity:
    """The resolved supplier a graph query is about."""

    supplier_id: str
    legal_name: str
    partition: str
    sector: str
    country_code: str


@dataclass(frozen=True, slots=True)
class OwnershipEvidenceRow:
    """One ownership relationship returned by a bounded graph traversal."""

    child_key: str
    child_name: str
    parent_key: str
    parent_name: str
    relationship_type: OwnershipType
    ownership_percentage: float
    hops: int


@dataclass(frozen=True, slots=True)
class OwnershipEvidence:
    """Deterministic, deduplicated ownership evidence for one supplier."""

    supplier: SupplierIdentity
    relationships: tuple[OwnershipEvidenceRow, ...]
    truncated: bool


def build_graph_plan(dataset: SupplierDataset) -> GraphPlan:
    """Build and validate the deterministic graph write plan for ``dataset``.

    Raises:
        GraphDataIntegrityError: if ownership records are malformed, duplicate,
            self-referential, or reference an entity outside the dataset.
    """
    suppliers = tuple(
        SupplierRow(
            supplier_id=record.profile.supplier_id,
            legal_name=record.profile.legal_name,
            country_code=record.profile.country_code,
            sector=record.profile.sector,
            partition=record.partition,
        )
        for record in sorted(dataset.suppliers, key=lambda item: item.profile.supplier_id)
    )
    supplier_ids = {row.supplier_id for row in suppliers}

    company_names: dict[str, str] = {}
    ownership: list[OwnershipRow] = []
    for record in sorted(dataset.suppliers, key=lambda item: item.profile.supplier_id):
        link = record.ownership
        if link is None:
            continue
        if link.child_supplier_id != record.profile.supplier_id:
            raise GraphDataIntegrityError(
                f"ownership record for {link.child_supplier_id} is nested under "
                f"{record.profile.supplier_id}"
            )
        ownership.append(
            OwnershipRow(
                child_supplier_id=link.child_supplier_id,
                parent_company_id=link.parent_company_id,
                relationship_type=link.relationship_type,
                ownership_percentage=link.ownership_percentage,
            )
        )
        known_name = company_names.get(link.parent_company_id)
        if known_name is None:
            company_names[link.parent_company_id] = link.parent_company_name
        elif known_name != link.parent_company_name:
            raise GraphDataIntegrityError(
                f"parent company {link.parent_company_id} has inconsistent names in the dataset"
            )
        if link.parent_company_id in supplier_ids:
            raise GraphDataIntegrityError(
                f"parent_company_id {link.parent_company_id} collides with a supplier_id"
            )

    plan = GraphPlan(
        suppliers=suppliers,
        companies=tuple(
            CompanyRow(parent_company_id=company_id, parent_company_name=company_names[company_id])
            for company_id in sorted(company_names)
        ),
        ownership=tuple(
            sorted(
                ownership,
                key=lambda row: (
                    row.child_supplier_id,
                    row.parent_company_id,
                    row.relationship_type,
                ),
            )
        ),
    )
    validate_graph_plan(plan)
    return plan


def validate_graph_plan(plan: GraphPlan) -> None:
    """Validate node keys, endpoints, ownership types, and duplicate edges.

    This duplicates the key contract rules deliberately: ingestion must not rely
    on a dataset having been validated somewhere else, and an invalid record must
    never become a plausible-looking edge.

    Raises:
        GraphDataIntegrityError: with every problem found, joined into one message.
    """
    issues: list[str] = []

    if not plan.suppliers:
        issues.append("graph plan contains no suppliers")

    supplier_ids: set[str] = set()
    company_names: dict[str, str] = {}
    for supplier in plan.suppliers:
        if _SUPPLIER_ID_PATTERN.fullmatch(supplier.supplier_id) is None:
            issues.append(f"invalid supplier_id: {supplier.supplier_id!r}")
        if supplier.supplier_id in supplier_ids:
            issues.append(f"duplicate supplier_id: {supplier.supplier_id}")
        supplier_ids.add(supplier.supplier_id)
        if not supplier.legal_name.strip():
            issues.append(f"blank legal_name for {supplier.supplier_id}")
        if supplier.partition not in _PARTITIONS:
            issues.append(f"invalid partition for {supplier.supplier_id}: {supplier.partition!r}")
        if _SUPPLIER_ID_PATTERN.fullmatch(supplier.supplier_id) is not None:
            try:
                expected = partition_for_supplier_id(supplier.supplier_id)
            except ValueError:
                expected = None
            if expected is not None and supplier.partition != expected:
                issues.append(
                    f"partition for {supplier.supplier_id} must be {expected}; "
                    f"found {supplier.partition}"
                )

    for company in plan.companies:
        if _COMPANY_ID_PATTERN.fullmatch(company.parent_company_id) is None:
            issues.append(f"invalid parent_company_id: {company.parent_company_id!r}")
        if company.parent_company_id in company_names:
            issues.append(f"duplicate parent_company_id: {company.parent_company_id}")
        company_names[company.parent_company_id] = company.parent_company_name
        if company.parent_company_id in supplier_ids:
            issues.append(
                f"parent_company_id {company.parent_company_id} collides with a supplier_id"
            )
        if not company.parent_company_name.strip():
            issues.append(f"blank parent_company_name for {company.parent_company_id}")

    seen_edges: set[tuple[str, str]] = set()
    children: set[str] = set()
    for edge in plan.ownership:
        if edge.child_supplier_id not in supplier_ids:
            issues.append(f"unknown child supplier endpoint: {edge.child_supplier_id}")
        if edge.child_supplier_id == edge.parent_company_id:
            issues.append(f"self-referential ownership edge: {edge.child_supplier_id}")
        if edge.parent_company_id in supplier_ids:
            issues.append(
                f"ownership parent {edge.parent_company_id} is a supplier; "
                "parent companies must be ORG entities"
            )
        if edge.parent_company_id not in company_names:
            issues.append(f"unknown parent company endpoint: {edge.parent_company_id}")
        if edge.relationship_type not in OWNERSHIP_RELATIONSHIP_TYPES:
            issues.append(
                f"unsupported ownership relationship type: {edge.relationship_type!r}"
            )
        if not math.isfinite(edge.ownership_percentage) or not (
            0.0 < edge.ownership_percentage <= 100.0
        ):
            issues.append(
                f"ownership_percentage for {edge.child_supplier_id} must be in (0, 100]"
            )
        elif (
            edge.relationship_type == "SUBSIDIARY_OF"
            and edge.ownership_percentage <= 50.0
        ):
            issues.append(
                f"SUBSIDIARY_OF for {edge.child_supplier_id} requires "
                "ownership_percentage greater than 50"
            )
        elif (
            edge.relationship_type == "JOINT_VENTURE_WITH"
            and edge.ownership_percentage > 50.0
        ):
            issues.append(
                f"JOINT_VENTURE_WITH for {edge.child_supplier_id} requires "
                "ownership_percentage at most 50"
            )

        edge_key = (edge.child_supplier_id, edge.parent_company_id)
        if edge_key in seen_edges:
            issues.append(
                f"duplicate logical ownership edge: {edge.child_supplier_id} -> "
                f"{edge.parent_company_id}"
            )
        seen_edges.add(edge_key)
        if edge.child_supplier_id in children:
            issues.append(
                f"multiple ownership edges for supplier {edge.child_supplier_id}; "
                "the current schema allows at most one parent"
            )
        children.add(edge.child_supplier_id)

    if issues:
        raise GraphDataIntegrityError("; ".join(issues))


def build_evidence(
    supplier: SupplierIdentity,
    rows: Sequence[OwnershipEvidenceRow],
) -> OwnershipEvidence:
    """Validate, deduplicate, order, and truncate traversal rows.

    Duplicate relationships reached by more than one path keep the smallest hop
    count. Ordering is deterministic: hop count, then child key, parent key, and
    relationship type. At most :data:`MAX_OWNERSHIP_RELATIONSHIPS` rows are kept.
    """
    merged: dict[tuple[str, str, str], OwnershipEvidenceRow] = {}
    for row in rows:
        _validate_evidence_row(row)
        key = (row.child_key, row.parent_key, row.relationship_type)
        existing = merged.get(key)
        if existing is None:
            merged[key] = row
            continue
        if existing.ownership_percentage != row.ownership_percentage:
            raise GraphDataIntegrityError(
                f"ownership edge {row.child_key} -> {row.parent_key} reports "
                "conflicting ownership percentages"
            )
        if row.hops < existing.hops:
            merged[key] = replace(existing, hops=row.hops)

    ordered = sorted(
        merged.values(),
        key=lambda row: (
            row.hops,
            row.child_key,
            row.parent_key,
            row.relationship_type,
        ),
    )
    truncated = len(ordered) > MAX_OWNERSHIP_RELATIONSHIPS
    return OwnershipEvidence(
        supplier=supplier,
        relationships=tuple(ordered[:MAX_OWNERSHIP_RELATIONSHIPS]),
        truncated=truncated,
    )


def _validate_evidence_row(row: OwnershipEvidenceRow) -> None:
    if not row.child_key or not row.parent_key:
        raise GraphDataIntegrityError("ownership evidence contains a blank node key")
    if not row.child_name or not row.parent_name:
        raise GraphDataIntegrityError(
            f"ownership evidence for {row.child_key} -> {row.parent_key} has a missing name"
        )
    if row.relationship_type not in OWNERSHIP_RELATIONSHIP_TYPES:
        raise GraphDataIntegrityError(
            f"ownership evidence has unsupported relationship type {row.relationship_type!r}"
        )
    if not math.isfinite(row.ownership_percentage) or not (
        0.0 < row.ownership_percentage <= 100.0
    ):
        raise GraphDataIntegrityError(
            f"ownership evidence for {row.child_key} has an invalid percentage"
        )
    if isinstance(row.hops, bool) or not isinstance(row.hops, int) or not (
        1 <= row.hops <= MAX_HOPS
    ):
        raise GraphDataIntegrityError(
            f"ownership evidence for {row.child_key} has an invalid hop count"
        )
