"""Deterministic Neo4j ingestion for the Module 2 ownership records.

Run::

    uv run --locked python -m supplier_risk.graph.ingest

The command loads the canonical generated dataset, validates every ownership
record into a graph plan, checks that an existing graph does not structurally
contradict the plan, creates the uniqueness constraints, and writes nodes and
edges in one transaction. Re-running it against the same dataset is idempotent.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from supplier_risk.config import ConfigError, load_neo4j_settings
from supplier_risk.contracts import DatasetValidationError, SupplierDataset
from supplier_risk.data.generate import DEFAULT_OUTPUT as DEFAULT_DATASET_PATH
from supplier_risk.data.generate import load_dataset
from supplier_risk.graph.errors import GraphDataIntegrityError, GraphError
from supplier_risk.graph.schema import GraphPlan, OwnershipRow, build_graph_plan
from supplier_risk.graph.store import GraphCounts, Neo4jGraphStore

GENERATE_COMMAND: Final = "uv run --locked python -m supplier_risk.data.generate"


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """Validated counts and write statistics for one ingestion run."""

    dataset_path: Path
    supplier_count: int
    company_count: int
    ownership_count: int
    suppliers_created: int
    companies_created: int
    relationships_created: int
    graph_counts: GraphCounts
    constraints: tuple[str, ...]


def validate_existing_graph(
    existing: Sequence[OwnershipRow], plan: GraphPlan
) -> None:
    """Reject a graph that structurally contradicts the canonical plan.

    Only edges whose child is a canonical supplier are checked; nodes and edges
    outside this dataset are left untouched. A different parent or relationship
    type would create a second logical edge, so it aborts before any write. A
    changed percentage is safe to update and is not treated as a conflict.

    Raises:
        GraphDataIntegrityError: for a duplicate edge, a different parent, a
            different relationship type, or several edges for one supplier.
    """
    expected = {
        (row.child_supplier_id, row.parent_company_id): row for row in plan.ownership
    }
    by_child: dict[str, list[OwnershipRow]] = {}
    for row in existing:
        by_child.setdefault(row.child_supplier_id, []).append(row)

    for child, rows in sorted(by_child.items()):
        if child not in plan.supplier_ids:
            continue
        if len(rows) > 1:
            raise GraphDataIntegrityError(
                f"graph has {len(rows)} ownership edges for supplier {child}; "
                "reconcile the graph before ingesting"
            )
        stored = rows[0]
        canonical = expected.get((child, stored.parent_company_id))
        if canonical is None:
            raise GraphDataIntegrityError(
                f"graph has an ownership edge {child} -> {stored.parent_company_id} "
                "that the dataset does not contain; reconcile the graph before ingesting"
            )
        if canonical.relationship_type != stored.relationship_type:
            raise GraphDataIntegrityError(
                f"graph ownership edge {child} -> {stored.parent_company_id} uses "
                f"{stored.relationship_type}, but the dataset requires "
                f"{canonical.relationship_type}; reconcile the graph before ingesting"
            )


def ingest_dataset(
    store: Neo4jGraphStore,
    dataset: SupplierDataset,
    *,
    dataset_path: Path,
) -> IngestSummary:
    """Validate, conflict-check, and write one dataset to Neo4j.

    The graph must be reachable; connection and query failures propagate as
    typed errors so the CLI can return a failure status. Writes happen in one
    transaction, so a mid-ingestion failure cannot look successful.
    """
    plan = build_graph_plan(dataset)
    store.verify_connectivity()
    store.ensure_constraints()

    existing = store.fetch_existing_ownership(plan.supplier_ids)
    validate_existing_graph(existing, plan)

    stats = store.upsert_plan(plan)
    counts = store.canonical_counts(
        supplier_ids=plan.supplier_ids, company_ids=plan.company_ids
    )
    if (counts.suppliers, counts.companies, counts.relationships) != (
        len(plan.suppliers),
        len(plan.companies),
        len(plan.ownership),
    ):
        raise GraphDataIntegrityError(
            "graph counts after ingestion do not match the validated dataset"
        )

    return IngestSummary(
        dataset_path=dataset_path,
        supplier_count=len(plan.suppliers),
        company_count=len(plan.companies),
        ownership_count=len(plan.ownership),
        suppliers_created=stats.suppliers_created,
        companies_created=stats.companies_created,
        relationships_created=stats.relationships_created,
        graph_counts=counts,
        constraints=store.constraint_names(),
    )


def _load_dataset(path: Path) -> SupplierDataset:
    try:
        return load_dataset(path)
    except FileNotFoundError:
        raise GraphDataIntegrityError(
            f"supplier dataset not found at {path}; run `{GENERATE_COMMAND}` first."
        ) from None
    except OSError:
        raise GraphDataIntegrityError(f"could not read the supplier dataset at {path}") from None
    except DatasetValidationError as error:
        raise GraphDataIntegrityError(str(error)) from None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the graph-ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Load the canonical supplier ownership graph into Neo4j."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Validated supplier dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    args = parser.parse_args(argv)

    try:
        dataset = _load_dataset(args.dataset)
        settings = load_neo4j_settings()
        with Neo4jGraphStore.from_settings(settings) as store:
            summary = ingest_dataset(store, dataset, dataset_path=args.dataset)
    except (ConfigError, GraphError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Dataset: {summary.dataset_path} (seed {dataset.seed}, "
        f"{summary.supplier_count} suppliers)"
    )
    print(
        f"Suppliers: {summary.supplier_count} "
        f"(created {summary.suppliers_created}, "
        f"matched {summary.supplier_count - summary.suppliers_created})"
    )
    print(
        f"Companies: {summary.company_count} "
        f"(created {summary.companies_created}, "
        f"matched {summary.company_count - summary.companies_created})"
    )
    print(
        f"Ownership edges: {summary.ownership_count} "
        f"(created {summary.relationships_created}, "
        f"matched {summary.ownership_count - summary.relationships_created})"
    )
    print(
        "Graph counts: "
        f"Supplier={summary.graph_counts.suppliers} "
        f"Company={summary.graph_counts.companies} "
        f"ownership_relationships={summary.graph_counts.relationships}"
    )
    print(f"Constraints: {', '.join(summary.constraints) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
