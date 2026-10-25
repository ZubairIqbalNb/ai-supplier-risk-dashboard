"""Command-line build for the Module 5 SQLite structured supplier store.

Run::

    uv run --locked python -m supplier_risk.sql.build

The command loads the canonical generated dataset, validates it into a SQLite
load plan, writes the database in one transaction, verifies the row counts, and
prints the result. Re-running it against the same dataset replaces the file with
identical content. See ``docs/schema.md`` section 8 for the table design.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from supplier_risk.contracts import DatasetValidationError, SupplierDataset
from supplier_risk.data.generate import DEFAULT_OUTPUT as DEFAULT_DATASET_PATH
from supplier_risk.data.generate import load_dataset
from supplier_risk.sql.errors import SqlError
from supplier_risk.sql.schema import build_sql_plan
from supplier_risk.sql.store import (
    DEFAULT_DATABASE_PATH,
    SqlBuildSummary,
    build_database,
)

GENERATE_COMMAND = "uv run --locked python -m supplier_risk.data.generate"


def build_from_dataset(
    dataset: SupplierDataset, database_path: Path | str = DEFAULT_DATABASE_PATH
) -> SqlBuildSummary:
    """Validate ``dataset`` into a plan and write it to ``database_path``."""
    return build_database(build_sql_plan(dataset), database_path)


def _load_dataset(path: Path) -> SupplierDataset:
    try:
        return load_dataset(path)
    except FileNotFoundError:
        raise DatasetValidationError(
            f"supplier dataset not found at {path}; run `{GENERATE_COMMAND}` first."
        ) from None
    except OSError:
        raise DatasetValidationError(f"could not read the supplier dataset at {path}") from None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SQLite build CLI."""
    parser = argparse.ArgumentParser(
        description="Build the SQLite delivery/defect metrics store."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Validated supplier dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE_PATH,
        help=f"SQLite database path (default: {DEFAULT_DATABASE_PATH}).",
    )
    args = parser.parse_args(argv)

    try:
        dataset = _load_dataset(args.dataset)
        summary = build_from_dataset(dataset, args.database)
    except (DatasetValidationError, SqlError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Dataset: {args.dataset} (seed {dataset.seed})")
    print(f"Suppliers: {summary.suppliers}")
    print(f"Delivery metrics rows: {summary.delivery_metrics}")
    print(f"Database: {summary.database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
