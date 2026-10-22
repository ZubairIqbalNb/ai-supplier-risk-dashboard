from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from supplier_risk.contracts import (
    PARTITION_SIZES,
    DatasetValidationError,
    SupplierDataset,
    parse_dataset_json,
    partition_for_supplier_id,
)
from supplier_risk.data.generate import (
    DEFAULT_OUTPUT,
    build_dataset,
    load_dataset,
    main,
    serialize_dataset,
    write_dataset,
)


@pytest.fixture(scope="module")
def dataset() -> SupplierDataset:
    return build_dataset()


def _payload(dataset: SupplierDataset) -> dict[str, Any]:
    return json.loads(serialize_dataset(dataset))


def test_generation_is_deterministic(dataset: SupplierDataset) -> None:
    rebuilt = build_dataset()

    assert rebuilt == dataset
    assert serialize_dataset(rebuilt) == serialize_dataset(dataset)


def test_different_seed_changes_generated_values(dataset: SupplierDataset) -> None:
    assert serialize_dataset(build_dataset(seed=1)) != serialize_dataset(dataset)


def test_partition_counts_match_the_approved_design(dataset: SupplierDataset) -> None:
    counts = Counter(record.partition for record in dataset.suppliers)

    assert counts == Counter(PARTITION_SIZES)
    assert sum(counts.values()) == 24
    assert 20 <= len(dataset.suppliers) <= 30


def test_partition_membership_is_fixed(dataset: SupplierDataset) -> None:
    for index, record in enumerate(dataset.suppliers, start=1):
        assert record.profile.supplier_id == f"SUP-{index:03d}"
        assert record.partition == partition_for_supplier_id(record.profile.supplier_id)


def test_partition_mapping_covers_the_manifest_and_nothing_else() -> None:
    partitions = [partition_for_supplier_id(f"SUP-{number:03d}") for number in range(1, 25)]

    assert Counter(partitions) == Counter(PARTITION_SIZES)
    for supplier_id in ("SUP-000", "SUP-025", "ORG-001"):
        with pytest.raises(ValueError, match="outside the manifest numbering range"):
            partition_for_supplier_id(supplier_id)


def test_supplier_identities_do_not_leak_across_partitions(dataset: SupplierDataset) -> None:
    supplier_ids = [record.profile.supplier_id for record in dataset.suppliers]
    legal_names = [record.profile.legal_name.casefold() for record in dataset.suppliers]

    assert len(supplier_ids) == len(set(supplier_ids))
    assert len(legal_names) == len(set(legal_names))

    partitions_by_supplier: dict[str, set[str]] = {}
    for record in dataset.suppliers:
        partitions_by_supplier.setdefault(record.profile.supplier_id, set()).add(record.partition)
    assert all(len(partitions) == 1 for partitions in partitions_by_supplier.values())


def test_generated_metrics_are_within_documented_ranges(dataset: SupplierDataset) -> None:
    for record in dataset.suppliers:
        metrics = record.delivery_metrics
        assert 0.0 <= metrics.on_time_delivery_rate <= 1.0
        assert 0.0 <= metrics.defect_rate <= 1.0
        assert 0.0 < metrics.average_lead_time_days <= 365.0
        assert metrics.orders_fulfilled >= 0


def test_every_contract_and_complaint_belongs_to_its_supplier(dataset: SupplierDataset) -> None:
    for record in dataset.suppliers:
        supplier_id = record.profile.supplier_id
        assert record.delivery_metrics.supplier_id == supplier_id
        assert all(contract.supplier_id == supplier_id for contract in record.contracts)
        assert all(complaint.supplier_id == supplier_id for complaint in record.complaints)
        if record.ownership is not None:
            assert record.ownership.child_supplier_id == supplier_id


def test_ownership_links_use_consistent_parent_companies(dataset: SupplierDataset) -> None:
    parent_names: dict[str, str] = {}
    for record in dataset.suppliers:
        if record.ownership is None:
            continue
        ownership = record.ownership
        known_name = parent_names.setdefault(
            ownership.parent_company_id, ownership.parent_company_name
        )
        assert known_name == ownership.parent_company_name


def test_serialized_dataset_round_trips_through_disk(
    dataset: SupplierDataset, tmp_path: Path
) -> None:
    output = write_dataset(dataset, tmp_path / "suppliers.json")

    loaded = load_dataset(output)

    assert loaded == dataset
    assert serialize_dataset(loaded) == serialize_dataset(dataset)


def test_cli_is_deterministic_for_the_same_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    assert main(["--seed", "321", "--output", str(first)]) == 0
    assert main(["--seed", "321", "--output", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    assert load_dataset(first).seed == 321
    output = capsys.readouterr().out
    assert "Wrote 24 synthetic suppliers" in output
    assert "train=14 dev=4 held_out=6" in output


def test_wrong_partition_count_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    payload["suppliers"].pop()

    with pytest.raises(DatasetValidationError, match="partition counts"):
        parse_dataset_json(json.dumps(payload))


def test_out_of_range_seed_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_dataset(seed=-1)


@pytest.mark.parametrize("seed", ["-1", str(2**32)])
def test_cli_rejects_out_of_range_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], seed: str
) -> None:
    output = tmp_path / "rejected.json"

    with pytest.raises(SystemExit) as exc_info:
        main(["--seed", seed, "--output", str(output)])

    assert exc_info.value.code == 2
    assert "--seed must be between 0 and" in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_non_integer_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "rejected.json"

    with pytest.raises(SystemExit) as exc_info:
        main(["--seed", "not-a-seed", "--output", str(output)])

    assert exc_info.value.code == 2
    assert "invalid int value" in capsys.readouterr().err
    assert not output.exists()


def test_generation_is_deterministic_across_processes(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    for output, hash_seed in ((first, "1"), (second, "2")):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        subprocess.run(
            [
                sys.executable,
                "-m",
                "supplier_risk.data.generate",
                "--seed",
                "987",
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            env=environment,
        )

    assert first.read_bytes() == second.read_bytes()


def test_default_output_lives_under_the_ignored_generated_directory() -> None:
    assert DEFAULT_OUTPUT.parts[-2:] == ("generated", "suppliers.json")
    assert "data" in DEFAULT_OUTPUT.parts
