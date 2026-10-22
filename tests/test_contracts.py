from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from supplier_risk.contracts import (
    PARTITION_SIZES,
    SCHEMA_VERSION,
    DatasetValidationError,
    DeliveryMetrics,
    OwnershipLink,
    SchemaVersion,
    SupplierDataset,
    parse_dataset_json,
    parse_dataset_payload,
    partition_for_supplier_id,
)
from supplier_risk.data.generate import build_dataset, serialize_dataset


@pytest.fixture(scope="module")
def dataset() -> SupplierDataset:
    return build_dataset()


def _payload(dataset: SupplierDataset) -> dict[str, Any]:
    return json.loads(serialize_dataset(dataset))


def _parse(payload: dict[str, Any]) -> SupplierDataset:
    return parse_dataset_json(json.dumps(payload))


def _set(payload: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_valid_generated_payload_round_trips(dataset: SupplierDataset) -> None:
    parsed = _parse(_payload(dataset))

    assert parsed == dataset
    assert len(parsed.suppliers) == 24


def test_python_typed_payload_round_trips(dataset: SupplierDataset) -> None:
    assert parse_dataset_payload(dataset.model_dump()) == dataset


def test_optional_complaints_and_ownership_may_be_absent(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    record = payload["suppliers"][0]
    record["complaints"] = []
    record["ownership"] = None

    parsed = _parse(payload)

    assert parsed.suppliers[0].complaints == ()
    assert parsed.suppliers[0].ownership is None


def test_missing_required_field_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    del payload["suppliers"][0]["profile"]

    with pytest.raises(DatasetValidationError, match=r"suppliers\.0\.profile"):
        _parse(payload)


def test_unknown_fields_are_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    payload["suppliers"][0]["risk_score"] = 12

    with pytest.raises(DatasetValidationError, match="risk_score"):
        _parse(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("suppliers", 0, "profile", "employee_count"), "500"),
        (("suppliers", 0, "profile", "founded_year"), 1954.5),
        (("suppliers", 0, "delivery_metrics", "on_time_delivery_rate"), 1.2),
        (("suppliers", 0, "delivery_metrics", "defect_rate"), -0.01),
        (("suppliers", 0, "delivery_metrics", "average_lead_time_days"), 0.0),
        (("suppliers", 0, "contracts", 0, "annual_value_usd"), -1),
        (("suppliers", 0, "profile", "country_code"), "gb"),
        (("suppliers", 0, "profile", "sector"), "aerospace"),
        (("suppliers", 0, "partition"), "validation"),
        (("suppliers", 0, "contracts", 0, "contract_text"), ""),
    ],
)
def test_invalid_types_ranges_and_categories_are_rejected(
    dataset: SupplierDataset, path: tuple[str | int, ...], value: Any
) -> None:
    payload = _payload(dataset)
    _set(payload, path, value)

    with pytest.raises(DatasetValidationError):
        _parse(payload)


def test_empty_complaint_list_remains_valid(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    payload["suppliers"][0]["complaints"] = []

    assert _parse(payload).suppliers[0].complaints == ()


@pytest.mark.parametrize(
    "field", ["on_time_delivery_rate", "defect_rate", "average_lead_time_days"]
)
@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_every_float_delivery_field_rejects_non_finite(field: str, invalid_value: float) -> None:
    values: dict[str, Any] = {
        "supplier_id": "SUP-001",
        "reporting_period": "FY2025",
        "on_time_delivery_rate": 0.9,
        "defect_rate": 0.01,
        "average_lead_time_days": 14.0,
        "orders_fulfilled": 100,
    }
    values[field] = invalid_value

    with pytest.raises(ValidationError):
        DeliveryMetrics(**values)


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_ownership_percentage_rejects_non_finite(invalid_value: float) -> None:
    with pytest.raises(ValidationError):
        OwnershipLink(
            parent_company_id="ORG-001",
            parent_company_name="Meridian Industrial Holdings",
            child_supplier_id="SUP-001",
            ownership_percentage=invalid_value,
            relationship_type="SUBSIDIARY_OF",
        )


def test_integer_values_are_widened_for_float_fields() -> None:
    """Pydantic strict mode accepts int for float; other type mismatches stay rejected."""
    metrics = DeliveryMetrics(
        supplier_id="SUP-001",
        reporting_period="FY2025",
        on_time_delivery_rate=1,
        defect_rate=0,
        average_lead_time_days=14,
        orders_fulfilled=100,
    )
    link = OwnershipLink(
        parent_company_id="ORG-001",
        parent_company_name="Meridian Industrial Holdings",
        child_supplier_id="SUP-001",
        ownership_percentage=100,
        relationship_type="SUBSIDIARY_OF",
    )

    assert metrics.on_time_delivery_rate == 1.0
    assert metrics.defect_rate == 0.0
    assert link.ownership_percentage == 100.0


def test_duplicate_supplier_identifier_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    payload["suppliers"].append(payload["suppliers"][0])

    with pytest.raises(DatasetValidationError, match="duplicate supplier_id"):
        _parse(payload)


def test_duplicate_contract_identifier_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    duplicate_id = payload["suppliers"][0]["contracts"][0]["contract_id"]
    _set(payload, ("suppliers", 1, "contracts", 0, "contract_id"), duplicate_id)

    with pytest.raises(DatasetValidationError, match="duplicate contract_id"):
        _parse(payload)


def test_duplicate_contract_id_within_one_supplier_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    record = next(record for record in payload["suppliers"] if len(record["contracts"]) > 1)
    record["contracts"] = [record["contracts"][0], record["contracts"][0]]

    with pytest.raises(DatasetValidationError, match="unique within a supplier record"):
        _parse(payload)


def test_duplicate_complaint_id_within_one_supplier_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    record = next(record for record in payload["suppliers"] if len(record["complaints"]) > 1)
    record["complaints"] = [record["complaints"][0], record["complaints"][0]]

    with pytest.raises(DatasetValidationError, match="unique within a supplier record"):
        _parse(payload)


def test_duplicate_complaint_identifier_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    source = payload["suppliers"][0]["complaints"][0]
    duplicate = dict(source)
    duplicate["supplier_id"] = payload["suppliers"][1]["profile"]["supplier_id"]
    payload["suppliers"][1]["complaints"].append(duplicate)

    with pytest.raises(DatasetValidationError, match="duplicate complaint_id"):
        _parse(payload)


def test_duplicate_supplier_legal_name_is_rejected_case_insensitively(
    dataset: SupplierDataset,
) -> None:
    payload = _payload(dataset)
    first_name = payload["suppliers"][0]["profile"]["legal_name"]
    _set(payload, ("suppliers", 1, "profile", "legal_name"), first_name.upper())

    with pytest.raises(DatasetValidationError, match="duplicate supplier legal_name"):
        _parse(payload)


def test_contract_expiry_must_be_after_effective_date(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    contract = payload["suppliers"][0]["contracts"][0]
    contract["expiry_date"] = contract["effective_date"]

    with pytest.raises(DatasetValidationError, match="expiry_date"):
        _parse(payload)


def test_nested_records_must_match_supplier_identity(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    contract = payload["suppliers"][0]["contracts"][0]
    contract["supplier_id"] = "SUP-002"

    with pytest.raises(DatasetValidationError, match="nested records"):
        _parse(payload)


def test_delivery_metrics_must_match_supplier_identity(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    _set(payload, ("suppliers", 0, "delivery_metrics", "supplier_id"), "SUP-002")

    with pytest.raises(DatasetValidationError, match="nested records"):
        _parse(payload)


def test_complaint_must_match_supplier_identity(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    index = next(
        index for index, record in enumerate(payload["suppliers"]) if record["complaints"]
    )
    payload["suppliers"][index]["complaints"][0]["supplier_id"] = "SUP-024"

    with pytest.raises(DatasetValidationError, match="nested records"):
        _parse(payload)


def test_ownership_child_must_match_supplier_identity(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    ownership = payload["suppliers"][0]["ownership"]
    assert ownership is not None
    ownership["child_supplier_id"] = "SUP-002"

    with pytest.raises(DatasetValidationError, match="nested records"):
        _parse(payload)


def test_ownership_relationship_must_match_percentage(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    ownership = payload["suppliers"][0]["ownership"]
    assert ownership is not None
    ownership["relationship_type"] = "SUBSIDIARY_OF"
    ownership["ownership_percentage"] = 40.0

    with pytest.raises(DatasetValidationError, match="SUBSIDIARY_OF"):
        _parse(payload)


def test_parent_company_names_must_be_consistent(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    parent_id = payload["suppliers"][0]["ownership"]["parent_company_id"]
    ownership = payload["suppliers"][1]["ownership"]
    ownership["parent_company_id"] = parent_id
    ownership["parent_company_name"] = "Mismatched Holdings"

    with pytest.raises(DatasetValidationError, match="inconsistent parent_company_name"):
        _parse(payload)


def test_valid_generated_dataset_passes_partition_membership(dataset: SupplierDataset) -> None:
    parsed = _parse(_payload(dataset))

    for record in parsed.suppliers:
        assert record.partition == partition_for_supplier_id(record.profile.supplier_id)


def test_partition_membership_swap_preserving_counts_is_rejected(dataset: SupplierDataset) -> None:
    payload = _payload(dataset)
    train_record = payload["suppliers"][0]
    dev_record = payload["suppliers"][PARTITION_SIZES["train"]]
    train_record["partition"], dev_record["partition"] = (
        dev_record["partition"],
        train_record["partition"],
    )
    assert Counter(record["partition"] for record in payload["suppliers"]) == Counter(
        PARTITION_SIZES
    )

    with pytest.raises(DatasetValidationError, match="must be in partition train"):
        _parse(payload)


def test_supplier_id_outside_the_manifest_is_rejected(dataset: SupplierDataset) -> None:
    payload = json.dumps(_payload(dataset)).replace('"SUP-001"', '"SUP-099"')

    with pytest.raises(DatasetValidationError, match="outside the fixed manifest"):
        parse_dataset_json(payload)


def test_many_integrity_issues_are_aggregated_without_truncation(
    dataset: SupplierDataset,
) -> None:
    payload = _payload(dataset)
    payload["suppliers"] = [payload["suppliers"][0]] * 30

    with pytest.raises(DatasetValidationError) as exc_info:
        _parse(payload)

    message = str(exc_info.value)
    assert message.count("duplicate supplier_id") >= 2
    assert "partition counts must be" in message
    assert "more error(s)" not in message


def test_schema_version_has_a_single_runtime_source(dataset: SupplierDataset) -> None:
    assert get_args(SchemaVersion) == (SCHEMA_VERSION,)
    assert dataset.schema_version == SCHEMA_VERSION
    assert json.loads(serialize_dataset(dataset))["schema_version"] == SCHEMA_VERSION


def test_invalid_json_is_reported_without_record_contents() -> None:
    with pytest.raises(DatasetValidationError, match="Invalid supplier dataset"):
        parse_dataset_json('{"suppliers": [}')


def test_error_message_does_not_echo_rejected_record_values(
    dataset: SupplierDataset,
) -> None:
    sentinel = "sentinel-secret-value"
    payload = _payload(dataset)
    _set(payload, ("suppliers", 0, "profile", "legal_name"), f"  {sentinel}  ")

    with pytest.raises(DatasetValidationError) as exc_info:
        _parse(payload)

    message = str(exc_info.value)
    assert "legal_name" in message
    assert sentinel not in message
