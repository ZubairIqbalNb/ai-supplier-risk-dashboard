"""Deterministic expected-verdict label tests for Module 7.

The label policy is a total, deterministic function of a raw supplier record. It
is exercised against the real generated dataset plus a few hand-built boundary
records, with no model, network, or store access.
"""

from __future__ import annotations

import pytest

from supplier_risk.contracts import DeliveryMetrics
from supplier_risk.data.generate import build_dataset
from supplier_risk.decision.verdict import VERDICT_VALUES
from supplier_risk.optimization.labels import (
    FLAG_DEFECT_RATE_THRESHOLD,
    FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD,
    expected_verdict,
    label_policy_description,
)

DATASET = build_dataset()
RECORDS = {record.profile.supplier_id: record for record in DATASET.suppliers}


def _with_metrics(supplier_id: str, **changes: object) -> object:
    record = RECORDS[supplier_id]
    metrics = record.delivery_metrics.model_dump()
    metrics.update(changes)
    return record.model_copy(
        update={"delivery_metrics": DeliveryMetrics(**metrics), "complaints": ()}
    )


def test_every_supplier_gets_a_valid_verdict() -> None:
    assert all(expected_verdict(record) in VERDICT_VALUES for record in DATASET.suppliers)


def test_high_severity_complaint_flags() -> None:
    # SUP-002 has a HIGH severity complaint and a defect rate above the threshold.
    assert expected_verdict(RECORDS["SUP-002"]) == "FLAG"


def test_high_defect_rate_flags() -> None:
    # SUP-008 has only a MEDIUM complaint but a defect rate of 0.0358.
    assert RECORDS["SUP-008"].delivery_metrics.defect_rate > FLAG_DEFECT_RATE_THRESHOLD
    assert expected_verdict(RECORDS["SUP-008"]) == "FLAG"


def test_low_on_time_delivery_flags() -> None:
    # SUP-010 has a moderate defect rate but on_time_delivery_rate 0.816.
    assert RECORDS["SUP-010"].delivery_metrics.on_time_delivery_rate < (
        FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD
    )
    assert expected_verdict(RECORDS["SUP-010"]) == "FLAG"


def test_clean_supplier_is_approved() -> None:
    assert expected_verdict(RECORDS["SUP-006"]) == "APPROVE"


@pytest.mark.parametrize("value", [FLAG_DEFECT_RATE_THRESHOLD])
def test_defect_rate_at_the_threshold_is_not_flagged(value: float) -> None:
    record = _with_metrics(
        "SUP-006",
        defect_rate=value,
        on_time_delivery_rate=0.99,
    )
    assert expected_verdict(record) == "APPROVE"


def test_on_time_rate_at_the_threshold_is_not_flagged() -> None:
    record = _with_metrics(
        "SUP-006",
        defect_rate=0.001,
        on_time_delivery_rate=FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD,
    )
    assert expected_verdict(record) == "APPROVE"


def test_policy_is_deterministic() -> None:
    record = RECORDS["SUP-003"]
    assert expected_verdict(record) == expected_verdict(record)


def test_both_verdicts_occur_in_the_dataset() -> None:
    labels = {expected_verdict(record) for record in DATASET.suppliers}
    assert labels == {"APPROVE", "FLAG"}


def test_label_policy_description_is_serializable() -> None:
    description = label_policy_description()
    assert description["policy_version"] == "module7.labels.v1"
    assert description["flag_defect_rate_threshold"] == FLAG_DEFECT_RATE_THRESHOLD
    assert description["flag_on_time_delivery_rate_threshold"] == (
        FLAG_ON_TIME_DELIVERY_RATE_THRESHOLD
    )
