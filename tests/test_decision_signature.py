"""Signature and verdict-contract tests for Module 6.

These tests cover the DSPy Signature surface and the typed contract without any
model call: they never need credentials or network access.
"""

from __future__ import annotations

import json

import pytest

from supplier_risk.decision import (
    MAX_FINDINGS,
    MAX_RATIONALE_CHARS,
    VERDICT_FIELDS,
    DecisionInputError,
    DecisionResponseError,
    SupplierMismatchError,
    SupplierRiskSignature,
    SupplierRiskVerdict,
    parse_supplier_identity,
    parse_verdict_payload,
    require_matching_supplier,
)
from tests.decision_support import CLEAN_FINDINGS, CLEAN_RATIONALE, prediction

OUTPUT_FIELDS = {
    "supplier_id",
    "verdict",
    "rationale",
    "findings",
    "missing_evidence",
}


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "supplier_id": "SUP-001",
        "verdict": "FLAG",
        "rationale": CLEAN_RATIONALE,
        "findings": list(CLEAN_FINDINGS),
        "missing_evidence": [],
    }
    payload.update(overrides)
    return payload


def test_signature_has_exactly_three_evidence_inputs() -> None:
    assert list(SupplierRiskSignature.input_fields) == [
        "sql_context",
        "faiss_context",
        "graph_context",
    ]


def test_signature_outputs_match_the_verdict_contract() -> None:
    assert set(SupplierRiskSignature.output_fields) == OUTPUT_FIELDS
    assert set(SupplierRiskVerdict.model_fields) == OUTPUT_FIELDS
    assert tuple(VERDICT_FIELDS) == (
        "supplier_id",
        "verdict",
        "rationale",
        "findings",
        "missing_evidence",
    )


def test_signature_instructions_require_evidence_discipline() -> None:
    instructions = SupplierRiskSignature.instructions

    assert "APPROVE" in instructions
    assert "FLAG" in instructions
    assert "missing_evidence" in instructions
    assert "[SQL ERROR]" in instructions
    assert "Never invent supplier facts" in instructions
    assert "failed source is never evidence" in instructions


@pytest.mark.parametrize("verdict", ["APPROVE", "FLAG"])
def test_valid_verdicts_are_accepted(verdict: str) -> None:
    parsed = parse_verdict_payload(_payload(verdict=verdict))

    assert parsed.verdict == verdict
    assert parsed.supplier_id == "SUP-001"


def test_verdict_and_identity_are_normalized() -> None:
    parsed = parse_verdict_payload(_payload(verdict=" approve ", supplier_id="sup-009"))

    assert parsed.verdict == "APPROVE"
    assert parsed.supplier_id == "SUP-009"


def test_invalid_verdict_value_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="APPROVE, FLAG"):
        parse_verdict_payload(_payload(verdict="MAYBE"))


def test_missing_required_field_is_rejected() -> None:
    payload = _payload()
    del payload["findings"]

    with pytest.raises(DecisionResponseError, match="findings"):
        parse_verdict_payload(payload)


def test_missing_field_on_a_prediction_is_rejected() -> None:
    import dspy

    partial_prediction = dspy.Prediction(
        supplier_id="SUP-001",
        verdict="FLAG",
        rationale=CLEAN_RATIONALE,
        findings=list(CLEAN_FINDINGS),
    )

    with pytest.raises(DecisionResponseError, match="missing_evidence"):
        parse_verdict_payload(partial_prediction)


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="confidence"):
        parse_verdict_payload(_payload(confidence=0.93))


def test_free_form_text_is_not_a_verdict() -> None:
    with pytest.raises(DecisionResponseError, match="free-form"):
        parse_verdict_payload("APPROVE - the supplier looks fine")


def test_none_payload_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="no structured output"):
        parse_verdict_payload(None)


def test_letters_in_supplier_id_are_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="supplier_id"):
        parse_verdict_payload(_payload(supplier_id="Norvend Components Ltd"))


def test_blank_rationale_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="rationale"):
        parse_verdict_payload(_payload(rationale="   "))


def test_short_rationale_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="rationale"):
        parse_verdict_payload(_payload(rationale="Too short"))


def test_overlong_rationale_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="rationale"):
        parse_verdict_payload(_payload(rationale="x" * (MAX_RATIONALE_CHARS + 1)))


def test_empty_findings_are_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="findings"):
        parse_verdict_payload(_payload(findings=[]))


def test_blank_finding_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match=r"findings\[1\]"):
        parse_verdict_payload(_payload(findings=["[SQL] defect_rate=0.0291", "   "]))


def test_non_string_finding_is_rejected() -> None:
    with pytest.raises(DecisionResponseError, match="findings"):
        parse_verdict_payload(_payload(findings=[42]))


def test_too_many_findings_are_rejected() -> None:
    findings = [f"[SQL] finding number {index}" for index in range(MAX_FINDINGS + 1)]

    with pytest.raises(DecisionResponseError, match="findings"):
        parse_verdict_payload(_payload(findings=findings))


def test_missing_evidence_may_be_empty_but_not_blank() -> None:
    assert parse_verdict_payload(_payload(missing_evidence=[])).missing_evidence == ()

    with pytest.raises(DecisionResponseError, match="missing_evidence"):
        parse_verdict_payload(_payload(missing_evidence=[" "]))


def test_mismatched_supplier_identity_is_rejected() -> None:
    verdict = parse_verdict_payload(_payload(supplier_id="SUP-002"))

    with pytest.raises(SupplierMismatchError, match="SUP-002"):
        require_matching_supplier(verdict, "SUP-001")


def test_matching_supplier_identity_is_returned() -> None:
    verdict = parse_verdict_payload(_payload())

    assert require_matching_supplier(verdict, "SUP-001") is verdict


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("SUP-007", "SUP-007"), (" sup-007 ", "SUP-007")],
)
def test_supplier_identity_is_normalized(raw: str, expected: str) -> None:
    assert parse_supplier_identity(raw) == expected


@pytest.mark.parametrize("raw", ["Norvend Components Ltd", "SUP-7", "SUP-0001", "", "  "])
def test_unusable_supplier_identity_is_rejected(raw: str) -> None:
    with pytest.raises(DecisionInputError):
        parse_supplier_identity(raw)


def test_serialization_is_deterministic_and_round_trips() -> None:
    verdict = parse_verdict_payload(_payload())

    first = verdict.to_dict()
    second = verdict.to_dict()

    assert first == second
    assert list(first) == list(VERDICT_FIELDS)
    assert json.dumps(first, sort_keys=False) == json.dumps(second, sort_keys=False)
    assert parse_verdict_payload(first) == verdict


def test_chain_of_thought_reasoning_is_not_part_of_the_contract() -> None:
    verdict = parse_verdict_payload(prediction())

    payload = verdict.to_dict()
    assert "reasoning" not in payload
    assert "internal chain-of-thought" not in json.dumps(payload)
