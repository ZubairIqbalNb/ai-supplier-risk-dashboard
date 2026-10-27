"""Deterministic judge tests for Module 6.

The judge is pure: these tests assert the exact rules, the conservative
downgrade, repeatability, and that no model or DSPy call happens.
"""

from __future__ import annotations

import dspy
import pytest

import supplier_risk.decision.judge as judge_module
from supplier_risk.decision import (
    BANNED_TERMS,
    banned_terms_in,
    judge_payload,
    judge_verdict,
)
from tests.decision_support import clean_verdict, make_retrieval

PARTIAL = {"graph": "error"}
FAILED = {"sql": "error", "faiss": "error", "graph": "error"}
EMPTY = {"sql": "empty", "faiss": "empty", "graph": "empty"}
GRAPH_ERROR = "[GRAPH ERROR] RuntimeError: store unavailable"
SQL_ERROR = "[SQL ERROR] RuntimeError: store unavailable"
FAISS_ERROR = "[FAISS ERROR] RuntimeError: store unavailable"

APPROVE_RATIONALE = (
    "SQL delivery metrics are stable, FAISS shows no adverse contract or complaint "
    "wording, and the graph confirms a majority-owned subsidiary."
)


def test_clean_flag_verdict_passes() -> None:
    result = judge_verdict(clean_verdict(), make_retrieval())

    assert result.passed is True
    assert result.model_verdict == "FLAG"
    assert result.judge_verdict == "FLAG"
    assert result.triggered_rules == ()
    assert result.rule_triggered is False


def test_clean_approve_verdict_with_evidence_passes() -> None:
    result = judge_verdict(
        clean_verdict(verdict="APPROVE", rationale=APPROVE_RATIONALE), make_retrieval()
    )

    assert result.passed is True
    assert result.judge_verdict == "APPROVE"


def test_banned_content_rule_is_triggered() -> None:
    verdict = clean_verdict(
        rationale="The supplier has no risk and is guaranteed to deliver on time."
    )

    result = judge_verdict(verdict, make_retrieval())

    assert result.passed is False
    assert result.triggered_rules == ("banned_content",)
    assert result.model_verdict == "FLAG"
    assert result.judge_verdict == "FLAG"
    assert "no risk" in result.reasons[0]
    assert "guarantee" in result.reasons[0]


def test_banned_content_rule_scans_findings_too() -> None:
    verdict = clean_verdict(findings=["[GRAPH] no sanctions apply to this supplier"])

    assert "sanction" in banned_terms_in(verdict)
    assert "banned_content" in judge_verdict(verdict, make_retrieval()).triggered_rules


def test_banned_content_rule_is_not_triggered_by_clean_evidence() -> None:
    verdict = clean_verdict()

    assert banned_terms_in(verdict) == ()
    assert "banned_content" not in judge_verdict(verdict, make_retrieval()).triggered_rules


@pytest.mark.parametrize("term", BANNED_TERMS)
def test_every_documented_banned_term_is_enforced(term: str) -> None:
    verdict = clean_verdict(rationale=f"The supplied evidence shows {term} for this supplier.")

    assert term in banned_terms_in(verdict)


def test_conflicting_model_verdict_is_downgraded_not_upgraded() -> None:
    verdict = clean_verdict(
        verdict="APPROVE",
        rationale="SQL and FAISS evidence is available, but the graph source failed.",
        missing_evidence=[GRAPH_ERROR],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=FAILED))

    assert result.passed is False
    assert result.model_verdict == "APPROVE"
    assert result.judge_verdict == "FLAG"
    assert "retrieval_failed" in result.triggered_rules


def test_missing_decision_is_rejected() -> None:
    result = judge_verdict(None, make_retrieval())

    assert result.passed is False
    assert result.model_verdict is None
    assert result.judge_verdict == "FLAG"
    assert result.triggered_rules == ("decision_invalid",)


def test_partial_retrieval_is_accepted_when_acknowledged() -> None:
    verdict = clean_verdict(
        rationale="SQL and FAISS evidence is available, but the ownership graph failed.",
        missing_evidence=[GRAPH_ERROR],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=PARTIAL))

    assert result.passed is True
    assert result.judge_verdict == "FLAG"


def test_partial_retrieval_without_acknowledgement_is_rejected() -> None:
    result = judge_verdict(clean_verdict(), make_retrieval(statuses=PARTIAL))

    assert result.passed is False
    assert result.triggered_rules == ("unacknowledged_failed_source",)
    assert "graph" in result.reasons[0]


def test_graph_failure_acknowledged_by_its_source_label_passes() -> None:
    verdict = clean_verdict(
        rationale="SQL and FAISS evidence is present, but ownership evidence is missing.",
        missing_evidence=["[GRAPH] ownership evidence could not be retrieved"],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=PARTIAL))

    assert result.passed is True
    assert result.triggered_rules == ()
    assert result.judge_verdict == "FLAG"


def test_graph_failure_with_only_an_unrelated_missing_item_is_rejected() -> None:
    verdict = clean_verdict(
        rationale="SQL and FAISS evidence is present, but FY2024 metrics were absent.",
        missing_evidence=["[FAISS] FY2024 contract text was not available"],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=PARTIAL))

    assert result.passed is False
    assert result.triggered_rules == ("unacknowledged_failed_source",)
    assert result.reasons[0].endswith("graph")
    assert result.judge_verdict == "FLAG"


def test_acknowledgement_does_not_depend_on_exception_wording() -> None:
    verdict = clean_verdict(
        rationale="SQL and FAISS evidence is present, but one store did not answer.",
        missing_evidence=["the store was unavailable during retrieval"],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=PARTIAL))

    assert result.passed is False
    assert result.triggered_rules == ("unacknowledged_failed_source",)


def test_each_failed_source_needs_its_own_acknowledgement() -> None:
    verdict = clean_verdict(
        rationale="SQL is acknowledged as failed, but the FAISS and graph sources also failed.",
        missing_evidence=[SQL_ERROR],
    )

    result = judge_verdict(
        verdict, make_retrieval(statuses={"sql": "error", "graph": "error"})
    )

    assert result.passed is False
    assert result.triggered_rules == ("unacknowledged_failed_source",)
    assert result.reasons[0].endswith("graph")
    assert "sql" not in result.reasons[0]
    assert result.judge_verdict == "FLAG"


def test_every_failed_source_acknowledged_leaves_only_retrieval_failed() -> None:
    verdict = clean_verdict(
        rationale="SQL, FAISS, and graph all failed, so no evidence supports a decision.",
        missing_evidence=[SQL_ERROR, FAISS_ERROR, GRAPH_ERROR],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=FAILED))

    assert result.passed is False
    assert result.triggered_rules == ("retrieval_failed",)
    assert result.judge_verdict == "FLAG"


def test_total_failure_with_every_source_acknowledged_downgrades_approve() -> None:
    verdict = clean_verdict(
        verdict="APPROVE",
        rationale=APPROVE_RATIONALE,
        missing_evidence=[SQL_ERROR, FAISS_ERROR, GRAPH_ERROR],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=FAILED))

    assert result.passed is False
    assert result.model_verdict == "APPROVE"
    assert result.judge_verdict == "FLAG"
    assert "unacknowledged_failed_source" not in result.triggered_rules
    assert {"retrieval_failed", "no_evidence_for_approval"} <= set(
        result.triggered_rules
    )


def test_failed_retrieval_is_never_accepted() -> None:
    verdict = clean_verdict(
        rationale="All three retrieval sources failed, so no evidence supports a decision.",
        missing_evidence=["sql failed", "faiss failed", "graph failed"],
    )

    result = judge_verdict(verdict, make_retrieval(statuses=FAILED))

    assert result.passed is False
    assert "retrieval_failed" in result.triggered_rules


def test_approval_without_any_evidence_is_rejected() -> None:
    verdict = clean_verdict(
        verdict="APPROVE",
        rationale="All three stores returned no documents, which supports approval.",
    )

    result = judge_verdict(verdict, make_retrieval(statuses=EMPTY))

    assert result.passed is False
    assert "no_evidence_for_approval" in result.triggered_rules
    assert result.judge_verdict == "FLAG"


def test_supplier_identity_mismatch_is_rejected() -> None:
    verdict = clean_verdict(supplier_id="SUP-002")

    result = judge_verdict(verdict, make_retrieval("SUP-002"), requested_supplier_id="SUP-001")

    assert result.passed is False
    assert result.triggered_rules == ("supplier_identity_mismatch",)


def test_judge_payload_rejects_an_invalid_verdict_value() -> None:
    payload = {
        "supplier_id": "SUP-001",
        "verdict": "MAYBE",
        "rationale": "The evidence is inconclusive and should be reviewed.",
        "findings": ["[SQL] defect_rate=0.0291"],
        "missing_evidence": [],
    }

    result = judge_payload(payload, make_retrieval())

    assert result.passed is False
    assert result.model_verdict is None
    assert result.triggered_rules == ("decision_invalid",)


def test_judge_payload_reports_identity_mismatch() -> None:
    payload = {
        "supplier_id": "SUP-002",
        "verdict": "FLAG",
        "rationale": "The evidence for this supplier shows a HIGH severity complaint.",
        "findings": ["[FAISS] complaint CMP-002-01 severity=HIGH"],
        "missing_evidence": [],
    }

    result = judge_payload(
        payload, make_retrieval("SUP-002"), requested_supplier_id="SUP-001"
    )

    assert result.passed is False
    assert result.model_verdict == "FLAG"
    assert result.triggered_rules == ("supplier_identity_mismatch",)


def test_judge_is_repeatable_for_identical_inputs() -> None:
    retrieval = make_retrieval(statuses=PARTIAL)
    verdict = clean_verdict(missing_evidence=[GRAPH_ERROR])

    first = judge_verdict(verdict, retrieval, requested_supplier_id="SUP-001")
    second = judge_verdict(verdict, retrieval, requested_supplier_id="SUP-001")

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_judge_result_serialization_is_deterministic() -> None:
    result = judge_verdict(clean_verdict(), make_retrieval())

    assert result.to_dict() == {
        "passed": True,
        "model_verdict": "FLAG",
        "judge_verdict": "FLAG",
        "triggered_rules": [],
        "reasons": [],
    }


def test_judge_requires_a_validated_verdict() -> None:
    with pytest.raises(TypeError, match="SupplierRiskVerdict"):
        judge_verdict({"verdict": "FLAG"}, make_retrieval())  # type: ignore[arg-type]


def test_judge_module_does_not_depend_on_dspy() -> None:
    assert not any(name.startswith("dspy") for name in vars(judge_module))


def test_judge_never_calls_an_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        dspy.LM, "__call__", lambda *args, **kwargs: calls.append(args), raising=False
    )

    result = judge_verdict(clean_verdict(), make_retrieval())

    assert result.passed is True
    assert calls == []
