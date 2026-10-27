"""Metric tests for Module 7.

The metric wraps the real Module 6 judge. These tests assert the exact scoring
policy, validity rejection, the hard content rule, failed-source acknowledgement,
determinism, and that no language model is ever called.
"""

from __future__ import annotations

import dspy
import pytest

from supplier_risk.decision.verdict import Verdict
from supplier_risk.optimization.errors import OptimizationError
from supplier_risk.optimization.examples import LabeledDecisionExample
from supplier_risk.optimization.metric import (
    SCORE_CORRECT,
    SCORE_OTHERWISE,
    build_decision_metric,
    evaluate_decision,
)
from tests.optimization_support import prediction, retrieval_for


def _gold(
    supplier_id: str = "SUP-001",
    expected: Verdict = "FLAG",
    *,
    statuses: dict[str, str] | None = None,
) -> object:
    retrieval = retrieval_for(supplier_id, statuses)  # type: ignore[arg-type]
    example = LabeledDecisionExample(
        supplier_id=supplier_id,
        partition="train",
        expected_verdict=expected,
        retrieval=retrieval,
    )
    return example.to_dspy_example()


def test_correct_valid_prediction_scores_one() -> None:
    result = evaluate_decision(_gold(), prediction(verdict="FLAG"))

    assert result.score == SCORE_CORRECT
    assert result.correct is True
    assert result.accepted is True
    assert result.structurally_valid is True
    assert result.predicted_verdict == "FLAG"


def test_incorrect_verdict_scores_zero() -> None:
    result = evaluate_decision(_gold(), prediction(verdict="APPROVE"))

    assert result.score == SCORE_OTHERWISE
    assert result.correct is False
    assert result.accepted is True
    assert result.predicted_verdict == "APPROVE"


def test_free_form_text_is_rejected() -> None:
    result = evaluate_decision(_gold(), "the supplier looks fine to me")

    assert result.score == SCORE_OTHERWISE
    assert result.structurally_valid is False
    assert result.predicted_verdict is None
    assert result.triggered_rules == ("decision_invalid",)


def test_missing_structured_field_is_rejected() -> None:
    incomplete = dspy.Prediction(supplier_id="SUP-001", verdict="FLAG")

    result = evaluate_decision(_gold(), incomplete)

    assert result.score == SCORE_OTHERWISE
    assert result.structurally_valid is False


def test_invalid_verdict_value_is_rejected() -> None:
    result = evaluate_decision(_gold(), prediction(verdict="MAYBE"))

    assert result.score == SCORE_OTHERWISE
    assert result.structurally_valid is False


def test_supplier_identity_mismatch_is_rejected() -> None:
    result = evaluate_decision(_gold("SUP-001"), prediction(supplier_id="SUP-002"))

    assert result.score == SCORE_OTHERWISE
    assert result.triggered_rules == ("supplier_identity_mismatch",)


def test_banned_content_is_rejected_even_when_the_verdict_matches() -> None:
    bad = prediction(
        verdict="FLAG",
        rationale="This supplier has no risk and delivery is guaranteed.",
    )

    result = evaluate_decision(_gold(expected="FLAG"), bad)

    assert result.score == SCORE_OTHERWISE
    assert result.accepted is False
    assert "banned_content" in result.triggered_rules
    assert result.structurally_valid is True


def test_unacknowledged_failed_source_is_rejected() -> None:
    result = evaluate_decision(
        _gold(statuses={"graph": "error"}), prediction(verdict="FLAG")
    )

    assert result.score == SCORE_OTHERWISE
    assert result.triggered_rules == ("unacknowledged_failed_source",)


def test_acknowledged_failed_source_is_accepted() -> None:
    acknowledged = prediction(
        verdict="FLAG",
        missing_evidence=["[GRAPH ERROR] RuntimeError: store unavailable"],
    )

    result = evaluate_decision(_gold(statuses={"graph": "error"}), acknowledged)

    assert result.score == SCORE_CORRECT
    assert result.accepted is True


def test_approval_without_evidence_is_not_rewarded_for_matching_the_label() -> None:
    empty = {"sql": "empty", "faiss": "empty", "graph": "empty"}
    result = evaluate_decision(
        _gold("SUP-006", expected="APPROVE", statuses=empty),
        prediction(supplier_id="SUP-006", verdict="APPROVE"),
    )

    assert result.predicted_verdict == "APPROVE"
    assert result.score == SCORE_OTHERWISE
    assert "no_evidence_for_approval" in result.triggered_rules


def test_failed_retrieval_is_never_accepted() -> None:
    failed = {"sql": "error", "faiss": "error", "graph": "error"}
    acknowledged = prediction(
        verdict="FLAG",
        missing_evidence=[
            "[SQL ERROR] x",
            "[FAISS ERROR] x",
            "[GRAPH ERROR] x",
        ],
    )

    result = evaluate_decision(_gold(statuses=failed), acknowledged)

    assert result.score == SCORE_OTHERWISE
    assert "retrieval_failed" in result.triggered_rules


def test_metric_callable_returns_the_same_score() -> None:
    metric = build_decision_metric()

    assert metric(_gold(), prediction(verdict="FLAG"), None) == SCORE_CORRECT
    assert metric(_gold(), prediction(verdict="APPROVE")) == SCORE_OTHERWISE


def test_metric_is_deterministic() -> None:
    gold = _gold()
    pred = prediction(verdict="APPROVE")

    first = evaluate_decision(gold, pred)
    second = evaluate_decision(gold, pred)

    assert first == second


def test_metric_requires_a_well_formed_gold_example() -> None:
    with pytest.raises(OptimizationError, match="expected_verdict"):
        evaluate_decision(dspy.Example(supplier_id="SUP-001"), prediction())


def test_metric_requires_real_retrieval_evidence() -> None:
    gold = dspy.Example(supplier_id="SUP-001", expected_verdict="FLAG")

    with pytest.raises(OptimizationError, match="OrchestrationResult"):
        evaluate_decision(gold, prediction())


def test_metric_never_calls_a_language_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        dspy.LM, "__call__", lambda *args, **kwargs: calls.append(args), raising=False
    )

    evaluate_decision(_gold(), prediction(verdict="FLAG"))

    assert calls == []
