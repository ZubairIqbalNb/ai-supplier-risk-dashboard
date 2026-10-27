"""The deterministic, judge-backed optimization metric for Module 7.

The metric is the only signal the DSPy optimizer sees, so it must not be a
rubber stamp. It reuses the Module 6 deterministic judge as the authority for
validity and never calls a language model, so the same prediction and the same
gold example always produce the same score.

Scoring policy (documented, binary, and independent of any model call):

- ``1.0`` — the prediction parses into a :class:`SupplierRiskVerdict`, the judge
  accepts it (no rule triggered), and the validated verdict equals the example's
  expected verdict.
- ``0.0`` — anything else: a malformed or missing structured output, a supplier
  identity mismatch, a banned term, an unacknowledged failed source, an approval
  without evidence, or a valid verdict that simply disagrees with the label.

Binary rather than partial credit is deliberate: an invalid output scores zero no
matter what verdict it claims, so the optimizer can never win by emitting a
correct-looking verdict in a malformed or rule-breaking payload. The optimizer
treats any truthy score as a successful demonstration.

This module imports the judge and the DSPy-free verdict parser only. It never
imports a provider SDK and never opens a socket.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from supplier_risk.decision.judge import JudgeResult, judge_payload
from supplier_risk.decision.verdict import VERDICT_VALUES, Verdict
from supplier_risk.optimization.errors import OptimizationError
from supplier_risk.orchestration.result import OrchestrationResult

Metric = Callable[..., float]

SCORE_CORRECT: Final = 1.0
SCORE_OTHERWISE: Final = 0.0


@dataclass(frozen=True, slots=True)
class DecisionMetricResult:
    """One scored prediction, keeping validity, judgment, and correctness apart.

    ``predicted_verdict`` is the *validated* model verdict when the payload parsed
    (``None`` when it did not). ``accepted`` is the judge's decision.
    ``correct`` requires both acceptance and agreement with the label, so a
    rejected or malformed prediction is never counted as correct.
    """

    expected_verdict: Verdict
    predicted_verdict: Verdict | None
    judge_verdict: Verdict | None
    structurally_valid: bool
    accepted: bool
    correct: bool
    score: float
    triggered_rules: tuple[str, ...]
    reasons: tuple[str, ...]


def _field(example: object, name: str) -> object:
    return getattr(example, name, None)


def _expected_verdict(example: object) -> Verdict:
    value = _field(example, "expected_verdict")
    if value not in VERDICT_VALUES:
        raise OptimizationError(
            "the gold example has no valid expected_verdict; build examples with "
            "supplier_risk.optimization.examples.build_example_set"
        )
    return value  # type: ignore[return-value]


def _retrieval(example: object) -> OrchestrationResult:
    value = _field(example, "retrieval")
    if not isinstance(value, OrchestrationResult):
        raise OptimizationError(
            "the gold example has no OrchestrationResult; the judge cannot be "
            "evaluated without the real retrieval evidence"
        )
    return value


def _supplier_id(example: object) -> str:
    value = _field(example, "supplier_id")
    if not isinstance(value, str) or not value.strip():
        raise OptimizationError("the gold example has no supplier_id")
    return value.strip().upper()


def evaluate_decision(example: object, prediction: object) -> DecisionMetricResult:
    """Validate, judge, and score one prediction against one gold example.

    Raises:
        OptimizationError: if the gold example is missing its label, supplier
            identity, or retrieval result. This is a construction bug, not a
            model failure; the optimizer surfaces it instead of scoring it.
    """
    expected = _expected_verdict(example)
    retrieval = _retrieval(example)
    supplier_id = _supplier_id(example)

    judgment: JudgeResult = judge_payload(
        prediction, retrieval, requested_supplier_id=supplier_id
    )
    structurally_valid = judgment.model_verdict is not None
    predicted = judgment.model_verdict
    correct = judgment.passed and predicted == expected
    return DecisionMetricResult(
        expected_verdict=expected,
        predicted_verdict=predicted,
        judge_verdict=judgment.judge_verdict,
        structurally_valid=structurally_valid,
        accepted=judgment.passed,
        correct=correct,
        score=SCORE_CORRECT if correct else SCORE_OTHERWISE,
        triggered_rules=judgment.triggered_rules,
        reasons=judgment.reasons,
    )


def build_decision_metric() -> Metric:
    """Return the DSPy-compatible metric callable.

    The callable accepts the DSPy ``(gold, prediction, trace)`` signature (the
    trace is ignored) and returns :data:`SCORE_CORRECT` or
    :data:`SCORE_OTHERWISE`.
    """

    def metric(example: object, prediction: object, trace: object = None) -> float:
        return evaluate_decision(example, prediction).score

    return metric
