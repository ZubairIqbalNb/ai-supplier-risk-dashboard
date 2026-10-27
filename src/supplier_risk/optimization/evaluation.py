"""Baseline/optimized evaluation and comparison for Module 7.

Evaluation applies the same judge-backed metric the optimizer used
(:func:`~supplier_risk.optimization.metric.evaluate_decision`) to every example in
a partition and reduces the per-example outcomes into a typed
:class:`EvaluationReport`. The baseline and the optimized student are always
evaluated over the *same* examples, in the same order, so the two reports are
directly comparable.

Undefined metric conventions
----------------------------
Precision, recall, and F1 are undefined when their denominator is zero, and are
reported as ``None`` (JSON ``null``) rather than silently coerced to ``0.0``:

- ``precision = TP / (TP + FP)`` is ``None`` when the model predicted the class
  nowhere.
- ``recall = TP / (TP + FN)`` is ``None`` when the label never occurs.
- ``F1`` is ``None`` when either precision or recall is ``None``; it is ``0.0``
  only when both are defined and sum to zero.
- ``macro_f1`` is the arithmetic mean of the APPROVE and FLAG F1 values. It is
  ``None`` as soon as *either* class F1 is undefined. The defined classes are
  never averaged on their own, so an evaluation that only produces one class can
  never be reported as a two-class macro-F1 (the held-out partition is exactly
  that case: six ``FLAG`` examples and no ``APPROVE``).
- accuracy is ``None`` for an empty example set.

An invalid or judge-rejected prediction is treated as *no prediction*: it can
never be a true positive, it counts against recall for the expected class, and it
counts against accuracy. It is never credited for a verdict it did not
legitimately produce.

Comparison integrity
--------------------
:func:`compare_reports` compares *the same* examples only. It verifies that the
two reports cover the same partition and the same supplier ids before it reports
any outcome, and rejects any other pair with
:class:`~supplier_risk.optimization.errors.ComparisonError`. Report ordering is
not part of the contract: the comparison reads aggregate metrics, so the same
suppliers in a different order are still comparable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from supplier_risk.decision.errors import sanitize_error_message
from supplier_risk.decision.verdict import VERDICT_VALUES, Verdict
from supplier_risk.optimization.errors import ComparisonError
from supplier_risk.optimization.examples import ExampleSet, LabeledDecisionExample
from supplier_risk.optimization.metric import evaluate_decision

Program = Callable[..., object]
ComparisonOutcome = Literal["improved", "regressed", "unchanged"]

CLASS_LABELS: Final[tuple[Verdict, ...]] = VERDICT_VALUES


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return ``numerator / denominator`` or ``None`` when the denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Per-class precision, recall, and F1, with explicit undefined handling."""

    label: str
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def class_metrics(
    label: Verdict, expected: list[Verdict], predicted: list[Verdict | None]
) -> ClassMetrics:
    """Compute precision/recall/F1 for ``label`` over parallel label/prediction lists.

    ``predicted[i]`` is ``None`` when example ``i`` produced no accepted verdict.
    """
    true_positives = false_positives = false_negatives = support = 0
    for expected_verdict, predicted_verdict in zip(expected, predicted, strict=True):
        if expected_verdict == label:
            support += 1
        if predicted_verdict == label and expected_verdict == label:
            true_positives += 1
        elif predicted_verdict == label and expected_verdict != label:
            false_positives += 1
        elif expected_verdict == label and predicted_verdict != label:
            false_negatives += 1

    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    if precision is None or recall is None:
        f1: float | None = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return ClassMetrics(
        label=label,
        support=support,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass(frozen=True, slots=True)
class ExampleOutcome:
    """One auditable, per-example evaluation result."""

    supplier_id: str
    expected_verdict: Verdict
    predicted_verdict: Verdict | None
    judge_verdict: Verdict | None
    structurally_valid: bool
    accepted: bool
    correct: bool
    score: float
    triggered_rules: tuple[str, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "supplier_id": self.supplier_id,
            "expected_verdict": self.expected_verdict,
            "predicted_verdict": self.predicted_verdict,
            "judge_verdict": self.judge_verdict,
            "structurally_valid": self.structurally_valid,
            "accepted": self.accepted,
            "correct": self.correct,
            "score": self.score,
            "triggered_rules": list(self.triggered_rules),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregated metrics plus the per-example outcomes for one program."""

    name: str
    partition: str
    total: int
    structurally_valid: int
    invalid: int
    accepted: int
    correct: int
    incorrect: int
    judge_rejections: int
    accuracy: float | None
    mean_score: float
    approve: ClassMetrics
    flag: ClassMetrics
    macro_f1: float | None
    outcomes: tuple[ExampleOutcome, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "partition": self.partition,
            "total": self.total,
            "structurally_valid": self.structurally_valid,
            "invalid": self.invalid,
            "accepted": self.accepted,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "judge_rejections": self.judge_rejections,
            "accuracy": self.accuracy,
            "mean_score": self.mean_score,
            "approve": self.approve.to_dict(),
            "flag": self.flag.to_dict(),
            "macro_f1": self.macro_f1,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


def _outcome_for(example: LabeledDecisionExample, program: Program) -> ExampleOutcome:
    retrieval = example.retrieval
    try:
        prediction = program(
            sql_context=retrieval.sql.context,
            faiss_context=retrieval.faiss.context,
            graph_context=retrieval.graph.context,
        )
    except Exception as error:  # noqa: BLE001 - a failed call is an invalid prediction
        return ExampleOutcome(
            supplier_id=example.supplier_id,
            expected_verdict=example.expected_verdict,
            predicted_verdict=None,
            judge_verdict=None,
            structurally_valid=False,
            accepted=False,
            correct=False,
            score=0.0,
            triggered_rules=(),
            error=f"{type(error).__name__}: {sanitize_error_message(str(error))}",
        )

    result = evaluate_decision(example.to_dspy_example(), prediction)
    return ExampleOutcome(
        supplier_id=example.supplier_id,
        expected_verdict=example.expected_verdict,
        predicted_verdict=result.predicted_verdict,
        judge_verdict=result.judge_verdict,
        structurally_valid=result.structurally_valid,
        accepted=result.accepted,
        correct=result.correct,
        score=result.score,
        triggered_rules=result.triggered_rules,
    )


def evaluate_program(
    program: Program,
    example_set: ExampleSet,
    *,
    name: str,
) -> EvaluationReport:
    """Evaluate ``program`` over one example set and return a typed report.

    Args:
        program: A callable invoked as
            ``program(sql_context=..., faiss_context=..., graph_context=...)``.
            A :class:`~supplier_risk.decision.module.VerdictGenerator` (compiled or
            not) satisfies this; offline tests inject a fake.
        example_set: The examples to run over, in stable order. The same set is
            used for the baseline and the optimized program.
        name: Label for the report, for example ``"baseline"`` or ``"optimized"``.

    Returns:
        An :class:`EvaluationReport`.
    """
    outcomes = tuple(_outcome_for(example, program) for example in example_set.examples)

    expected = [outcome.expected_verdict for outcome in outcomes]
    predicted: list[Verdict | None] = [
        outcome.predicted_verdict if outcome.accepted else None for outcome in outcomes
    ]
    per_class = {label: class_metrics(label, expected, predicted) for label in CLASS_LABELS}
    approve_metrics = per_class["APPROVE"]
    flag_metrics = per_class["FLAG"]
    approve_f1 = approve_metrics.f1
    flag_f1 = flag_metrics.f1
    # Macro-F1 is the mean of both class F1 values, or undefined when either class
    # is: averaging the defined classes alone would misreport a single-class
    # evaluation as a two-class score.
    macro_f1 = (
        (approve_f1 + flag_f1) / 2
        if approve_f1 is not None and flag_f1 is not None
        else None
    )
    total = len(outcomes)
    correct = sum(1 for outcome in outcomes if outcome.correct)

    return EvaluationReport(
        name=name,
        partition=example_set.partition,
        total=total,
        structurally_valid=sum(1 for outcome in outcomes if outcome.structurally_valid),
        invalid=sum(1 for outcome in outcomes if not outcome.structurally_valid),
        accepted=sum(1 for outcome in outcomes if outcome.accepted),
        correct=correct,
        incorrect=total - correct,
        judge_rejections=sum(1 for outcome in outcomes if not outcome.accepted),
        accuracy=_ratio(correct, total),
        mean_score=(sum(outcome.score for outcome in outcomes) / total) if total else 0.0,
        approve=approve_metrics,
        flag=flag_metrics,
        macro_f1=macro_f1,
        outcomes=outcomes,
    )


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """Baseline, optimized, and absolute change for one scalar metric."""

    name: str
    baseline: float | int | None
    optimized: float | int | None
    delta: float | int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "baseline": self.baseline,
            "optimized": self.optimized,
            "delta": self.delta,
        }


def _delta(name: str, baseline: float | None, optimized: float | None) -> MetricDelta:
    if baseline is None or optimized is None:
        return MetricDelta(name=name, baseline=baseline, optimized=optimized, delta=None)
    return MetricDelta(
        name=name, baseline=baseline, optimized=optimized, delta=optimized - baseline
    )


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Baseline vs optimized comparison over the same held-out examples."""

    baseline: EvaluationReport
    optimized: EvaluationReport
    outcome: ComparisonOutcome
    deltas: tuple[MetricDelta, ...]
    notes: tuple[str, ...]

    def delta_for(self, name: str) -> MetricDelta:
        for delta in self.deltas:
            if delta.name == name:
                return delta
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "baseline": self.baseline.to_dict(),
            "optimized": self.optimized.to_dict(),
            "deltas": [delta.to_dict() for delta in self.deltas],
            "notes": list(self.notes),
        }


def require_same_examples(baseline: EvaluationReport, optimized: EvaluationReport) -> None:
    """Reject two reports that do not cover the same partition and suppliers.

    A comparison is only meaningful when both reports were produced over the same
    examples, so this is checked *before* any outcome is derived. Supplier ids are
    compared as sorted multisets: the same suppliers in a different order are
    comparable because the metrics are aggregates, while a different supplier set
    (or a different number of occurrences of one) is not.

    Raises:
        ComparisonError: if the partitions differ, or the supplier id multisets
            differ.
    """
    if baseline.partition != optimized.partition:
        raise ComparisonError(
            "cannot compare reports from different partitions: "
            f"{baseline.partition!r} vs {optimized.partition!r}"
        )

    baseline_ids = sorted(outcome.supplier_id for outcome in baseline.outcomes)
    optimized_ids = sorted(outcome.supplier_id for outcome in optimized.outcomes)
    if baseline_ids != optimized_ids:
        only_baseline = sorted(set(baseline_ids) - set(optimized_ids))
        only_optimized = sorted(set(optimized_ids) - set(baseline_ids))
        raise ComparisonError(
            "cannot compare reports covering different suppliers: "
            f"baseline={baseline_ids} optimized={optimized_ids} "
            f"only_in_baseline={only_baseline} only_in_optimized={only_optimized}"
        )


def compare_reports(baseline: EvaluationReport, optimized: EvaluationReport) -> ComparisonReport:
    """Compare two reports over the same examples.

    Both reports must cover the same partition and the same supplier ids; see
    :func:`require_same_examples`. A pair that does not is rejected with
    :class:`~supplier_risk.optimization.errors.ComparisonError` rather than being
    reported as ``improved``, ``regressed``, or ``unchanged``, because a
    difference in the evaluated examples is not evidence about the program.

    The decision is driven by overall accuracy, the assignment's headline metric.
    A positive change is ``improved``, a negative change is ``regressed``, and no
    change is ``unchanged``. When either accuracy is undefined (an empty set),
    the outcome is ``unchanged`` with an explanatory note. The comparison never
    asserts that optimization helps; it reports what the measurements show.
    """
    require_same_examples(baseline, optimized)

    notes: list[str] = []
    baseline_accuracy = baseline.accuracy
    optimized_accuracy = optimized.accuracy
    if baseline_accuracy is None or optimized_accuracy is None:
        outcome: ComparisonOutcome = "unchanged"
        notes.append("accuracy is undefined for an empty example set")
    elif optimized_accuracy > baseline_accuracy:
        outcome = "improved"
    elif optimized_accuracy < baseline_accuracy:
        outcome = "regressed"
    else:
        outcome = "unchanged"

    deltas = (
        _delta("accuracy", baseline.accuracy, optimized.accuracy),
        _delta("macro_f1", baseline.macro_f1, optimized.macro_f1),
        _delta("approve_f1", baseline.approve.f1, optimized.approve.f1),
        _delta("flag_f1", baseline.flag.f1, optimized.flag.f1),
        _delta("mean_score", baseline.mean_score, optimized.mean_score),
        _delta("judge_rejections", baseline.judge_rejections, optimized.judge_rejections),
        _delta("correct", baseline.correct, optimized.correct),
    )
    return ComparisonReport(
        baseline=baseline,
        optimized=optimized,
        outcome=outcome,
        deltas=deltas,
        notes=tuple(notes),
    )
