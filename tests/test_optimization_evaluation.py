"""Evaluation and comparison tests for Module 7.

Everything is offline: the examples carry the Module 5 result shape and the
programs are local callables. The metric and judge are the real ones.
"""

from __future__ import annotations

import json
import re

import dspy
import pytest

from supplier_risk.optimization.errors import ComparisonError
from supplier_risk.optimization.evaluation import (
    ClassMetrics,
    class_metrics,
    compare_reports,
    evaluate_program,
)
from supplier_risk.optimization.examples import ExampleSet, LabeledDecisionExample
from tests.optimization_support import prediction, retrieval_for

SUPPLIER_PATTERN = re.compile(r"supplier=(SUP-\d{3})")


def _examples_in(partition: str, *specs: tuple[str, str]) -> ExampleSet:
    return ExampleSet(
        partition=partition,  # type: ignore[arg-type]
        examples=tuple(
            LabeledDecisionExample(
                supplier_id=supplier_id,
                partition=partition,  # type: ignore[arg-type]
                expected_verdict=verdict,  # type: ignore[arg-type]
                retrieval=retrieval_for(supplier_id),
            )
            for supplier_id, verdict in specs
        ),
    )


def _examples(*specs: tuple[str, str]) -> ExampleSet:
    return _examples_in("held_out", *specs)


def _program(verdicts: dict[str, str], *, default: str = "FLAG") -> object:
    def call(*, sql_context: str, faiss_context: str, graph_context: str) -> object:
        match = SUPPLIER_PATTERN.search(sql_context)
        assert match is not None, sql_context
        supplier_id = match.group(1)
        return prediction(supplier_id=supplier_id, verdict=verdicts.get(supplier_id, default))

    return call


# --- class metrics -----------------------------------------------------------------


def test_class_metrics_are_hand_computable() -> None:
    metrics = class_metrics("APPROVE", ["APPROVE", "APPROVE", "FLAG"], ["APPROVE", "FLAG", None])

    assert metrics.support == 2
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)


def test_precision_is_none_when_the_class_is_never_predicted() -> None:
    metrics = class_metrics("APPROVE", ["APPROVE"], [None])

    assert metrics.precision is None
    assert metrics.recall == 0.0
    assert metrics.f1 is None


def test_recall_is_none_when_the_label_never_occurs() -> None:
    metrics = class_metrics("APPROVE", ["FLAG"], ["FLAG"])

    assert metrics.support == 0
    assert metrics.recall is None
    assert metrics.f1 is None


def test_f1_is_zero_when_both_are_defined_and_zero() -> None:
    metrics = class_metrics("APPROVE", ["FLAG", "FLAG", "APPROVE"], ["APPROVE", "APPROVE", "FLAG"])

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_class_metrics_serialization_is_deterministic() -> None:
    metrics = class_metrics("FLAG", ["FLAG"], ["FLAG"])
    assert metrics.to_dict() == ClassMetrics(**metrics.to_dict()).to_dict()  # type: ignore[arg-type]


# --- program evaluation ------------------------------------------------------------


def test_perfect_program_scores_full_accuracy() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-006", "APPROVE"))
    program = _program({"SUP-001": "FLAG", "SUP-006": "APPROVE"})

    report = evaluate_program(program, example_set, name="baseline")

    assert report.total == 2
    assert report.correct == 2
    assert report.incorrect == 0
    assert report.accuracy == 1.0
    assert report.judge_rejections == 0
    assert report.mean_score == 1.0
    assert report.approve.precision == 1.0
    assert report.flag.precision == 1.0


def test_wrong_verdicts_reduce_accuracy() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-006", "APPROVE"))
    program = _program({"SUP-001": "APPROVE", "SUP-006": "APPROVE"})

    report = evaluate_program(program, example_set, name="baseline")

    assert report.correct == 1
    assert report.accuracy == 0.5
    assert report.flag.recall == 0.0


def test_judge_rejection_is_reported_and_not_counted_as_correct() -> None:
    example_set = _examples(("SUP-001", "FLAG"))

    def program(*, sql_context: str, faiss_context: str, graph_context: str) -> object:
        return prediction(
            supplier_id="SUP-001",
            verdict="FLAG",
            rationale="There is no risk and delivery is guaranteed for this supplier.",
        )

    report = evaluate_program(program, example_set, name="baseline")

    assert report.judge_rejections == 1
    assert report.correct == 0
    assert report.structurally_valid == 1
    assert "banned_content" in report.outcomes[0].triggered_rules


def test_program_exception_is_recorded_as_invalid() -> None:
    example_set = _examples(("SUP-001", "FLAG"))

    def program(*, sql_context: str, faiss_context: str, graph_context: str) -> object:
        raise RuntimeError("GEMINI_API_KEY=sk-secret-token; provider down")

    report = evaluate_program(program, example_set, name="baseline")

    assert report.invalid == 1
    assert report.accuracy == 0.0
    outcome = report.outcomes[0]
    assert outcome.error is not None
    assert "sk-secret-token" not in outcome.error
    assert "GEMINI_API_KEY=***" in outcome.error


def test_invalid_prediction_counts_against_recall() -> None:
    example_set = _examples(("SUP-001", "FLAG"))

    def program(*, sql_context: str, faiss_context: str, graph_context: str) -> object:
        return prediction(supplier_id="SUP-999", verdict="FLAG")

    report = evaluate_program(program, example_set, name="baseline")

    assert report.accepted == 0
    assert report.flag.recall == 0.0


# --- macro-F1 policy -------------------------------------------------------------


def test_macro_f1_is_the_mean_of_both_defined_class_f1_values() -> None:
    example_set = _examples(
        ("SUP-001", "FLAG"),
        ("SUP-002", "FLAG"),
        ("SUP-003", "FLAG"),
        ("SUP-006", "APPROVE"),
    )
    program = _program({"SUP-003": "APPROVE", "SUP-006": "APPROVE"})

    report = evaluate_program(program, example_set, name="baseline")

    assert report.approve.precision == 0.5
    assert report.approve.recall == 1.0
    assert report.approve.f1 == pytest.approx(2 / 3)
    assert report.flag.precision == 1.0
    assert report.flag.recall == pytest.approx(2 / 3)
    assert report.flag.f1 == pytest.approx(0.8)
    assert report.macro_f1 == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_is_none_when_approve_f1_is_undefined() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-002", "FLAG"))
    report = evaluate_program(_program({}), example_set, name="baseline")

    assert report.approve.f1 is None
    assert report.flag.f1 == 1.0
    assert report.macro_f1 is None


def test_macro_f1_is_none_when_flag_f1_is_undefined() -> None:
    example_set = _examples(("SUP-001", "APPROVE"), ("SUP-002", "APPROVE"))
    report = evaluate_program(_program({}, default="APPROVE"), example_set, name="baseline")

    assert report.approve.f1 == 1.0
    assert report.flag.f1 is None
    assert report.macro_f1 is None


def test_macro_f1_is_none_when_both_class_f1_values_are_undefined() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-002", "FLAG"))
    report = evaluate_program(_program({}, default="APPROVE"), example_set, name="baseline")

    assert report.approve.f1 is None
    assert report.flag.f1 is None
    assert report.macro_f1 is None
    assert report.accuracy == 0.0


def test_undefined_macro_f1_serializes_as_json_null() -> None:
    example_set = _examples(("SUP-001", "FLAG"))
    report = evaluate_program(_program({}), example_set, name="baseline")

    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["macro_f1"] is None
    assert payload["approve"]["f1"] is None
    assert payload["flag"]["f1"] == 1.0


def test_empty_example_set_has_undefined_accuracy() -> None:
    report = evaluate_program(_program({}), ExampleSet(partition="dev", examples=()), name="x")

    assert report.total == 0
    assert report.accuracy is None
    assert report.macro_f1 is None
    assert report.mean_score == 0.0


def test_outcomes_preserve_example_order_and_are_stable() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-006", "APPROVE"))
    program = _program({"SUP-001": "FLAG", "SUP-006": "APPROVE"})

    first = evaluate_program(program, example_set, name="baseline")
    second = evaluate_program(program, example_set, name="baseline")

    assert [outcome.supplier_id for outcome in first.outcomes] == ["SUP-001", "SUP-006"]
    assert first.to_dict() == second.to_dict()


# --- comparison --------------------------------------------------------------------


def test_improved_when_accuracy_increases() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-006", "APPROVE"))
    baseline = evaluate_program(
        _program({"SUP-001": "APPROVE", "SUP-006": "APPROVE"}), example_set, name="baseline"
    )
    optimized = evaluate_program(
        _program({"SUP-001": "FLAG", "SUP-006": "APPROVE"}), example_set, name="optimized"
    )

    comparison = compare_reports(baseline, optimized)

    assert comparison.outcome == "improved"
    assert comparison.delta_for("accuracy").delta == 0.5


def test_regressed_when_accuracy_decreases() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-006", "APPROVE"))
    baseline = evaluate_program(
        _program({"SUP-001": "FLAG", "SUP-006": "APPROVE"}), example_set, name="baseline"
    )
    optimized = evaluate_program(
        _program({"SUP-001": "APPROVE", "SUP-006": "APPROVE"}), example_set, name="optimized"
    )

    comparison = compare_reports(baseline, optimized)

    assert comparison.outcome == "regressed"
    assert comparison.delta_for("accuracy").delta == -0.5


def test_unchanged_when_accuracy_is_equal() -> None:
    example_set = _examples(("SUP-001", "FLAG"))
    baseline = evaluate_program(_program({"SUP-001": "FLAG"}), example_set, name="baseline")
    optimized = evaluate_program(_program({"SUP-001": "FLAG"}), example_set, name="optimized")

    comparison = compare_reports(baseline, optimized)

    assert comparison.outcome == "unchanged"
    assert comparison.delta_for("accuracy").delta == 0.0


def test_unchanged_when_accuracy_is_undefined() -> None:
    empty = ExampleSet(partition="held_out", examples=())
    baseline = evaluate_program(_program({}), empty, name="baseline")
    optimized = evaluate_program(_program({}), empty, name="optimized")

    comparison = compare_reports(baseline, optimized)

    assert comparison.outcome == "unchanged"
    assert comparison.delta_for("accuracy").delta is None
    assert any("undefined" in note for note in comparison.notes)


def test_comparison_succeeds_for_the_same_partition_and_supplier_ids() -> None:
    example_set = _examples(("SUP-001", "FLAG"), ("SUP-006", "APPROVE"))
    baseline = evaluate_program(_program({"SUP-001": "APPROVE"}), example_set, name="baseline")
    optimized = evaluate_program(_program({"SUP-001": "FLAG"}), example_set, name="optimized")

    comparison = compare_reports(baseline, optimized)

    assert comparison.outcome == "improved"
    assert comparison.baseline.partition == comparison.optimized.partition == "held_out"


def test_comparison_rejects_reports_covering_different_suppliers() -> None:
    baseline = evaluate_program(
        _program({}), _examples(("SUP-001", "FLAG"), ("SUP-002", "FLAG")), name="baseline"
    )
    optimized = evaluate_program(
        _program({}), _examples(("SUP-002", "FLAG"), ("SUP-003", "FLAG")), name="optimized"
    )

    with pytest.raises(ComparisonError) as info:
        compare_reports(baseline, optimized)

    message = str(info.value)
    assert "different suppliers" in message
    assert "SUP-001" in message
    assert "SUP-003" in message


def test_comparison_rejects_reports_from_different_partitions() -> None:
    baseline = evaluate_program(
        _program({}), _examples_in("held_out", ("SUP-001", "FLAG")), name="baseline"
    )
    optimized = evaluate_program(
        _program({}), _examples_in("dev", ("SUP-001", "FLAG")), name="optimized"
    )

    with pytest.raises(ComparisonError) as info:
        compare_reports(baseline, optimized)

    assert "different partitions" in str(info.value)


def test_comparison_is_order_independent_for_the_same_supplier_ids() -> None:
    baseline = evaluate_program(
        _program({}), _examples(("SUP-001", "FLAG"), ("SUP-006", "APPROVE")), name="baseline"
    )
    optimized = evaluate_program(
        _program({}), _examples(("SUP-006", "APPROVE"), ("SUP-001", "FLAG")), name="optimized"
    )

    comparison = compare_reports(baseline, optimized)

    assert [outcome.supplier_id for outcome in baseline.outcomes] == ["SUP-001", "SUP-006"]
    assert [outcome.supplier_id for outcome in optimized.outcomes] == ["SUP-006", "SUP-001"]
    assert comparison.outcome == "unchanged"


def test_comparison_serialization_is_deterministic() -> None:
    example_set = _examples(("SUP-001", "FLAG"))
    report = evaluate_program(_program({"SUP-001": "FLAG"}), example_set, name="baseline")
    comparison = compare_reports(report, report)

    assert comparison.to_dict() == comparison.to_dict()
    assert set(comparison.to_dict()) == {"outcome", "baseline", "optimized", "deltas", "notes"}


def test_dspy_import_is_not_required_by_pure_metric_math() -> None:
    metrics = class_metrics("FLAG", ["FLAG"], ["FLAG"])
    assert isinstance(metrics, ClassMetrics)
    assert isinstance(dspy.Example(a=1), dspy.Example)
