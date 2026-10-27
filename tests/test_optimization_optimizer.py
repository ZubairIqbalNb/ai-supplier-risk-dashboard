"""Optimizer orchestration tests for Module 7.

The optimizer *interface* is faked with a recording double so the tests assert
this project's orchestration (partition selection, roles, error handling) rather
than DSPy internals. One test builds the real ``dspy.BootstrapFewShot`` to prove
the factory wires the required optimizer and configuration.
"""

from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest
from pydantic import ValidationError

from supplier_risk.decision.lm import RoleLanguageModels
from supplier_risk.optimization.errors import CompilationError, OptimizationConfigError
from supplier_risk.optimization.examples import SIGNATURE_INPUTS
from supplier_risk.optimization.metric import build_decision_metric
from supplier_risk.optimization.optimizer import (
    DEFAULT_OPTIMIZER_NAME,
    OptimizerSettings,
    build_bootstrap_optimizer,
    optimize_student,
)
from tests.optimization_support import RecordingOptimizer


def _lms(teacher: str = "gemini/teacher", student: str = "gemini/student") -> RoleLanguageModels:
    return RoleLanguageModels(
        teacher=SimpleNamespace(model=teacher),
        student=SimpleNamespace(model=student),
    )


def _trainset() -> list[dspy.Example]:
    return [
        dspy.Example(
            sql_context="[SQL] a",
            faiss_context="[FAISS] b",
            graph_context="[GRAPH] c",
        ).with_inputs(*SIGNATURE_INPUTS)
    ]


def test_optimizer_settings_defaults_are_safe() -> None:
    settings = OptimizerSettings()

    assert settings.max_bootstrapped_demos == 4
    assert settings.max_labeled_demos == 0
    assert settings.max_rounds == 1
    assert settings.max_errors is None
    assert settings.metric_threshold is None
    assert settings.train_limit is None
    assert settings.dev_limit is None
    assert settings.seed == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_bootstrapped_demos": -1},
        {"max_labeled_demos": -2},
        {"max_rounds": 0},
        {"max_errors": 0},
        {"train_limit": 0},
        {"dev_limit": 0},
        {"seed": -1},
        {"unknown_setting": 1},
    ],
)
def test_invalid_optimizer_settings_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        OptimizerSettings(**kwargs)


def test_settings_serialization_is_deterministic() -> None:
    assert OptimizerSettings(seed=7).to_dict()["seed"] == 7
    assert set(OptimizerSettings().to_dict()) == {
        "max_bootstrapped_demos",
        "max_labeled_demos",
        "max_rounds",
        "max_errors",
        "metric_threshold",
        "train_limit",
        "dev_limit",
        "seed",
    }


def test_build_bootstrap_optimizer_uses_the_real_dspy_optimizer() -> None:
    metric = build_decision_metric()
    settings = OptimizerSettings(max_bootstrapped_demos=2, max_rounds=3, metric_threshold=1.0)
    lms = _lms()

    optimizer = build_bootstrap_optimizer(metric, settings, lms)

    assert isinstance(optimizer, dspy.BootstrapFewShot)
    assert DEFAULT_OPTIMIZER_NAME == "BootstrapFewShot"
    assert optimizer.metric is metric
    assert optimizer.metric_threshold == 1.0
    assert optimizer.max_bootstrapped_demos == 2
    assert optimizer.max_labeled_demos == 0
    assert optimizer.max_rounds == 3


def test_optimizer_teacher_role_is_the_teacher_lm() -> None:
    lms = _lms()

    optimizer = build_bootstrap_optimizer(build_decision_metric(), OptimizerSettings(), lms)

    assert optimizer.teacher_settings == {"lm": lms.teacher}
    assert optimizer.teacher_settings["lm"] is not lms.student


def test_optimize_student_passes_the_trainset_and_roles() -> None:
    optimizer = RecordingOptimizer()
    student = dspy.Module()
    teacher = dspy.Module()
    trainset = _trainset()

    result = optimize_student(
        trainset=trainset,
        metric=build_decision_metric(),
        student=student,
        teacher=teacher,
        lms=_lms(),
        optimizer=optimizer,  # type: ignore[arg-type]
    )

    assert result is student
    assert optimizer.calls == [
        {"student": student, "teacher": teacher, "trainset": trainset}
    ]


def test_optimize_student_never_receives_held_out_examples() -> None:
    optimizer = RecordingOptimizer()
    trainset = _trainset()

    optimize_student(
        trainset=trainset,
        metric=build_decision_metric(),
        student=dspy.Module(),
        teacher=dspy.Module(),
        lms=_lms(),
        optimizer=optimizer,  # type: ignore[arg-type]
    )

    passed = optimizer.calls[0]["trainset"]
    assert passed == trainset
    assert all(example.sql_context == "[SQL] a" for example in passed)  # type: ignore[attr-defined]


def test_empty_trainset_is_rejected() -> None:
    with pytest.raises(OptimizationConfigError, match="training set is empty"):
        optimize_student(
            trainset=[],
            metric=build_decision_metric(),
            student=dspy.Module(),
            teacher=dspy.Module(),
            lms=_lms(),
            optimizer=RecordingOptimizer(),  # type: ignore[arg-type]
        )


def test_compilation_failure_is_wrapped_and_sanitized() -> None:
    optimizer = RecordingOptimizer(error=RuntimeError("GEMINI_API_KEY=sk-live-secret boom"))

    with pytest.raises(CompilationError) as exc_info:
        optimize_student(
            trainset=_trainset(),
            metric=build_decision_metric(),
            student=dspy.Module(),
            teacher=dspy.Module(),
            lms=_lms(),
            optimizer=optimizer,  # type: ignore[arg-type]
        )

    message = str(exc_info.value)
    assert "sk-live-secret" not in message
    assert "GEMINI_API_KEY=***" in message


def test_none_result_is_reported_as_a_compilation_error() -> None:
    optimizer = RecordingOptimizer(result=None)

    class NoneOptimizer:
        def compile(self, **kwargs: object) -> None:
            return None

    with pytest.raises(CompilationError, match="no compiled program"):
        optimize_student(
            trainset=_trainset(),
            metric=build_decision_metric(),
            student=dspy.Module(),
            teacher=dspy.Module(),
            lms=_lms(),
            optimizer=NoneOptimizer(),  # type: ignore[arg-type]
        )
    assert optimizer.calls == []


def test_default_optimizer_is_used_when_none_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    optimizer = RecordingOptimizer()

    def fake_build(metric: object, settings: object, lms: object) -> RecordingOptimizer:
        recorded["metric"] = metric
        recorded["settings"] = settings
        recorded["lms"] = lms
        return optimizer

    monkeypatch.setattr(
        "supplier_risk.optimization.optimizer.build_bootstrap_optimizer", fake_build
    )
    lms = _lms()
    settings = OptimizerSettings(train_limit=2)

    optimize_student(
        trainset=_trainset(),
        metric=build_decision_metric(),
        student=dspy.Module(),
        teacher=dspy.Module(),
        lms=lms,
        settings=settings,
    )

    assert recorded["settings"] is settings
    assert recorded["lms"] is lms
    assert callable(recorded["metric"])
    assert optimizer.calls


def test_roles_are_distinct_objects() -> None:
    student = dspy.Module()
    teacher = dspy.Module()
    optimizer = RecordingOptimizer()

    optimize_student(
        trainset=_trainset(),
        metric=build_decision_metric(),
        student=student,
        teacher=teacher,
        lms=_lms(),
        optimizer=optimizer,  # type: ignore[arg-type]
    )

    call = optimizer.calls[0]
    assert call["student"] is not call["teacher"]
