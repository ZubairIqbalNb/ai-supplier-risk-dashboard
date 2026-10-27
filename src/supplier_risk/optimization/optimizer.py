"""The Module 7 DSPy ``BootstrapFewShot`` optimization workflow.

The optimizer is the assignment's required ``dspy.BootstrapFewShot``. This module
builds it from explicit, validated :class:`OptimizerSettings`, runs it under the
Module 6 teacher/student roles without mutating global DSPy configuration, and
returns the compiled student.

Teacher/student roles
---------------------
The teacher LM is supplied to the optimizer through ``teacher_settings`` so only
the bootstrap demonstrations come from the stronger model. The whole compilation
runs inside ``dspy.context(lm=student)`` so the program that is actually optimized
uses the student LM. Global ``dspy.configure`` is never called, so configuration
cannot leak between runs or tests.

Demonstration limits
--------------------
``max_labeled_demos`` defaults to ``0``. BootstrapFewShot would otherwise seed the
prompt with raw training examples, and this project's examples carry no gold
free-text rationale or findings (the Module 2 dataset has no verdict labels at
all; see :mod:`supplier_risk.optimization.labels`). Seeding with label-less
examples would only add "field not supplied" placeholders. With ``0``, every
demonstration is *bootstrapped*: the teacher runs on a training input and the
trace is kept only when the deterministic judge metric accepts it. Raise the
value deliberately if a future dataset supplies complete labeled outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import dspy
from pydantic import BaseModel, ConfigDict, Field

from supplier_risk.decision.errors import sanitize_error_message
from supplier_risk.decision.lm import RoleLanguageModels
from supplier_risk.optimization.errors import CompilationError, OptimizationConfigError
from supplier_risk.optimization.metric import Metric

DEFAULT_OPTIMIZER_NAME: Final = "BootstrapFewShot"


class OptimizerSettings(BaseModel):
    """Validated, explicit settings for the BootstrapFewShot workflow.

    Defaults are safe and offline-friendly. ``seed`` is recorded for
    reproducibility of this project's own example ordering and limits;
    ``dspy.BootstrapFewShot`` in DSPy 3.x shuffles its own validation split with a
    fixed ``random.Random(0)`` and exposes no seed argument, which is noted in the
    compiled artifact metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_bootstrapped_demos: int = Field(default=4, ge=0)
    max_labeled_demos: int = Field(default=0, ge=0)
    max_rounds: int = Field(default=1, ge=1)
    max_errors: int | None = Field(default=None, ge=1)
    metric_threshold: float | None = Field(default=None, ge=0.0)
    train_limit: int | None = Field(default=None, ge=1)
    dev_limit: int | None = Field(default=None, ge=1)
    seed: int = Field(default=0, ge=0)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible mapping of the settings."""
        return self.model_dump(mode="json")


def build_bootstrap_optimizer(
    metric: Metric,
    settings: OptimizerSettings,
    lms: RoleLanguageModels,
) -> dspy.BootstrapFewShot:
    """Build the real ``dspy.BootstrapFewShot`` optimizer for this project.

    Args:
        metric: The judge-backed metric; passed through to the optimizer.
        settings: Validated optimizer settings.
        lms: The teacher/student LMs. The teacher LM is handed to the optimizer
            via ``teacher_settings`` and is never used as the student.
    """
    return dspy.BootstrapFewShot(
        metric=metric,
        metric_threshold=settings.metric_threshold,
        teacher_settings={"lm": lms.teacher},
        max_bootstrapped_demos=settings.max_bootstrapped_demos,
        max_labeled_demos=settings.max_labeled_demos,
        max_rounds=settings.max_rounds,
        max_errors=settings.max_errors,
    )


def optimize_student(
    *,
    trainset: Sequence[dspy.Example],
    metric: Metric,
    student: dspy.Module,
    teacher: dspy.Module,
    lms: RoleLanguageModels,
    settings: OptimizerSettings | None = None,
    optimizer: dspy.BootstrapFewShot | None = None,
) -> dspy.Module:
    """Compile the student with BootstrapFewShot over the training examples.

    Args:
        trainset: The training-partition examples. Only these examples may reach
            the optimizer; the held-out partition must never be passed here.
        metric: The judge-backed metric.
        student: The uncompiled student program.
        teacher: The teacher program with the same structure as the student.
        lms: Teacher/student LMs.
        settings: Optional optimizer settings; defaults are used when omitted.
        optimizer: Optional pre-built optimizer. Tests inject a recording fake;
            production builds the real ``dspy.BootstrapFewShot``.

    Returns:
        The compiled student program. It is configured to run with the student
        LM when called inside ``dspy.context(lm=lms.student)``; the teacher LM is
        used only while compiling.

    Raises:
        OptimizationConfigError: if ``trainset`` is empty.
        CompilationError: if the optimizer raises or returns no program. The
            provider message is sanitized; credentials are never logged.
    """
    if not trainset:
        raise OptimizationConfigError(
            "the training set is empty; BootstrapFewShot needs at least one example"
        )
    active_settings = OptimizerSettings() if settings is None else settings
    active_optimizer = (
        build_bootstrap_optimizer(metric, active_settings, lms)
        if optimizer is None
        else optimizer
    )

    try:
        with dspy.context(lm=lms.student):
            compiled = active_optimizer.compile(
                student=student, teacher=teacher, trainset=list(trainset)
            )
    except Exception as error:  # noqa: BLE001 - one optimizer failure, one project error
        raise CompilationError(
            f"BootstrapFewShot compilation failed ({type(error).__name__}): "
            f"{sanitize_error_message(str(error))}"
        ) from None

    if compiled is None:
        raise CompilationError("BootstrapFewShot returned no compiled program")
    return compiled
