"""Module 7: DSPy BootstrapFewShot optimization and evaluation.

This package builds a reproducible optimization workflow on top of the Module 6
decision system. It does not redefine the Signature, verdict contract, judge, or
decision pipeline; it reuses them.

Public entry points:

- :func:`build_partition_examples` — turn the Module 2 dataset into labeled DSPy
  examples, one set per partition, using the real Module 5 retrieval.
- :func:`build_decision_metric` — the deterministic, judge-backed metric.
- :func:`evaluate_program`, :func:`compare_reports` — baseline/optimized
  evaluation and comparison.
- :func:`optimize_student` — run the real ``dspy.BootstrapFewShot`` optimizer.
- :func:`save_compiled_student`, :func:`load_compiled_student` — artifact
  persistence and reload.

Importing this package imports DSPy and the Module 3-5 retrieval jobs, but
creates no provider client, reads no environment variable, opens no database, and
loads no embedding model. Nothing here makes a model or network call until a
caller runs an evaluation or the optimizer.
"""

from supplier_risk.optimization.errors import (
    ArtifactError,
    ComparisonError,
    CompilationError,
    ExampleConstructionError,
    OptimizationConfigError,
    OptimizationError,
)
from supplier_risk.optimization.evaluation import (
    ClassMetrics,
    ComparisonReport,
    EvaluationReport,
    ExampleOutcome,
    MetricDelta,
    class_metrics,
    compare_reports,
    evaluate_program,
    require_same_examples,
)
from supplier_risk.optimization.examples import (
    SIGNATURE_INPUTS,
    ExampleSet,
    LabeledDecisionExample,
    build_example_set,
    build_partition_examples,
    dataset_fingerprint,
)
from supplier_risk.optimization.labels import (
    LABEL_POLICY_VERSION,
    expected_verdict,
    label_policy_description,
)
from supplier_risk.optimization.metric import (
    DecisionMetricResult,
    build_decision_metric,
    evaluate_decision,
)
from supplier_risk.optimization.optimizer import (
    DEFAULT_OPTIMIZER_NAME,
    OptimizerSettings,
    build_bootstrap_optimizer,
    optimize_student,
)
from supplier_risk.optimization.persistence import (
    ARTIFACT_VERSION,
    ArtifactMetadata,
    SavedStudentArtifact,
    build_artifact_metadata,
    load_and_verify,
    load_compiled_student,
    read_artifact_metadata,
    save_compiled_student,
)

__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_OPTIMIZER_NAME",
    "LABEL_POLICY_VERSION",
    "SIGNATURE_INPUTS",
    "ArtifactError",
    "ArtifactMetadata",
    "ClassMetrics",
    "ComparisonError",
    "ComparisonReport",
    "CompilationError",
    "DecisionMetricResult",
    "EvaluationReport",
    "ExampleConstructionError",
    "ExampleOutcome",
    "ExampleSet",
    "LabeledDecisionExample",
    "MetricDelta",
    "OptimizationConfigError",
    "OptimizationError",
    "OptimizerSettings",
    "SavedStudentArtifact",
    "build_artifact_metadata",
    "build_bootstrap_optimizer",
    "build_decision_metric",
    "build_example_set",
    "build_partition_examples",
    "class_metrics",
    "compare_reports",
    "dataset_fingerprint",
    "evaluate_decision",
    "evaluate_program",
    "expected_verdict",
    "label_policy_description",
    "load_and_verify",
    "load_compiled_student",
    "optimize_student",
    "read_artifact_metadata",
    "require_same_examples",
    "save_compiled_student",
]
