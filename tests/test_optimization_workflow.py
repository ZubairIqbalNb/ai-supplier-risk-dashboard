"""End-to-end workflow test for Module 7 (offline).

The real example builder, metric, judge, evaluation, comparison, and persistence
run here. Only the retrieval engine (a deterministic double) and the optimizer's
*interface* (a recording double) are substituted, so the test exercises this
project's orchestration without a model or store.
"""

from __future__ import annotations

from pathlib import Path

from supplier_risk.data.generate import build_dataset
from supplier_risk.decision.lm import RoleLanguageModels
from supplier_risk.optimization.cli import run_workflow
from supplier_risk.optimization.optimizer import OptimizerSettings
from supplier_risk.optimization.persistence import read_artifact_metadata
from tests.optimization_support import RecordingOptimizer, dummy_lm, retrieval_for

DATASET = build_dataset()


def _run(tmp_path: Path, **overrides: object) -> tuple[object, RecordingOptimizer]:
    lm = dummy_lm(DATASET)
    lms = RoleLanguageModels(teacher=lm, student=lm)
    optimizer = RecordingOptimizer()
    settings = OptimizerSettings(train_limit=3, dev_limit=2)
    result = run_workflow(
        dataset=DATASET,
        lms=lms,
        settings=settings,
        retrieve=retrieval_for,
        artifact_dir=tmp_path,
        optimizer=optimizer,  # type: ignore[arg-type]
        held_out_limit=2,
        compiled_at="2026-10-27T00:00:00+00:00",
        **overrides,  # type: ignore[arg-type]
    )
    return result, optimizer


def test_workflow_produces_comparable_baseline_and_optimized_reports(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)

    assert result.baseline.total == 2
    assert result.optimized.total == 2
    assert result.baseline.partition == "held_out"
    assert result.optimized.partition == "held_out"
    assert result.baseline.accuracy == 1.0
    assert result.optimized.accuracy == 1.0
    assert result.comparison.outcome == "unchanged"


def test_workflow_evaluates_the_same_held_out_examples_for_both(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)

    baseline_ids = [outcome.supplier_id for outcome in result.baseline.outcomes]
    optimized_ids = [outcome.supplier_id for outcome in result.optimized.outcomes]
    assert baseline_ids == optimized_ids == ["SUP-019", "SUP-020"]
    assert result.held_out_supplier_ids == ("SUP-019", "SUP-020")


def test_optimizer_receives_only_training_examples(tmp_path: Path) -> None:
    result, optimizer = _run(tmp_path)

    assert len(optimizer.calls) == 1
    trainset = optimizer.calls[0]["trainset"]
    assert len(trainset) == 3
    contexts = [example.sql_context for example in trainset]  # type: ignore[attr-defined]
    assert all("SUP-019" not in context for context in contexts)
    assert all("SUP-020" not in context for context in contexts)
    assert result.train_supplier_ids == ("SUP-001", "SUP-002", "SUP-003")


def test_workflow_writes_artifact_and_metadata(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)

    assert result.artifact.program_path.exists()
    metadata = read_artifact_metadata(result.artifact.metadata_path)
    assert metadata.compiled_at == "2026-10-27T00:00:00+00:00"
    assert metadata.partitions["train"] == ["SUP-001", "SUP-002", "SUP-003"]
    assert metadata.partitions["held_out"] == ["SUP-019", "SUP-020"]
    assert metadata.metrics is not None
    assert metadata.metrics["outcome"] == "unchanged"


def test_workflow_dev_comparison_can_be_disabled(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, evaluate_dev=False)

    assert result.dev_comparison is None
