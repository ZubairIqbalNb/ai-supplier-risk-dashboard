"""Command-line workflow for Module 7 optimization and evaluation.

Run it with::

    uv run --locked python -m supplier_risk.optimization [options]

The command validates configuration before any expensive work, announces every
real language-model call, builds the partition example sets with the real Module
5 retrieval, evaluates the baseline, compiles the student with
``dspy.BootstrapFewShot``, re-evaluates it on the *same* held-out examples, writes
the compiled artifact plus its metadata and the evaluation results, and returns a
nonzero exit status on a configuration, compilation, or persistence failure.

Credentials are never printed: only the role-to-model mapping and non-secret
options are shown.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy

from supplier_risk.config import ConfigError, load_language_model_settings
from supplier_risk.contracts import SupplierDataset
from supplier_risk.data.generate import DEFAULT_SEED, build_dataset
from supplier_risk.decision.lm import RoleLanguageModels, build_role_language_models
from supplier_risk.decision.module import VerdictGenerator
from supplier_risk.optimization.errors import OptimizationError
from supplier_risk.optimization.evaluation import (
    ComparisonReport,
    EvaluationReport,
    compare_reports,
    evaluate_program,
)
from supplier_risk.optimization.examples import (
    Retrieve,
    build_partition_examples,
    dataset_fingerprint,
)
from supplier_risk.optimization.metric import build_decision_metric
from supplier_risk.optimization.optimizer import OptimizerSettings, optimize_student
from supplier_risk.optimization.persistence import (
    DEFAULT_ARTIFACT_DIR,
    ArtifactMetadata,
    SavedStudentArtifact,
    build_artifact_metadata,
    save_compiled_student,
)
from supplier_risk.orchestration.pipeline import query_all_sources

RESULTS_FILENAME = "evaluation_results.json"
ProgramFactory = Callable[[], dspy.Module]


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Everything one optimization run produced, for reporting and tests."""

    baseline: EvaluationReport
    optimized: EvaluationReport
    comparison: ComparisonReport
    dev_comparison: ComparisonReport | None
    metadata: ArtifactMetadata
    artifact: SavedStudentArtifact
    train_supplier_ids: tuple[str, ...]
    held_out_supplier_ids: tuple[str, ...]


def _evaluate(
    program: dspy.Module,
    example_set: Any,
    *,
    name: str,
    lms: RoleLanguageModels,
) -> EvaluationReport:
    with dspy.context(lm=lms.student):
        return evaluate_program(program, example_set, name=name)


def run_workflow(
    *,
    dataset: SupplierDataset,
    lms: RoleLanguageModels,
    settings: OptimizerSettings,
    retrieve: Retrieve | None = None,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    evaluate_dev: bool = True,
    held_out_limit: int | None = None,
    optimizer: dspy.BootstrapFewShot | None = None,
    student_factory: ProgramFactory = VerdictGenerator,
    compiled_at: str | None = None,
) -> WorkflowResult:
    """Run the full baseline -> optimize -> optimized evaluation workflow.

    Args:
        dataset: The validated dataset (in memory; the dataset file is never
            rewritten).
        lms: Teacher/student LMs.
        settings: Optimizer settings, including optional train/dev limits.
        retrieve: Retrieval callable; defaults to the real orchestrator.
        artifact_dir: Where the compiled student, metadata, and results are written.
        evaluate_dev: Also evaluate the development partition (used as a second,
            independent view because the held-out partition is single-class; see
            the Module 7 documentation).
        held_out_limit: Optional cap on held-out examples, in stable order.
        optimizer: Optional pre-built optimizer (tests inject a fake).
        student_factory: Program factory; defaults to the Module 6
            :class:`~supplier_risk.decision.module.VerdictGenerator`.
        compiled_at: Optional fixed compilation timestamp for deterministic tests.
    """
    example_sets = build_partition_examples(
        dataset,
        retrieve=retrieve,
        train_limit=settings.train_limit,
        dev_limit=settings.dev_limit,
        held_out_limit=held_out_limit,
    )
    train_set = example_sets["train"]
    dev_set = example_sets["dev"]
    held_out_set = example_sets["held_out"]

    metric = build_decision_metric()
    baseline = _evaluate(student_factory(), held_out_set, name="baseline", lms=lms)
    optimized_program = optimize_student(
        trainset=train_set.dspy_examples(),
        metric=metric,
        student=student_factory(),
        teacher=student_factory(),
        lms=lms,
        settings=settings,
        optimizer=optimizer,
    )
    optimized = _evaluate(optimized_program, held_out_set, name="optimized", lms=lms)
    comparison = compare_reports(baseline, optimized)

    dev_comparison: ComparisonReport | None = None
    if evaluate_dev:
        dev_baseline = _evaluate(student_factory(), dev_set, name="baseline", lms=lms)
        dev_optimized = _evaluate(optimized_program, dev_set, name="optimized", lms=lms)
        dev_comparison = compare_reports(dev_baseline, dev_optimized)

    partitions = {
        "train": list(train_set.supplier_ids),
        "dev": list(dev_set.supplier_ids),
        "held_out": list(held_out_set.supplier_ids),
    }
    metadata = build_artifact_metadata(
        lms=lms,
        settings=settings,
        dataset=dataset,
        dataset_fingerprint=dataset_fingerprint(dataset),
        partitions=partitions,
        metrics=_metrics_summary(comparison, dev_comparison),
        compiled_at=compiled_at,
    )
    artifact = save_compiled_student(
        optimized_program, metadata, directory=Path(artifact_dir)
    )
    return WorkflowResult(
        baseline=baseline,
        optimized=optimized,
        comparison=comparison,
        dev_comparison=dev_comparison,
        metadata=metadata,
        artifact=artifact,
        train_supplier_ids=train_set.supplier_ids,
        held_out_supplier_ids=held_out_set.supplier_ids,
    )


def _report_summary(report: EvaluationReport) -> dict[str, object]:
    return {
        "partition": report.partition,
        "total": report.total,
        "structurally_valid": report.structurally_valid,
        "invalid": report.invalid,
        "judge_rejections": report.judge_rejections,
        "correct": report.correct,
        "incorrect": report.incorrect,
        "accuracy": report.accuracy,
        "mean_score": report.mean_score,
        "macro_f1": report.macro_f1,
        "approve_precision": report.approve.precision,
        "approve_recall": report.approve.recall,
        "approve_f1": report.approve.f1,
        "flag_precision": report.flag.precision,
        "flag_recall": report.flag.recall,
        "flag_f1": report.flag.f1,
    }


def _metrics_summary(
    comparison: ComparisonReport, dev_comparison: ComparisonReport | None
) -> dict[str, object]:
    summary: dict[str, object] = {
        "outcome": comparison.outcome,
        "held_out": {
            "baseline": _report_summary(comparison.baseline),
            "optimized": _report_summary(comparison.optimized),
        },
    }
    if dev_comparison is not None:
        summary["dev"] = {
            "outcome": dev_comparison.outcome,
            "baseline": _report_summary(dev_comparison.baseline),
            "optimized": _report_summary(dev_comparison.optimized),
        }
    return summary


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m supplier_risk.optimization",
        description="Compile the Module 6 student with DSPy BootstrapFewShot and evaluate it.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Dataset seed.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Artifact directory (default: {DEFAULT_ARTIFACT_DIR}).",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="dotenv file to read.")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration and exit without calling any model or retrieval engine.",
    )
    parser.add_argument(
        "--skip-dev",
        action="store_true",
        help="Do not also evaluate the development partition.",
    )
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    parser.add_argument("--max-labeled-demos", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--held-out-limit", type=int, default=None)
    return parser.parse_args(argv)


def _build_settings(args: argparse.Namespace) -> OptimizerSettings:
    try:
        return OptimizerSettings(
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            max_rounds=args.max_rounds,
            seed=args.optimizer_seed,
            train_limit=args.train_limit,
            dev_limit=args.dev_limit,
        )
    except Exception as error:  # noqa: BLE001 - pydantic validation message is safe
        raise OptimizationError(f"invalid optimizer settings: {type(error).__name__}") from None


def _load_language_models(env_file: Path) -> RoleLanguageModels:
    settings = load_language_model_settings(env_file=env_file)
    return build_role_language_models(settings)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Module 7 CLI. Returns a POSIX exit status."""
    args = _parse_args(argv)

    try:
        lms = _load_language_models(args.env_file)
        settings = _build_settings(args)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except OptimizationError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    masked = lms.masked()
    print(
        "Module 7 optimization workflow\n"
        f"  teacher model: {masked['teacher_model']}\n"
        f"  student model: {masked['student_model']}\n"
        f"  optimizer:     BootstrapFewShot {settings.to_dict()}\n"
        "  This will make real language-model API calls and run local retrieval "
        "(SQLite + FAISS/BM25 + Neo4j)."
    )

    if args.check_config:
        print("configuration is valid; no model or retrieval call was made.")
        return 0

    try:
        dataset = build_dataset(seed=args.seed)
        result = run_workflow(
            dataset=dataset,
            lms=lms,
            settings=settings,
            retrieve=query_all_sources,
            artifact_dir=args.artifact_dir,
            evaluate_dev=not args.skip_dev,
            held_out_limit=args.held_out_limit,
        )
        _write_results(result, Path(args.artifact_dir))
    except OptimizationError as error:
        print(f"optimization failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - never leak a raw traceback with secrets
        print(f"optimization failed: {type(error).__name__}", file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


def _write_results(result: WorkflowResult, artifact_dir: Path) -> None:
    results = {
        "held_out": result.comparison.to_dict(),
        "dev": None if result.dev_comparison is None else result.dev_comparison.to_dict(),
    }
    path = artifact_dir / RESULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(result: WorkflowResult) -> None:
    comparison = result.comparison
    print(
        f"\nHeld-out comparison: {comparison.outcome}\n"
        f"  baseline accuracy:  {comparison.baseline.accuracy}\n"
        f"  optimized accuracy: {comparison.optimized.accuracy}\n"
        f"  baseline correct:   {comparison.baseline.correct}/{comparison.baseline.total}\n"
        f"  optimized correct:  {comparison.optimized.correct}/{comparison.optimized.total}\n"
        f"  judge rejections:   baseline={comparison.baseline.judge_rejections} "
        f"optimized={comparison.optimized.judge_rejections}"
    )
    print(
        f"  artifact:  {result.artifact.program_path}\n"
        f"  metadata:  {result.artifact.metadata_path}"
    )
    if result.dev_comparison is not None:
        dev = result.dev_comparison
        print(
            f"\nDevelopment comparison (secondary): {dev.outcome}\n"
            f"  baseline accuracy:  {dev.baseline.accuracy}\n"
            f"  optimized accuracy: {dev.optimized.accuracy}"
        )
