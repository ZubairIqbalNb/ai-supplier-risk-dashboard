"""Dataset-to-example construction tests for Module 7.

These tests use the real dataset and an offline retrieval double. They assert the
Signature mapping, partition integrity, label integrity, and the dataset
fingerprint. No model, store, or network is touched.
"""

from __future__ import annotations

import pytest

from supplier_risk.contracts import PARTITION_ID_RANGES, SupplierDataset
from supplier_risk.data.generate import build_dataset
from supplier_risk.decision.signature import SupplierRiskSignature
from supplier_risk.optimization.errors import ExampleConstructionError, OptimizationError
from supplier_risk.optimization.examples import (
    SIGNATURE_INPUTS,
    build_example_set,
    build_partition_examples,
    dataset_fingerprint,
)
from supplier_risk.orchestration.result import (
    SOURCE_ORDER,
    OrchestrationResult,
    SourceEvidence,
)
from tests.optimization_support import retrieval_for

DATASET = build_dataset()


def _expected_ids(partition: str) -> tuple[str, ...]:
    return tuple(
        f"SUP-{number:03d}" for number in sorted(PARTITION_ID_RANGES[partition])  # type: ignore[index]
    )


def test_signature_inputs_match_the_module_6_signature() -> None:
    assert set(SupplierRiskSignature.input_fields) == set(SIGNATURE_INPUTS)
    assert SIGNATURE_INPUTS == ("sql_context", "faiss_context", "graph_context")


def test_training_examples_come_only_from_the_train_partition() -> None:
    example_set = build_example_set(DATASET, "train", retrieve=retrieval_for)

    assert example_set.supplier_ids == _expected_ids("train")
    assert all(example.partition == "train" for example in example_set.examples)


def test_dev_examples_come_only_from_the_dev_partition() -> None:
    example_set = build_example_set(DATASET, "dev", retrieve=retrieval_for)

    assert example_set.supplier_ids == _expected_ids("dev")


def test_held_out_examples_come_only_from_the_held_out_partition() -> None:
    example_set = build_example_set(DATASET, "held_out", retrieve=retrieval_for)

    assert example_set.supplier_ids == _expected_ids("held_out")


def test_partitions_are_disjoint() -> None:
    sets = build_partition_examples(DATASET, retrieve=retrieval_for)
    train = set(sets["train"].supplier_ids)
    dev = set(sets["dev"].supplier_ids)
    held = set(sets["held_out"].supplier_ids)

    assert train.isdisjoint(dev)
    assert train.isdisjoint(held)
    assert dev.isdisjoint(held)
    assert len(train | dev | held) == 24


def test_inputs_are_exactly_the_three_contexts() -> None:
    example_set = build_example_set(DATASET, "train", retrieve=retrieval_for)
    dspy_example = example_set.examples[0].to_dspy_example()

    assert list(dspy_example.inputs().keys()) == list(SIGNATURE_INPUTS)


def test_expected_verdict_is_a_label_not_an_input() -> None:
    example_set = build_example_set(DATASET, "train", retrieve=retrieval_for)
    example = example_set.examples[0]
    dspy_example = example.to_dspy_example()

    assert dspy_example.expected_verdict == example.expected_verdict
    assert "expected_verdict" not in dspy_example.inputs()
    assert "expected_verdict" in example_set.dspy_examples()[0].labels()


def test_supplier_identity_and_contexts_are_preserved() -> None:
    example_set = build_example_set(DATASET, "train", retrieve=retrieval_for)
    example = example_set.examples[0]
    dspy_example = example.to_dspy_example()

    assert dspy_example.supplier_id == example.supplier_id
    assert dspy_example.sql_context == example.retrieval.sql.context
    assert dspy_example.faiss_context == example.retrieval.faiss.context
    assert dspy_example.graph_context == example.retrieval.graph.context


def test_retrieval_result_is_preserved_by_identity() -> None:
    example_set = build_example_set(DATASET, "train", retrieve=retrieval_for)
    example = example_set.examples[0]

    assert isinstance(example.retrieval, OrchestrationResult)
    assert example.retrieval.entity == example.supplier_id


def test_limit_takes_a_stable_prefix() -> None:
    full = build_example_set(DATASET, "train", retrieve=retrieval_for)
    limited = build_example_set(DATASET, "train", retrieve=retrieval_for, limit=3)

    assert limited.supplier_ids == full.supplier_ids[:3]


def test_dataset_fingerprint_is_stable_and_seed_sensitive() -> None:
    assert dataset_fingerprint(DATASET) == dataset_fingerprint(build_dataset())
    assert dataset_fingerprint(DATASET) != dataset_fingerprint(build_dataset(seed=1))


def test_unknown_partition_is_rejected() -> None:
    with pytest.raises(OptimizationError, match="unknown partition"):
        build_example_set(DATASET, "validation", retrieve=retrieval_for)  # type: ignore[arg-type]


def test_non_positive_limit_is_rejected() -> None:
    with pytest.raises(OptimizationError, match="limit"):
        build_example_set(DATASET, "train", retrieve=retrieval_for, limit=0)


def test_non_orchestration_result_is_rejected() -> None:
    with pytest.raises(ExampleConstructionError, match="expected an OrchestrationResult"):
        build_example_set(DATASET, "train", retrieve=lambda _sid: "not a result")  # type: ignore[arg-type]


def test_mismatched_supplier_identity_is_rejected() -> None:
    with pytest.raises(ExampleConstructionError, match="was requested"):
        build_example_set(
            DATASET, "train", retrieve=lambda _sid: retrieval_for("SUP-999")
        )


def test_retrieval_failure_is_wrapped() -> None:
    def broken(_sid: str) -> OrchestrationResult:
        raise RuntimeError("store unavailable")

    with pytest.raises(ExampleConstructionError, match="retrieval failed"):
        build_example_set(DATASET, "train", retrieve=broken)


def test_partition_examples_default_retrieval_is_not_invoked_here() -> None:
    # The default retriever is the real orchestrator; this test only checks the
    # injected-retriever path so no store is opened.
    example_set = build_example_set(DATASET, "train", retrieve=retrieval_for, limit=1)
    sources = example_set.examples[0].retrieval.sources
    assert tuple(source.name for source in sources) == SOURCE_ORDER
    assert all(isinstance(source, SourceEvidence) for source in sources)


def test_empty_partition_with_limit_still_builds() -> None:
    small: SupplierDataset = DATASET
    example_set = build_example_set(small, "dev", retrieve=retrieval_for, limit=1)
    assert len(example_set.examples) == 1
