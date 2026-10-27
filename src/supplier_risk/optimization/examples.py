"""Convert the Module 2 dataset into DSPy examples for Module 7.

The Module 6 signature takes exactly three evidence contexts (``sql_context``,
``faiss_context``, ``graph_context``) and produces the structured verdict. Module
7 mirrors that signature exactly:

- an example's **inputs** are the three labeled contexts produced by the real
  Module 5 orchestrator (:func:`~supplier_risk.orchestration.pipeline.query_all_sources`);
- the **labels** are the supplier identity, the deterministic expected verdict
  (:mod:`supplier_risk.optimization.labels`), and the underlying
  :class:`~supplier_risk.orchestration.result.OrchestrationResult`.

Because the expected verdict and the retrieval result are labels, never inputs,
they can never leak into the model prompt as unintended Signature inputs. The
metric reads them, and no fabricated retrieval is ever invented: the retrieval
result is exactly what the Module 3-5 engines returned, including tagged errors.

Partition membership is never recomputed here. The dataset already fixes it
(``PARTITION_ID_RANGES``), so a caller selects ``train``, ``dev``, or
``held_out`` and receives only records from that partition, in a stable order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import dspy

from supplier_risk.contracts import PARTITION_SIZES, Partition, SupplierDataset, SupplierRecord
from supplier_risk.data.generate import serialize_dataset
from supplier_risk.decision.verdict import VERDICT_VALUES, Verdict
from supplier_risk.optimization.errors import ExampleConstructionError, OptimizationError
from supplier_risk.optimization.labels import expected_verdict
from supplier_risk.orchestration.pipeline import query_all_sources
from supplier_risk.orchestration.result import OrchestrationResult

# The three inputs of the Module 6 Signature, in declaration order. Kept here so
# the example builder, the metric, and the tests agree on one definition.
SIGNATURE_INPUTS: Final[tuple[str, ...]] = (
    "sql_context",
    "faiss_context",
    "graph_context",
)

Retrieve = Callable[[str], OrchestrationResult]


def dataset_fingerprint(dataset: SupplierDataset) -> str:
    """Return a stable SHA-256 fingerprint of a validated dataset.

    The digest is computed over the same canonical JSON the Module 2 generator
    writes, so two builds of the same seed and manifest produce the same
    fingerprint and any change to a supplier record changes it.
    """
    return hashlib.sha256(serialize_dataset(dataset).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LabeledDecisionExample:
    """One supplier's evidence, expected verdict, and DSPy example."""

    supplier_id: str
    partition: Partition
    expected_verdict: Verdict
    retrieval: OrchestrationResult

    def to_dspy_example(self) -> dspy.Example:
        """Return the DSPy example for this supplier.

        The three contexts are marked as inputs via ``with_inputs``; every other
        field is a label and is therefore never passed to the program.
        """
        example = dspy.Example(
            sql_context=self.retrieval.sql.context,
            faiss_context=self.retrieval.faiss.context,
            graph_context=self.retrieval.graph.context,
            supplier_id=self.supplier_id,
            expected_verdict=self.expected_verdict,
            retrieval=self.retrieval,
        )
        return example.with_inputs(*SIGNATURE_INPUTS)


@dataclass(frozen=True, slots=True)
class ExampleSet:
    """An ordered, single-partition set of labeled decision examples."""

    partition: Partition
    examples: tuple[LabeledDecisionExample, ...]

    @property
    def supplier_ids(self) -> tuple[str, ...]:
        """The supplier ids in this set, in stable order."""
        return tuple(example.supplier_id for example in self.examples)

    def dspy_examples(self) -> list[dspy.Example]:
        """Return the DSPy examples in stable order."""
        return [example.to_dspy_example() for example in self.examples]


def _select_records(dataset: SupplierDataset, partition: Partition) -> list[SupplierRecord]:
    records = [record for record in dataset.suppliers if record.partition == partition]
    records.sort(key=lambda record: record.profile.supplier_id)
    return records


def build_example_set(
    dataset: SupplierDataset,
    partition: Partition,
    *,
    retrieve: Retrieve | None = None,
    limit: int | None = None,
) -> ExampleSet:
    """Build the labeled examples for one partition.

    Args:
        dataset: A validated Module 2 dataset. Its partition assignment is the
            single source of truth and is not recomputed.
        partition: ``"train"``, ``"dev"``, or ``"held_out"``.
        retrieve: Callable returning the Module 5 orchestration result for a
            supplier id. Defaults to the real
            :func:`~supplier_risk.orchestration.pipeline.query_all_sources`.
            Tests inject an offline callable; it is never used to fabricate
            evidence for the optimizer.
        limit: Optional maximum number of examples, taken in ascending supplier
            id order. ``None`` means every record in the partition.

    Returns:
        An :class:`ExampleSet` holding the partition's labeled examples.

    Raises:
        OptimizationError: if ``partition`` is unknown, or ``limit`` is not a
            positive integer.
        ExampleConstructionError: if the retrieval result is missing, is not an
            :class:`OrchestrationResult`, or names a different supplier than the
            record. Missing or malformed labels are rejected here so the metric
            can rely on well-formed examples.
    """
    if partition not in PARTITION_SIZES:
        raise OptimizationError(
            f"unknown partition {partition!r}; expected one of {sorted(PARTITION_SIZES)}"
        )
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise OptimizationError("limit must be a positive integer or None")

    retriever: Retrieve = query_all_sources if retrieve is None else retrieve
    records = _select_records(dataset, partition)
    if limit is not None:
        records = records[:limit]

    examples: list[LabeledDecisionExample] = []
    for record in records:
        supplier_id = record.profile.supplier_id
        label = expected_verdict(record)
        if label not in VERDICT_VALUES:
            raise ExampleConstructionError(
                f"label policy produced an invalid verdict for {supplier_id}: {label!r}"
            )

        try:
            retrieval = retriever(supplier_id)
        except Exception as error:  # noqa: BLE001 - one record must not abort the build
            raise ExampleConstructionError(
                f"retrieval failed for {supplier_id}: {type(error).__name__}"
            ) from None
        if not isinstance(retrieval, OrchestrationResult):
            raise ExampleConstructionError(
                f"retrieval for {supplier_id} returned {type(retrieval).__name__}, "
                "expected an OrchestrationResult"
            )
        if retrieval.entity.strip().upper() != supplier_id:
            raise ExampleConstructionError(
                f"retrieval is for {retrieval.entity!r}, but {supplier_id} was requested"
            )

        examples.append(
            LabeledDecisionExample(
                supplier_id=supplier_id,
                partition=partition,
                expected_verdict=label,
                retrieval=retrieval,
            )
        )

    return ExampleSet(partition=partition, examples=tuple(examples))


def build_partition_examples(
    dataset: SupplierDataset,
    *,
    retrieve: Retrieve | None = None,
    train_limit: int | None = None,
    dev_limit: int | None = None,
    held_out_limit: int | None = None,
) -> dict[Partition, ExampleSet]:
    """Build the train, dev, and held-out example sets in one call.

    The three partitions are built independently and the boundaries are never
    mixed: the held-out set is only ever used for final evaluation.
    """
    limits: dict[Partition, int | None] = {
        "train": train_limit,
        "dev": dev_limit,
        "held_out": held_out_limit,
    }
    return {
        partition: build_example_set(
            dataset, partition, retrieve=retrieve, limit=limits[partition]
        )
        for partition in ("train", "dev", "held_out")
    }
