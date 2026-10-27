"""Shared, offline test doubles for the Module 7 optimization tests.

Nothing here makes a network call, reads a credential, or touches a store. The
retrieval double returns the exact :class:`OrchestrationResult` shape the Module 5
orchestrator produces, including its error and no-evidence markers, so the real
metric and judge run unchanged. The dummy LM is DSPy's own
:class:`dspy.utils.dummies.DummyLM`, used only to exercise the real DSPy call
path offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dspy
from dspy.utils.dummies import DummyLM

from supplier_risk.contracts import SupplierDataset, SupplierRecord
from supplier_risk.decision.lm import RoleLanguageModels
from supplier_risk.decision.verdict import VERDICT_VALUES, Verdict
from supplier_risk.optimization.labels import expected_verdict
from supplier_risk.orchestration.result import (
    SOURCE_LABELS,
    SOURCE_ORDER,
    OrchestrationResult,
    SourceEvidence,
    SourceName,
    SourceStatus,
)

CLEAN_RATIONALE = (
    "The SQL delivery metrics, the FAISS contract and complaint evidence, and the "
    "graph ownership links together support the stated verdict."
)
CLEAN_FINDINGS = ["[SQL] metrics reviewed", "[FAISS] documents reviewed"]


def source_context(name: SourceName, supplier_id: str, status: SourceStatus) -> str:
    """Return a context string in the shape the Module 5 orchestrator produces."""
    label = SOURCE_LABELS[name]
    if status == "error":
        return f"[{label} ERROR] RuntimeError: store unavailable"
    if status == "empty":
        return f"[{label}] supplier={supplier_id} no evidence was found"
    return f"[{label}] supplier={supplier_id} partition=train evidence present"


def retrieval_for(
    supplier_id: str = "SUP-001",
    statuses: dict[SourceName, SourceStatus] | None = None,
) -> OrchestrationResult:
    """Build an offline Module 5 result, defaulting every source to ``ok``."""
    resolved = statuses or {}
    sources = tuple(
        SourceEvidence(
            name=name,
            status=resolved.get(name, "ok"),
            context=source_context(name, supplier_id, resolved.get(name, "ok")),
            elapsed_seconds=0.01,
            error=(
                source_context(name, supplier_id, "error")
                if resolved.get(name, "ok") == "error"
                else None
            ),
        )
        for name in SOURCE_ORDER
    )
    return OrchestrationResult(entity=supplier_id, sources=sources, elapsed_seconds=0.03)


def prediction(
    supplier_id: str = "SUP-001",
    verdict: Verdict = "FLAG",
    **overrides: object,
) -> dspy.Prediction:
    """A well-formed ``dspy.Prediction`` including ChainOfThought's ``reasoning``."""
    fields: dict[str, object] = {
        "supplier_id": supplier_id,
        "verdict": verdict,
        "rationale": CLEAN_RATIONALE,
        "findings": list(CLEAN_FINDINGS),
        "missing_evidence": [],
        "reasoning": "internal chain-of-thought that must never be exposed",
    }
    fields.update(overrides)
    return dspy.Prediction(**fields)


def dummy_lm(dataset: SupplierDataset) -> DummyLM:
    """A dummy LM that answers per supplier id with the dataset's expected verdict.

    The answer is keyed on the supplier id present in the prompt, so the real
    ChainOfThought module, the real parser, and the real judge all run offline and
    the metric scores one for every example whose label the policy defines.
    """
    answers: dict[str, dict[str, Any]] = {}
    for record in dataset.suppliers:
        supplier_id = record.profile.supplier_id
        answers[supplier_id] = {
            "reasoning": f"Evidence for {supplier_id} was reviewed.",
            "supplier_id": supplier_id,
            "verdict": expected_verdict(record),
            "rationale": (
                f"The SQL metrics, FAISS documents, and graph links for {supplier_id} "
                "support the stated verdict."
            ),
            "findings": [f"[SQL] metrics reviewed for {supplier_id}"],
            "missing_evidence": [],
        }
    return DummyLM(answers)


def role_lms(lm: object) -> RoleLanguageModels:
    """Wrap one LM as both teacher and student roles for offline tests."""
    return RoleLanguageModels(teacher=lm, student=lm)


def dataset_records_by_partition(dataset: SupplierDataset) -> dict[str, list[SupplierRecord]]:
    grouped: dict[str, list[SupplierRecord]] = {}
    for record in dataset.suppliers:
        grouped.setdefault(record.partition, []).append(record)
    return grouped


class RecordingOptimizer:
    """Fake DSPy optimizer that records its ``compile`` arguments.

    Only the optimizers' *interface* is faked. Every other Module 7 component
    (example construction, metric, evaluation, persistence) is the real one.
    """

    def __init__(self, *, result: object | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._result = result
        self._error = error

    def compile(self, *, student: object, teacher: object, trainset: Sequence[object]) -> object:
        self.calls.append(
            {"student": student, "teacher": teacher, "trainset": list(trainset)}
        )
        if self._error is not None:
            raise self._error
        return student if self._result is None else self._result


def all_labels_present(dataset: SupplierDataset) -> bool:
    """True when the label policy produces a valid verdict for every supplier."""
    return all(expected_verdict(record) in VERDICT_VALUES for record in dataset.suppliers)
