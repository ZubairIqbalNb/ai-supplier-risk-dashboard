"""Offline fixtures for the Module 6 decision tests.

Everything here is deterministic and local: no provider credentials, network
access, Docker, Neo4j, or embedding model is involved. Evidentiary strings follow
the exact shapes the Module 3-5 wrappers return, including their error and
no-evidence markers, so the real Module 5 orchestrator can classify them.
"""

from __future__ import annotations

from typing import Final

import dspy

from supplier_risk.decision.verdict import SupplierRiskVerdict, parse_verdict_payload
from supplier_risk.orchestration.result import (
    SOURCE_LABELS,
    SOURCE_ORDER,
    OrchestrationResult,
    SourceEvidence,
    SourceName,
    SourceStatus,
)

EVIDENCE: Final[dict[SourceName, str]] = {
    "sql": (
        "[SQL] supplier=SUP-001 name=Norvend Components Ltd partition=train\n"
        "scope=delivery+defect-metrics source=sqlite table=delivery_metrics\n"
        "delivery_metrics=1\n"
        "1. period=FY2025 on_time_delivery_rate=0.809 defect_rate=0.0291 "
        "average_lead_time_days=13.4 orders_fulfilled=3970"
    ),
    "faiss": (
        "[FAISS] supplier=SUP-001 name=Norvend Components Ltd partition=train documents=2\n"
        "1. document_id=complaint:CMP-001-01 severity=HIGH category=QUALITY text=..."
        "\n2. document_id=contract:CON-001-01 contract_type=MASTER_SUPPLY text=..."
    ),
    "graph": (
        "[GRAPH] supplier=SUP-001 ownership_relationships=1\n"
        "1. supplier SUP-001 -[SUBSIDIARY_OF 70.0%]-> company ORG-001 Norvend Holdings"
    ),
}

# Shapes that make the real Module 5 orchestrator classify a source as "empty".
NO_EVIDENCE: Final[dict[SourceName, str]] = {
    "sql": (
        "[SQL] supplier=SUP-001 name=Norvend Components Ltd partition=train\n"
        "delivery_metrics=0\nNo delivery or defect metrics were found for this supplier."
    ),
    "faiss": (
        "[FAISS] supplier=SUP-001 name=Norvend Components Ltd partition=train "
        "documents=0\nNo contract or complaint document matched this supplier query."
    ),
    "graph": (
        "[GRAPH] supplier=SUP-001 ownership_relationships=0\n"
        "No ownership evidence was found for this supplier."
    ),
}

ERROR: Final[dict[SourceName, str]] = {
    name: f"[{SOURCE_LABELS[name]} ERROR] RuntimeError: store unavailable"
    for name in SOURCE_ORDER
}

CLEAN_RATIONALE: Final = (
    "SQL shows a 2.9% defect rate and FAISS reports one HIGH quality complaint, "
    "so continued sourcing is flagged pending an inspection record."
)

CLEAN_FINDINGS: Final = [
    "[SQL] defect_rate=0.0291 above the reviewed sample",
    "[FAISS] complaint CMP-001-01 severity=HIGH category=QUALITY",
    "[GRAPH] SUP-001 is a 70% subsidiary of ORG-001",
]


def source_evidence(name: SourceName, status: SourceStatus) -> SourceEvidence:
    """Build one source section in the shape Module 5 produces."""
    context = (
        EVIDENCE[name]
        if status == "ok"
        else NO_EVIDENCE[name]
        if status == "empty"
        else ERROR[name]
    )
    return SourceEvidence(
        name=name,
        status=status,
        context=context,
        elapsed_seconds=0.01,
        error=ERROR[name] if status == "error" else None,
    )


def make_retrieval(
    entity: str = "SUP-001",
    statuses: dict[SourceName, SourceStatus] | None = None,
) -> OrchestrationResult:
    """Build a Module 5 result, defaulting every source to ``ok``."""
    resolved = statuses or {}
    return OrchestrationResult(
        entity=entity,
        sources=tuple(
            source_evidence(name, resolved.get(name, "ok")) for name in SOURCE_ORDER
        ),
        elapsed_seconds=0.03,
    )


def prediction(**overrides: object) -> dspy.Prediction:
    """A well-formed ``dspy.Prediction`` for the verdict contract.

    ``reasoning`` is present because ``dspy.ChainOfThought`` always returns it;
    the decision layer must never expose it.
    """
    fields: dict[str, object] = {
        "supplier_id": "SUP-001",
        "verdict": "FLAG",
        "rationale": CLEAN_RATIONALE,
        "findings": list(CLEAN_FINDINGS),
        "missing_evidence": [],
        "reasoning": "internal chain-of-thought that must never be exposed",
    }
    fields.update(overrides)
    return dspy.Prediction(**fields)


def clean_verdict(**overrides: object) -> SupplierRiskVerdict:
    """A validated verdict built through the public parser."""
    payload: dict[str, object] = {
        "supplier_id": "SUP-001",
        "verdict": "FLAG",
        "rationale": CLEAN_RATIONALE,
        "findings": list(CLEAN_FINDINGS),
        "missing_evidence": [],
    }
    payload.update(overrides)
    return parse_verdict_payload(payload)


class RecordingPredictor:
    """Fake predictor that records its calls and replays queued results.

    A queued result may be an exception, which is raised on that call. The
    predictor fails loudly if it is called more often than expected, so a silent
    retry or a second model call is caught by the tests.
    """

    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[dict[str, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def __call__(self, **kwargs: str) -> object:
        self.calls.append(dict(kwargs))
        if not self._results:
            raise AssertionError("predictor was called more often than expected")
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result
