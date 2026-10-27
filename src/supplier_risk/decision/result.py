"""Typed, transparent result of one Module 6 supplier-risk decision.

The result keeps every layer separate so a future UI can render the three raw
evidence contexts next to the model verdict and the deterministic judgment
without re-querying anything:

- the Module 5 :class:`OrchestrationResult`, with per-source status and errors,
- the validated model verdict,
- the deterministic judge result.

Nothing is collapsed into a single opaque string, and no new persistent storage
is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

from supplier_risk.decision.judge import JudgeResult
from supplier_risk.decision.verdict import SupplierRiskVerdict, Verdict
from supplier_risk.orchestration.result import OrchestrationResult, SourceName


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """One complete decision: evidence, model verdict, and judge result."""

    supplier_id: str
    retrieval: OrchestrationResult
    verdict: SupplierRiskVerdict
    judgment: JudgeResult

    def __post_init__(self) -> None:
        if self.verdict.supplier_id != self.supplier_id:
            raise ValueError(
                "decision result supplier identity does not match its verdict: "
                f"{self.supplier_id} != {self.verdict.supplier_id}"
            )

    @property
    def context(self) -> dict[SourceName, str]:
        """The raw labeled context for each source, keyed by source name."""
        return {source.name: source.context for source in self.retrieval.sources}

    @property
    def final_verdict(self) -> Verdict:
        """The deterministic judge's final verdict."""
        return self.judgment.judge_verdict

    @property
    def accepted(self) -> bool:
        """True only when every judge rule passed."""
        return self.judgment.passed

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible mapping for future presentation.

        The layout is intentionally explicit: retrieval outcome and per-source
        status, the three raw contexts, the model verdict, and the judge result.
        """
        return {
            "supplier_id": self.supplier_id,
            "retrieval": {
                "entity": self.retrieval.entity,
                "outcome": self.retrieval.outcome,
                "elapsed_seconds": self.retrieval.elapsed_seconds,
                "sources": [
                    {
                        "name": source.name,
                        "label": source.label,
                        "status": source.status,
                        "context": source.context,
                        "elapsed_seconds": source.elapsed_seconds,
                        "error": source.error,
                    }
                    for source in self.retrieval.sources
                ],
            },
            "verdict": self.verdict.to_dict(),
            "judgment": self.judgment.to_dict(),
        }
