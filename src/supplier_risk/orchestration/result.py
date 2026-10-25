"""Typed, provenance-preserving result structures for Module 5 orchestration.

Each retrieval source keeps its own labeled evidence section. The result records
the requested entity, one :class:`SourceEvidence` per source in a fixed order,
per-source status and timing, and the overall elapsed time. It deliberately
contains no risk score, confidence, or APPROVE/FLAG verdict: this module obtains
and assembles evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

SourceName = Literal["sql", "faiss", "graph"]
SourceStatus = Literal["ok", "empty", "error"]
Outcome = Literal["complete", "partial", "failed"]

SOURCE_ORDER: Final[tuple[SourceName, ...]] = ("sql", "faiss", "graph")
SOURCE_LABELS: Final[dict[SourceName, str]] = {
    "sql": "SQL",
    "faiss": "FAISS",
    "graph": "GRAPH",
}


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """One source's labeled outcome, timing, and error information.

    ``context`` always holds the source's complete labeled text, including a
    tagged error string when the source failed, so provenance is never lost.
    ``status`` distinguishes a successful query with no evidence (``"empty"``)
    from a failed query (``"error"``).
    """

    name: SourceName
    status: SourceStatus
    context: str
    elapsed_seconds: float
    error: str | None = None

    @property
    def label(self) -> str:
        """The provenance tag this source's context starts with."""
        return f"[{SOURCE_LABELS[self.name]}]"

    @property
    def succeeded(self) -> bool:
        """True when the source completed, with or without evidence."""
        return self.status != "error"

    @property
    def has_evidence(self) -> bool:
        """True only when the source returned usable evidence."""
        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """The assembled three-source evidence set for one requested entity."""

    entity: str
    sources: tuple[SourceEvidence, ...]
    elapsed_seconds: float

    def __post_init__(self) -> None:
        names = tuple(source.name for source in self.sources)
        if names != SOURCE_ORDER:
            raise ValueError(
                "an orchestration result must hold exactly the SQL, FAISS, and graph "
                f"sources in that order; found {names!r}"
            )

    def source(self, name: SourceName) -> SourceEvidence:
        """Return the evidence for ``name``."""
        for source in self.sources:
            if source.name == name:
                return source
        raise KeyError(name)

    @property
    def sql(self) -> SourceEvidence:
        return self.source("sql")

    @property
    def faiss(self) -> SourceEvidence:
        return self.source("faiss")

    @property
    def graph(self) -> SourceEvidence:
        return self.source("graph")

    @property
    def failed_sources(self) -> tuple[SourceName, ...]:
        """Names of the sources that failed, in canonical order."""
        return tuple(source.name for source in self.sources if not source.succeeded)

    @property
    def errors(self) -> tuple[str, ...]:
        """Tagged error strings for failed sources, in canonical order."""
        return tuple(source.error for source in self.sources if source.error is not None)

    @property
    def complete(self) -> bool:
        """True when no source failed; empty evidence still counts as complete."""
        return not self.failed_sources

    @property
    def outcome(self) -> Outcome:
        """``complete``, ``partial`` (some sources survived), or ``failed``."""
        failed = len(self.failed_sources)
        if failed == 0:
            return "complete"
        if failed == len(self.sources):
            return "failed"
        return "partial"

    def context(self) -> str:
        """All labeled evidence sections, joined in canonical source order.

        Every section keeps its own ``[SQL]``, ``[FAISS]``, or ``[GRAPH]`` tag so
        a downstream reader can attribute each fact to exactly one source.
        """
        return "\n\n".join(source.context for source in self.sources)
