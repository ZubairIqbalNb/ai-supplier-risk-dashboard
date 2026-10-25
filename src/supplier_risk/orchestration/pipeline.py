"""Parallel three-source retrieval orchestration for Module 5.

The SQL, FAISS/BM25, and Neo4j graph sources are submitted together to a bounded
:class:`concurrent.futures.ThreadPoolExecutor`, collected independently, and
assembled into one :class:`~supplier_risk.orchestration.result.OrchestrationResult`
that keeps a labeled evidence section per source.

Sources are the existing assignment-facing wrappers: ``query_sql(entity)``,
``query_faiss(entity)``, and ``query_graph(entity)``. They already return a
labeled string and tag their own failures (``[SQL ERROR]``, ``[FAISS ERROR]``,
``[GRAPH ERROR]``), so this module adapts them instead of changing their
contracts. Any callable with the same ``entity -> str`` shape can be injected.

Thread safety
-------------
The default sources are safe to run concurrently:

- SQLite opens one short-lived read-only connection per query, in the calling
  thread, and closes it before returning.
- The FAISS index is searched read-only and is never rebuilt or mutated, and the
  BM25 index is only queried. The Module 3 default retriever and its embedding
  model are loaded lazily on first use; the very first concurrent FAISS query
  may therefore load the model more than once, which wastes memory and time but
  does not corrupt results.
- Neo4j opens its own short-lived store and driver per query and closes it
  before returning; no driver is shared between threads.

This module never changes logging or environment configuration from a worker
thread. Worker threads only call the injected source callable and time it.

Timeouts
--------
``timeout_seconds`` bounds how long the orchestrator waits for results. It does
not terminate a source call that is already running: a timed-out future is
abandoned and reported as a tagged error, while the underlying call may still be
executing in its worker thread.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Final

from supplier_risk.graph.pipeline import query_graph
from supplier_risk.orchestration.result import (
    SOURCE_LABELS,
    SOURCE_ORDER,
    OrchestrationResult,
    SourceEvidence,
    SourceName,
    SourceStatus,
)
from supplier_risk.retrieval.pipeline import query_faiss
from supplier_risk.sql.pipeline import query_sql

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS: Final = 3
THREAD_NAME_PREFIX: Final = "supplier-risk-retrieval"

SourceCallable = Callable[[str], str]
DEFAULT_SOURCES: Final[Mapping[SourceName, SourceCallable]] = {
    "sql": query_sql,
    "faiss": query_faiss,
    "graph": query_graph,
}

# Each source tags its own failures. A context that starts with the tag is a
# failure; a successful context containing the source's documented no-evidence
# marker is an empty result, not an error.
_ERROR_TAGS: Final[dict[SourceName, str]] = {
    name: f"[{label} ERROR]" for name, label in SOURCE_LABELS.items()
}
_NO_EVIDENCE_MARKERS: Final[dict[SourceName, str]] = {
    "sql": "delivery_metrics=0",
    "faiss": "No contract or complaint document matched this supplier query.",
    "graph": "ownership_relationships=0",
}

_MAX_ERROR_MESSAGE_LENGTH: Final = 300
_CREDENTIAL_URI: Final = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)[^@/\s]+@")
_SECRET_ASSIGNMENT: Final = re.compile(
    r"(?i)(?P<key>[A-Za-z0-9_]*(?:password|secret|token|api[_-]?key|credential)"
    r"[A-Za-z0-9_]*)\s*[=:]\s*\S+"
)


class OrchestrationError(Exception):
    """Raised when the orchestrator itself is configured incorrectly."""


@dataclass(frozen=True, slots=True)
class _SourceOutcome:
    """Raw worker result before it is classified into :class:`SourceEvidence`."""

    text: str | None
    error_name: str | None
    error_message: str | None
    elapsed_seconds: float


def query_all_sources(
    entity: str,
    *,
    sources: Mapping[SourceName, SourceCallable] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout_seconds: float | None = None,
) -> OrchestrationResult:
    """Retrieve entity evidence from all three sources concurrently.

    Args:
        entity: A supplier id such as ``SUP-007`` or an exact supplier legal
            name. Resolution is delegated unchanged to each source; this module
            performs no fuzzy matching and never rewrites the identity.
        sources: Optional mapping overriding the default source callables. It
            must provide callables for ``"sql"``, ``"faiss"``, and ``"graph"``.
        max_workers: Executor capacity. Defaults to one worker per source.
        timeout_seconds: Optional bound on how long to wait for results. The
            timeout does not terminate a running source call.

    Returns:
        An :class:`OrchestrationResult` holding one labeled section per source,
        per-source status and timing, the overall elapsed time, and an explicit
        outcome of ``complete``, ``partial``, or ``failed``.

    Raises:
        OrchestrationError: if ``max_workers`` or ``timeout_seconds`` is invalid
            or a source callable is missing.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise OrchestrationError("max_workers must be a positive integer")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise OrchestrationError("timeout_seconds must be a positive number or None")

    active_sources: Mapping[SourceName, SourceCallable] = (
        DEFAULT_SOURCES if sources is None else sources
    )
    missing = tuple(name for name in SOURCE_ORDER if name not in active_sources)
    if missing:
        raise OrchestrationError(
            "source callables are missing for: " + ", ".join(missing)
        )
    calls = tuple((name, active_sources[name]) for name in SOURCE_ORDER)

    budget = 0.0 if timeout_seconds is None else float(timeout_seconds)
    started = time.perf_counter()
    deadline = None if timeout_seconds is None else started + budget

    executor = ThreadPoolExecutor(
        max_workers=min(max_workers, len(calls)),
        thread_name_prefix=THREAD_NAME_PREFIX,
    )
    timed_out = False
    evidence: list[SourceEvidence] = []
    try:
        futures: dict[SourceName, Future[_SourceOutcome]] = {
            name: executor.submit(_invoke, name, call, entity) for name, call in calls
        }
        for name in SOURCE_ORDER:
            remaining = (
                None if deadline is None else max(deadline - time.perf_counter(), 0.0)
            )
            try:
                outcome = futures[name].result(timeout=remaining)
            except FuturesTimeoutError:
                timed_out = True
                evidence.append(
                    _timeout_evidence(name, budget, time.perf_counter() - started)
                )
                continue
            evidence.append(_classify_outcome(name, outcome))
    finally:
        # Do not block on a source that overran the deadline; its future has
        # already been abandoned. Cancelling pending work keeps cleanup prompt.
        executor.shutdown(wait=not timed_out, cancel_futures=True)

    result = OrchestrationResult(
        entity=entity,
        sources=tuple(evidence),
        elapsed_seconds=time.perf_counter() - started,
    )
    _log_result(result)
    return result


def _invoke(name: SourceName, call: SourceCallable, entity: str) -> _SourceOutcome:
    """Call one source and time it; never let one source raise into the pool."""
    started = time.perf_counter()
    try:
        text = call(entity)
    except Exception as error:  # noqa: BLE001 - one source must not destroy the others
        return _SourceOutcome(
            text=None,
            error_name=type(error).__name__,
            error_message=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )
    elapsed = time.perf_counter() - started
    if not isinstance(text, str):
        return _SourceOutcome(
            text=None,
            error_name="TypeError",
            error_message=(
                f"{SOURCE_LABELS[name]} source returned {type(text).__name__}, "
                "expected a labeled context string"
            ),
            elapsed_seconds=elapsed,
        )
    return _SourceOutcome(
        text=text,
        error_name=None,
        error_message=None,
        elapsed_seconds=elapsed,
    )


def _classify_outcome(name: SourceName, outcome: _SourceOutcome) -> SourceEvidence:
    if outcome.text is None:
        message = _redact(outcome.error_message or "unknown failure")
        label = SOURCE_LABELS[name]
        text = f"[{label} ERROR] {outcome.error_name}: {message}"
        return SourceEvidence(
            name=name,
            status="error",
            context=text,
            elapsed_seconds=outcome.elapsed_seconds,
            error=text,
        )
    status = _classify_status(name, outcome.text)
    return SourceEvidence(
        name=name,
        status=status,
        context=outcome.text,
        elapsed_seconds=outcome.elapsed_seconds,
        error=outcome.text if status == "error" else None,
    )


def _classify_status(name: SourceName, text: str) -> SourceStatus:
    if text.startswith(_ERROR_TAGS[name]):
        return "error"
    if _NO_EVIDENCE_MARKERS[name] in text:
        return "empty"
    return "ok"


def _timeout_evidence(
    name: SourceName, budget: float, elapsed_seconds: float
) -> SourceEvidence:
    label = SOURCE_LABELS[name]
    text = (
        f"[{label} ERROR] retrieval timed out after {budget:.3f}s; the source call "
        "was abandoned, not terminated"
    )
    return SourceEvidence(
        name=name,
        status="error",
        context=text,
        elapsed_seconds=elapsed_seconds,
        error=text,
    )


def _redact(text: str) -> str:
    """Remove credential-like values from an unexpected exception message."""
    redacted = _CREDENTIAL_URI.sub(r"\g<scheme>***@", text)
    redacted = _SECRET_ASSIGNMENT.sub(r"\g<key>=***", redacted)
    return redacted[:_MAX_ERROR_MESSAGE_LENGTH]


def _log_result(result: OrchestrationResult) -> None:
    detail = " ".join(
        f"{source.name}={source.status}/{source.elapsed_seconds:.3f}s"
        for source in result.sources
    )
    if result.outcome == "complete":
        logger.info(
            "entity=%s outcome=complete elapsed=%.3fs %s",
            result.entity,
            result.elapsed_seconds,
            detail,
        )
        return
    logger.warning(
        "entity=%s outcome=%s elapsed=%.3fs %s",
        result.entity,
        result.outcome,
        result.elapsed_seconds,
        detail,
    )
