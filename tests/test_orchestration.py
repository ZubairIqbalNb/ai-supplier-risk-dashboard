"""Unit tests for the Module 5 parallel orchestration layer.

Every test injects deterministic source callables, so no test needs FAISS, a
model download, a database, or Neo4j. Concurrency is proven with
:class:`threading.Barrier` rather than with arbitrary sleeps wherever possible.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Callable, Mapping

import pytest

from supplier_risk.graph.pipeline import query_graph
from supplier_risk.orchestration import (
    DEFAULT_MAX_WORKERS,
    DEFAULT_SOURCES,
    SOURCE_ORDER,
    OrchestrationError,
    OrchestrationResult,
    SourceEvidence,
    query_all_sources,
)
from supplier_risk.orchestration.pipeline import THREAD_NAME_PREFIX
from supplier_risk.retrieval.pipeline import query_faiss
from supplier_risk.sql.pipeline import query_sql

SourceMap = Mapping[str, Callable[[str], str]]


def _context(label: str, entity: str) -> str:
    return f"[{label}] supplier={entity} evidence from {label.lower()}"


def _sources(
    *,
    sql: Callable[[str], str] | None = None,
    faiss: Callable[[str], str] | None = None,
    graph: Callable[[str], str] | None = None,
) -> dict[str, Callable[[str], str]]:
    return {
        "sql": sql if sql is not None else lambda entity: _context("SQL", entity),
        "faiss": faiss if faiss is not None else lambda entity: _context("FAISS", entity),
        "graph": graph if graph is not None else lambda entity: _context("GRAPH", entity),
    }


def _error_source(label: str) -> Callable[[str], str]:
    def call(entity: str) -> str:
        return f"[{label} ERROR] source unavailable"

    return call


def test_all_three_sources_are_called_for_a_valid_entity() -> None:
    calls: list[tuple[str, str]] = []

    def recorder(label: str) -> Callable[[str], str]:
        def call(entity: str) -> str:
            calls.append((label, entity))
            return _context(label, entity)

        return call

    result = query_all_sources(
        "SUP-001",
        sources={
            "sql": recorder("SQL"),
            "faiss": recorder("FAISS"),
            "graph": recorder("GRAPH"),
        },
    )

    assert sorted(calls) == [("FAISS", "SUP-001"), ("GRAPH", "SUP-001"), ("SQL", "SUP-001")]
    assert result.complete is True
    assert result.outcome == "complete"


def test_sources_are_submitted_concurrently() -> None:
    barrier = threading.Barrier(len(SOURCE_ORDER), timeout=5)

    def synchronised(label: str) -> Callable[[str], str]:
        def call(entity: str) -> str:
            # Deadlocks (then raises) unless all three calls run at the same time.
            barrier.wait()
            return _context(label, entity)

        return call

    result = query_all_sources(
        "SUP-001",
        sources={
            "sql": synchronised("SQL"),
            "faiss": synchronised("FAISS"),
            "graph": synchronised("GRAPH"),
        },
    )

    assert [source.status for source in result.sources] == ["ok", "ok", "ok"]


def test_total_latency_is_closer_to_the_slowest_source_than_to_the_sum() -> None:
    delay = 0.15

    def slow(label: str) -> Callable[[str], str]:
        def call(entity: str) -> str:
            time.sleep(delay)
            return _context(label, entity)

        return call

    started = time.perf_counter()
    result = query_all_sources(
        "SUP-001",
        sources={
            "sql": slow("SQL"),
            "faiss": slow("FAISS"),
            "graph": slow("GRAPH"),
        },
    )
    wall_clock = time.perf_counter() - started

    # Sequential execution would need at least 0.45s; the upper bound is
    # deliberately tolerant of a loaded CI machine.
    assert wall_clock < 3 * delay + 0.25
    assert result.elapsed_seconds <= wall_clock


def test_successful_result_has_labeled_sections_statuses_and_timings() -> None:
    result = query_all_sources("SUP-001", sources=_sources())

    assert isinstance(result, OrchestrationResult)
    assert result.entity == "SUP-001"
    assert tuple(source.name for source in result.sources) == SOURCE_ORDER
    assert result.sql.context.startswith("[SQL] ")
    assert result.faiss.context.startswith("[FAISS] ")
    assert result.graph.context.startswith("[GRAPH] ")
    assert result.sql.label == "[SQL]"
    for source in result.sources:
        assert isinstance(source, SourceEvidence)
        assert source.status == "ok"
        assert source.has_evidence is True
        assert source.elapsed_seconds >= 0.0
    assert result.elapsed_seconds >= 0.0
    assert result.complete is True
    assert result.failed_sources == ()
    assert result.errors == ()


def test_combined_context_keeps_every_label_in_canonical_order() -> None:
    result = query_all_sources("SUP-001", sources=_sources())

    combined = result.context()

    assert combined.index("[SQL] ") < combined.index("[FAISS] ") < combined.index("[GRAPH] ")
    assert combined.count("[FAISS] ") == 1


@pytest.mark.parametrize("failing", SOURCE_ORDER)
def test_each_single_source_failure_preserves_the_other_two(failing: str) -> None:
    sources = _sources()
    sources[failing] = _error_source(failing.upper())

    result = query_all_sources("SUP-001", sources=sources)

    assert result.outcome == "partial"
    assert result.failed_sources == (failing,)
    for name in SOURCE_ORDER:
        source = result.source(name)
        if name == failing:
            assert source.status == "error"
            assert source.error is not None
            assert source.error.startswith(f"[{source.label.strip('[]')} ERROR]")
            continue
        assert source.status == "ok"
        assert source.has_evidence is True
        assert source.context.startswith(f"[{source.label.strip('[]')}] ")


@pytest.mark.parametrize(
    ("failing", "surviving"),
    [
        (("sql", "faiss"), "graph"),
        (("sql", "graph"), "faiss"),
        (("faiss", "graph"), "sql"),
    ],
)
def test_two_source_failures_keep_the_single_survivor(
    failing: tuple[str, str], surviving: str
) -> None:
    sources = _sources()
    for name in failing:
        sources[name] = _error_source(name.upper())

    result = query_all_sources("SUP-001", sources=sources)

    assert result.outcome == "partial"
    assert result.failed_sources == failing
    assert len(result.errors) == 2
    survivor = result.source(surviving)  # type: ignore[arg-type]
    assert survivor.status == "ok"
    assert survivor.has_evidence is True
    assert result.context().count("evidence from") == 1


def test_all_sources_failing_has_an_explicit_deterministic_outcome() -> None:
    result = query_all_sources(
        "SUP-999",
        sources={
            "sql": _error_source("SQL"),
            "faiss": _error_source("FAISS"),
            "graph": _error_source("GRAPH"),
        },
    )

    assert result.outcome == "failed"
    assert result.complete is False
    assert result.failed_sources == SOURCE_ORDER
    assert len(result.errors) == 3
    assert all(source.status == "error" for source in result.sources)
    assert all(source.has_evidence is False for source in result.sources)
    assert all(source.context.startswith("[") for source in result.sources)


def test_empty_evidence_is_distinct_from_an_error() -> None:
    empty_sql = (
        "[SQL] supplier=SUP-001 name=Norvend Components Ltd partition=train\n"
        "scope=delivery+defect-metrics source=sqlite table=delivery_metrics\n"
        "delivery_metrics=0\n"
        "No delivery or defect metrics were found for this supplier."
    )

    result = query_all_sources(
        "SUP-001",
        sources=_sources(sql=lambda entity: empty_sql),
    )

    assert result.sql.status == "empty"
    assert result.sql.succeeded is True
    assert result.sql.has_evidence is False
    assert result.sql.error is None
    assert result.outcome == "complete"
    assert result.failed_sources == ()


def test_each_source_no_evidence_marker_is_recognised() -> None:
    empty_faiss = (
        "[FAISS] supplier=SUP-001 documents=1\n"
        "No contract or complaint document matched this supplier query."
    )
    empty_graph = "[GRAPH] supplier=SUP-001\nownership_relationships=0"

    result = query_all_sources(
        "SUP-001",
        sources=_sources(
            faiss=lambda entity: empty_faiss,
            graph=lambda entity: empty_graph,
        ),
    )

    assert [source.status for source in result.sources] == ["ok", "empty", "empty"]


def test_unknown_supplier_follows_the_source_contracts() -> None:
    def unknown(label: str) -> Callable[[str], str]:
        def call(entity: str) -> str:
            return (
                f"[{label} ERROR] unknown supplier identity {entity!r}; pass a supplier id "
                "such as SUP-007 or an exact supplier legal name"
            )

        return call

    result = query_all_sources(
        "SUP-999",
        sources={
            "sql": unknown("SQL"),
            "faiss": unknown("FAISS"),
            "graph": unknown("GRAPH"),
        },
    )

    assert result.outcome == "failed"
    assert all("unknown supplier identity" in (source.error or "") for source in result.sources)
    assert result.entity == "SUP-999"


def test_unexpected_exceptions_are_isolated_and_tagged() -> None:
    def explode(entity: str) -> str:
        raise RuntimeError("driver exploded")

    result = query_all_sources(
        "SUP-001",
        sources=_sources(graph=explode),
    )

    assert result.graph.status == "error"
    assert result.graph.context == "[GRAPH ERROR] RuntimeError: driver exploded"
    assert result.graph.error == result.graph.context
    assert result.outcome == "partial"
    assert result.sql.status == "ok"
    assert result.faiss.status == "ok"


def test_unexpected_exception_messages_do_not_leak_credentials() -> None:
    secret = "hunter2-not-a-real-secret"
    leaks = [
        f"connect failed: bolt://neo4j:{secret}@localhost:7687",
        f"NEO4J_PASSWORD={secret}",
        f"api_key: {secret}",
        f"Authorization token={secret}",
    ]

    for leak in leaks:
        def explode(entity: str, message: str = leak) -> str:
            raise ConnectionError(message)

        result = query_all_sources("SUP-001", sources=_sources(graph=explode))

        rendered = result.graph.context + (result.graph.error or "")
        assert secret not in rendered
        assert "***" in rendered


def test_non_string_source_results_are_tagged() -> None:
    result = query_all_sources(
        "SUP-001",
        sources=_sources(sql=lambda entity: {"metrics": 0}),  # type: ignore[dict-item,return-value]
    )

    assert result.sql.status == "error"
    assert "TypeError" in result.sql.context
    assert "expected a labeled context string" in result.sql.context
    assert result.outcome == "partial"


def test_result_order_is_deterministic_when_sources_finish_out_of_order() -> None:
    def delayed(label: str, delay: float) -> Callable[[str], str]:
        def call(entity: str) -> str:
            time.sleep(delay)
            return _context(label, entity)

        return call

    result = query_all_sources(
        "SUP-001",
        sources={
            "sql": delayed("SQL", 0.05),
            "faiss": delayed("FAISS", 0.02),
            "graph": delayed("GRAPH", 0.0),
        },
    )

    assert tuple(source.name for source in result.sources) == SOURCE_ORDER
    assert [source.context.split(" ", 1)[0] for source in result.sources] == [
        "[SQL]",
        "[FAISS]",
        "[GRAPH]",
    ]


def test_worker_threads_are_released_after_the_call() -> None:
    query_all_sources("SUP-001", sources=_sources())

    leftovers = [
        thread for thread in threading.enumerate() if thread.name.startswith(THREAD_NAME_PREFIX)
    ]
    assert leftovers == []


def test_timeout_is_reported_without_killing_successful_sources() -> None:
    release = threading.Event()
    finished = threading.Event()

    def blocking(entity: str) -> str:
        release.wait(10)
        finished.set()
        return _context("GRAPH", entity)

    try:
        result = query_all_sources(
            "SUP-001",
            sources=_sources(graph=blocking),
            timeout_seconds=0.05,
        )
    finally:
        release.set()
        finished.wait(5)

    assert result.outcome == "partial"
    assert result.graph.status == "error"
    assert "timed out after 0.050s" in result.graph.context
    assert "abandoned, not terminated" in result.graph.context
    assert result.sql.status == "ok"
    assert result.faiss.status == "ok"


def test_timeout_marks_every_unfinished_source() -> None:
    release = threading.Event()

    def blocking(label: str) -> Callable[[str], str]:
        def call(entity: str) -> str:
            release.wait(10)
            return _context(label, entity)

        return call

    try:
        result = query_all_sources(
            "SUP-001",
            sources={
                "sql": blocking("SQL"),
                "faiss": blocking("FAISS"),
                "graph": blocking("GRAPH"),
            },
            timeout_seconds=0.05,
        )
    finally:
        release.set()

    assert result.outcome == "failed"
    assert all("timed out" in source.context for source in result.sources)


@pytest.mark.parametrize("max_workers", [0, -1, 1.5, True])
def test_invalid_max_workers_is_rejected(max_workers: object) -> None:
    with pytest.raises(OrchestrationError, match="max_workers"):
        query_all_sources("SUP-001", max_workers=max_workers)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [0, -0.5, "1", True])
def test_invalid_timeout_is_rejected(timeout: object) -> None:
    with pytest.raises(OrchestrationError, match="timeout_seconds"):
        query_all_sources("SUP-001", timeout_seconds=timeout)  # type: ignore[arg-type]


def test_missing_source_callable_is_rejected() -> None:
    with pytest.raises(OrchestrationError, match="missing for: graph"):
        query_all_sources("SUP-001", sources={"sql": lambda e: "[SQL] x", "faiss": lambda e: "[FAISS] x"})


def test_default_worker_capacity_matches_the_three_sources() -> None:
    assert DEFAULT_MAX_WORKERS == len(SOURCE_ORDER) == 3


def test_default_sources_are_the_existing_engine_wrappers() -> None:
    assert DEFAULT_SOURCES == {"sql": query_sql, "faiss": query_faiss, "graph": query_graph}


@pytest.mark.parametrize("function", [query_sql, query_faiss, query_graph])
def test_engine_wrappers_keep_the_entity_contract(function: Callable[..., str]) -> None:
    signature = inspect.signature(function)

    assert next(iter(signature.parameters)) == "entity"
    assert signature.return_annotation == "str"


def test_complete_results_are_logged_at_info_and_failures_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="supplier_risk.orchestration.pipeline")

    query_all_sources("SUP-001", sources=_sources())
    query_all_sources("SUP-001", sources=_sources(graph=_error_source("GRAPH")))

    levels = [record.levelno for record in caplog.records]
    messages = [record.getMessage() for record in caplog.records]
    assert logging.INFO in levels
    assert logging.WARNING in levels
    assert any("outcome=complete" in message for message in messages)
    assert any("outcome=partial" in message for message in messages)
    assert all("SECRET" not in message for message in messages)


def test_a_single_worker_still_returns_all_three_sections() -> None:
    result = query_all_sources("SUP-001", sources=_sources(), max_workers=1)

    assert [source.status for source in result.sources] == ["ok", "ok", "ok"]
    assert result.outcome == "complete"


def test_source_map_type_alias_is_not_shadowed() -> None:
    sources: SourceMap = _sources()

    assert set(sources) == set(SOURCE_ORDER)
