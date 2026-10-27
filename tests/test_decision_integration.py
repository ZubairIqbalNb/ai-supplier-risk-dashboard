"""Integration-style tests for the Module 6 decision path.

The real Module 5 orchestrator runs over injected fake sources, so no Docker,
Neo4j server, network, or model credentials are required. The path exercised is:
canonical supplier identity -> Module 5 evidence result -> DSPy module (fake
predictor) -> validated verdict -> deterministic judge -> typed decision result.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from supplier_risk.decision import VerdictGenerator, decide_supplier
from supplier_risk.orchestration import SOURCE_ORDER, SourceName, query_all_sources
from tests.decision_support import (
    EVIDENCE,
    NO_EVIDENCE,
    RecordingPredictor,
    make_retrieval,
    prediction,
)

APPROVE_RATIONALE = (
    "SQL metrics are stable, FAISS reports no adverse contract or complaint wording, "
    "and the graph confirms a majority-owned subsidiary."
)


def _sources(
    statuses: dict[SourceName, str],
) -> dict[SourceName, Callable[[str], str]]:
    """Fake source callables with the Module 5 ``entity -> str`` contract."""

    def call_for(name: SourceName) -> Callable[[str], str]:
        def call(entity: str) -> str:
            status = statuses.get(name, "ok")
            if status == "error":
                raise RuntimeError("store unavailable")
            return EVIDENCE[name] if status == "ok" else NO_EVIDENCE[name]

        return call

    return {name: call_for(name) for name in SOURCE_ORDER}


def test_complete_evidence_produces_a_typed_transparent_decision() -> None:
    retrieval = query_all_sources("SUP-001", sources=_sources({}))
    predictor = RecordingPredictor(prediction(verdict="APPROVE", rationale=APPROVE_RATIONALE))

    result = decide_supplier(
        "SUP-001",
        retrieval_result=retrieval,
        generator=VerdictGenerator(predictor=predictor),
    )

    assert retrieval.outcome == "complete"
    assert predictor.call_count == 1
    assert predictor.calls[0] == {
        "sql_context": retrieval.sql.context,
        "faiss_context": retrieval.faiss.context,
        "graph_context": retrieval.graph.context,
    }
    assert result.verdict.verdict == "APPROVE"
    assert result.final_verdict == "APPROVE"
    assert result.accepted is True

    payload = result.to_dict()
    sources = payload["retrieval"]["sources"]
    assert [source["name"] for source in sources] == ["sql", "faiss", "graph"]
    assert [source["status"] for source in sources] == ["ok", "ok", "ok"]
    assert len({source["context"] for source in sources}) == 3
    assert payload["verdict"]["supplier_id"] == "SUP-001"
    assert payload["judgment"] == {
        "passed": True,
        "model_verdict": "APPROVE",
        "judge_verdict": "APPROVE",
        "triggered_rules": [],
        "reasons": [],
    }


def test_partial_retrieval_stays_visible_and_is_judged_as_flagged() -> None:
    retrieval = query_all_sources("SUP-001", sources=_sources({"graph": "error"}))
    predictor = RecordingPredictor(
        prediction(
            rationale="SQL and FAISS evidence is available, but the ownership graph failed.",
            missing_evidence=["[GRAPH ERROR] RuntimeError: store unavailable"],
        )
    )

    result = decide_supplier(
        "SUP-001",
        retrieval_result=retrieval,
        generator=VerdictGenerator(predictor=predictor),
    )

    assert retrieval.outcome == "partial"
    assert predictor.calls[0]["graph_context"].startswith("[GRAPH ERROR]")
    assert result.retrieval.graph.status == "error"
    assert result.accepted is True
    assert result.final_verdict == "FLAG"

    payload = result.to_dict()
    assert payload["retrieval"]["outcome"] == "partial"
    assert payload["retrieval"]["sources"][2]["status"] == "error"
    assert payload["retrieval"]["sources"][2]["error"].startswith("[GRAPH ERROR]")


def test_failed_retrieval_cannot_become_a_fabricated_approval() -> None:
    retrieval = query_all_sources(
        "SUP-001",
        sources=_sources({"sql": "error", "faiss": "error", "graph": "error"}),
    )
    predictor = RecordingPredictor(
        prediction(verdict="APPROVE", rationale=APPROVE_RATIONALE, missing_evidence=[])
    )

    result = decide_supplier(
        "SUP-001",
        retrieval_result=retrieval,
        generator=VerdictGenerator(predictor=predictor),
    )

    assert retrieval.outcome == "failed"
    for name in SOURCE_ORDER:
        assert predictor.calls[0][f"{name}_context"].startswith(f"[{name.upper()} ERROR]")

    assert result.judgment.model_verdict == "APPROVE"
    assert result.accepted is False
    assert result.final_verdict == "FLAG"
    assert {
        "retrieval_failed",
        "unacknowledged_failed_source",
        "no_evidence_for_approval",
    } <= set(result.judgment.triggered_rules)


def test_default_retrieval_path_uses_the_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float | None]] = []

    def fake_query_all_sources(
        entity: str,
        *,
        sources: object = None,
        max_workers: int = 3,
        timeout_seconds: float | None = None,
    ) -> object:
        calls.append((entity, timeout_seconds))
        return make_retrieval(entity)

    monkeypatch.setattr(
        "supplier_risk.decision.pipeline.query_all_sources", fake_query_all_sources
    )
    predictor = RecordingPredictor(prediction(supplier_id="SUP-007"))

    result = decide_supplier(
        " sup-007 ",
        generator=VerdictGenerator(predictor=predictor),
        retrieval_timeout_seconds=2.5,
    )

    assert calls == [("SUP-007", 2.5)]
    assert result.supplier_id == "SUP-007"
    assert result.verdict.supplier_id == "SUP-007"
