"""Decision-module tests for Module 6.

A recording fake predictor stands in for the language model, so these tests make
no provider call and need no credentials. They exercise the real ChainOfThought
wiring, the real verdict validation, and the real Module 5 result type.
"""

from __future__ import annotations

import dspy
import pytest

from supplier_risk.decision import (
    DecisionInputError,
    DecisionModelError,
    DecisionResponseError,
    SupplierMismatchError,
    VerdictGenerator,
    decide_supplier,
)
from tests.decision_support import (
    RecordingPredictor,
    make_retrieval,
    prediction,
)


def test_default_predictor_is_chain_of_thought() -> None:
    generator = VerdictGenerator()

    assert isinstance(generator.predictor, dspy.ChainOfThought)


def test_module_call_uses_the_three_contexts() -> None:
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    generator(sql_context="[SQL] a", faiss_context="[FAISS] b", graph_context="[GRAPH] c")

    assert predictor.call_count == 1


def test_forward_passes_each_context_to_its_own_field() -> None:
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    generator(
        sql_context="[SQL] delivery evidence",
        faiss_context="[FAISS] contract evidence",
        graph_context="[GRAPH] ownership evidence",
    )

    assert predictor.calls == [
        {
            "sql_context": "[SQL] delivery evidence",
            "faiss_context": "[FAISS] contract evidence",
            "graph_context": "[GRAPH] ownership evidence",
        }
    ]


def test_generate_makes_exactly_one_prediction() -> None:
    predictor = RecordingPredictor(prediction(), prediction())
    generator = VerdictGenerator(predictor=predictor)

    generator.generate(
        sql_context="[SQL] a",
        faiss_context="[FAISS] b",
        graph_context="[GRAPH] c",
    )

    assert predictor.call_count == 1


def test_decision_makes_exactly_one_prediction_per_request() -> None:
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    decide_supplier("SUP-001", retrieval_result=make_retrieval(), generator=generator)

    assert predictor.call_count == 1


def test_blank_context_is_rejected_before_the_model_call() -> None:
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    with pytest.raises(DecisionInputError, match="faiss_context"):
        generator.generate(
            sql_context="[SQL] a",
            faiss_context="   ",
            graph_context="[GRAPH] c",
        )

    assert predictor.call_count == 0


def test_source_markers_and_contexts_are_preserved() -> None:
    retrieval = make_retrieval(statuses={"graph": "error"})
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    decide_supplier("SUP-001", retrieval_result=retrieval, generator=generator)

    passed = predictor.calls[0]
    assert passed["sql_context"] == retrieval.sql.context
    assert passed["faiss_context"] == retrieval.faiss.context
    assert passed["graph_context"] == retrieval.graph.context
    assert passed["graph_context"].startswith("[GRAPH ERROR]")
    assert "[SQL]" in passed["sql_context"]


def test_failed_retrieval_reaches_the_model_as_three_errors() -> None:
    retrieval = make_retrieval(statuses={"sql": "error", "faiss": "error", "graph": "error"})
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    decide_supplier("SUP-001", retrieval_result=retrieval, generator=generator)

    passed = predictor.calls[0]
    assert passed["sql_context"].startswith("[SQL ERROR]")
    assert passed["faiss_context"].startswith("[FAISS ERROR]")
    assert passed["graph_context"].startswith("[GRAPH ERROR]")
    assert retrieval.outcome == "failed"


def test_malformed_verdict_fails_safely() -> None:
    generator = VerdictGenerator(predictor=RecordingPredictor(prediction(verdict="MAYBE")))

    with pytest.raises(DecisionResponseError, match="APPROVE, FLAG"):
        decide_supplier("SUP-001", retrieval_result=make_retrieval(), generator=generator)


def test_missing_structured_field_fails_safely() -> None:
    incomplete = dspy.Prediction(supplier_id="SUP-001", verdict="APPROVE")
    generator = VerdictGenerator(predictor=RecordingPredictor(incomplete))

    with pytest.raises(DecisionResponseError, match="rationale"):
        decide_supplier("SUP-001", retrieval_result=make_retrieval(), generator=generator)


def test_provider_timeout_is_not_an_approval() -> None:
    generator = VerdictGenerator(
        predictor=RecordingPredictor(TimeoutError("upstream request timed out"))
    )

    with pytest.raises(DecisionModelError, match="TimeoutError") as exc_info:
        decide_supplier("SUP-001", retrieval_result=make_retrieval(), generator=generator)

    assert "APPROVE" not in str(exc_info.value)


def test_provider_error_does_not_leak_credentials() -> None:
    leaked = RuntimeError(
        "GEMINI_API_KEY=sk-live-secret; connection postgres://scott:hunter2@db:5432/app"
    )
    generator = VerdictGenerator(predictor=RecordingPredictor(leaked))

    with pytest.raises(DecisionModelError) as exc_info:
        decide_supplier("SUP-001", retrieval_result=make_retrieval(), generator=generator)

    message = str(exc_info.value)
    assert "sk-live-secret" not in message
    assert "hunter2" not in message
    assert "GEMINI_API_KEY=***" in message


def test_provider_error_is_not_retried() -> None:
    predictor = RecordingPredictor(TimeoutError("timed out"))
    generator = VerdictGenerator(predictor=predictor)

    with pytest.raises(DecisionModelError):
        generator.generate(
            sql_context="[SQL] a",
            faiss_context="[FAISS] b",
            graph_context="[GRAPH] c",
        )

    assert predictor.call_count == 1


def test_none_prediction_is_rejected() -> None:
    generator = VerdictGenerator(predictor=RecordingPredictor(None))

    with pytest.raises(DecisionModelError, match="no prediction"):
        decide_supplier("SUP-001", retrieval_result=make_retrieval(), generator=generator)


def test_unknown_supplier_identity_is_rejected_before_retrieval() -> None:
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    with pytest.raises(DecisionInputError, match="canonical supplier id"):
        decide_supplier("Norvend Components Ltd", generator=generator)

    assert predictor.call_count == 0


def test_retrieval_for_another_supplier_is_rejected() -> None:
    predictor = RecordingPredictor(prediction())
    generator = VerdictGenerator(predictor=predictor)

    with pytest.raises(SupplierMismatchError, match="SUP-002"):
        decide_supplier(
            "SUP-001",
            retrieval_result=make_retrieval("SUP-002"),
            generator=generator,
        )

    assert predictor.call_count == 0


def test_verdict_for_another_supplier_is_rejected() -> None:
    generator = VerdictGenerator(
        predictor=RecordingPredictor(prediction(supplier_id="SUP-002"))
    )

    with pytest.raises(SupplierMismatchError, match="SUP-002"):
        decide_supplier("SUP-001", retrieval_result=make_retrieval(), generator=generator)


def test_requests_do_not_share_supplier_specific_results() -> None:
    predictor = RecordingPredictor(
        prediction(
            supplier_id="SUP-001",
            verdict="FLAG",
            rationale="SQL shows a 2.9% defect rate with one HIGH quality complaint.",
        ),
        prediction(
            supplier_id="SUP-007",
            verdict="APPROVE",
            rationale="SQL metrics and FAISS documents show no recorded quality complaint.",
        ),
    )
    generator = VerdictGenerator(predictor=predictor)

    first = decide_supplier(
        "SUP-001", retrieval_result=make_retrieval("SUP-001"), generator=generator
    )
    second = decide_supplier(
        "SUP-007", retrieval_result=make_retrieval("SUP-007"), generator=generator
    )

    assert (first.supplier_id, first.verdict.supplier_id) == ("SUP-001", "SUP-001")
    assert (second.supplier_id, second.verdict.supplier_id) == ("SUP-007", "SUP-007")
    assert first.verdict.verdict != second.verdict.verdict
    assert first.verdict.rationale != second.verdict.rationale
