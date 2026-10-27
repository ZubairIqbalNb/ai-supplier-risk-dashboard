"""The Module 6 decision pipeline.

``decide_supplier(entity)`` runs, in order:

1. canonical supplier identity validation,
2. Module 5 evidence assembly (or a supplied :class:`OrchestrationResult`),
3. one DSPy ``ChainOfThought`` prediction over the three labeled contexts,
4. verdict-contract validation and supplier-identity enforcement,
5. the deterministic rule-based judge.

The pipeline performs no retry, no second model call, no tool loop, and no
fallback verdict. When the model call fails, the request fails; it never becomes
an ``APPROVE``.
"""

from __future__ import annotations

import logging

from supplier_risk.decision.errors import SupplierMismatchError
from supplier_risk.decision.judge import judge_verdict
from supplier_risk.decision.module import VerdictGenerator
from supplier_risk.decision.result import DecisionResult
from supplier_risk.decision.verdict import (
    parse_supplier_identity,
    parse_verdict_payload,
    require_matching_supplier,
)
from supplier_risk.orchestration.pipeline import query_all_sources
from supplier_risk.orchestration.result import OrchestrationResult

logger = logging.getLogger(__name__)


def decide_supplier(
    entity: str,
    *,
    retrieval_result: OrchestrationResult | None = None,
    generator: VerdictGenerator | None = None,
    retrieval_timeout_seconds: float | None = None,
) -> DecisionResult:
    """Produce one evidence-grounded supplier-risk decision.

    Args:
        entity: A canonical supplier id such as ``SUP-007``. Legal-name
            resolution belongs to the store wrappers; restricting the decision
            layer to ids keeps the verdict's supplier identity exactly comparable.
        retrieval_result: A Module 5 result to use instead of querying again. Its
            ``entity`` must be the same canonical supplier id. Partial and
            failed results are allowed and stay visible to the model and judge.
        generator: Optional decision module. Defaults to a new
            :class:`VerdictGenerator`; tests inject a fake predictor through it,
            and Module 7 can inject a compiled student program the same way.
            Exactly one prediction is made.
        retrieval_timeout_seconds: Forwarded to
            :func:`~supplier_risk.orchestration.pipeline.query_all_sources` when
            ``retrieval_result`` is not supplied.

    Returns:
        A :class:`DecisionResult` holding the retrieval result, the validated
        model verdict, and the deterministic judgment.

    Raises:
        DecisionInputError: if ``entity`` is not a canonical supplier id.
        SupplierMismatchError: if the retrieval result or the verdict names a
            different supplier than ``entity``.
        DecisionModelError: if the language-model call fails. No verdict is
            produced.
        DecisionResponseError: if the model output violates the verdict contract.
        OrchestrationError: if the default retrieval is misconfigured.
    """
    supplier_id = parse_supplier_identity(entity)

    retrieval = (
        query_all_sources(supplier_id, timeout_seconds=retrieval_timeout_seconds)
        if retrieval_result is None
        else retrieval_result
    )
    if retrieval.entity.strip().upper() != supplier_id:
        raise SupplierMismatchError(
            f"retrieval result is for {retrieval.entity!r}, but {supplier_id} was requested"
        )

    active_generator = VerdictGenerator() if generator is None else generator
    prediction = active_generator.generate(
        sql_context=retrieval.sql.context,
        faiss_context=retrieval.faiss.context,
        graph_context=retrieval.graph.context,
    )

    verdict = require_matching_supplier(
        parse_verdict_payload(prediction), supplier_id
    )
    judgment = judge_verdict(verdict, retrieval, requested_supplier_id=supplier_id)

    logger.info(
        "decision supplier=%s outcome=%s verdict=%s judged=%s rules=%s",
        supplier_id,
        retrieval.outcome,
        verdict.verdict,
        judgment.judge_verdict,
        ",".join(judgment.triggered_rules) or "none",
    )
    return DecisionResult(
        supplier_id=supplier_id,
        retrieval=retrieval,
        verdict=verdict,
        judgment=judgment,
    )
