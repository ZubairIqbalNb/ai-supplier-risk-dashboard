"""The one-call DSPy ChainOfThought decision module for Module 6.

``VerdictGenerator`` wraps :class:`SupplierRiskSignature` in
``dspy.ChainOfThought`` and makes exactly one prediction per call. There is no
retry that would silently spend a second model call, no tool loop, no
self-reflection loop, and no agent. A failed call is raised as
:class:`DecisionModelError`; it is never turned into a verdict.

The predictor is injectable so offline tests can run the real module, the real
validation, and the real judge without network access or credentials.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import dspy

from supplier_risk.decision.errors import (
    DecisionInputError,
    DecisionModelError,
    sanitize_error_message,
)
from supplier_risk.decision.signature import SupplierRiskSignature

logger = logging.getLogger(__name__)

Predictor = Callable[..., object]


def _require_context(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise DecisionInputError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise DecisionInputError(f"{name} must not be blank")
    return value


class VerdictGenerator(dspy.Module):
    """Generate one structured supplier-risk verdict from three evidence contexts.

    Args:
        predictor: Optional callable used instead of the default
            ``dspy.ChainOfThought(SupplierRiskSignature)``. Tests inject a fake;
            Module 7 may inject a compiled student program with the same
            signature.
    """

    def __init__(self, *, predictor: Predictor | None = None) -> None:
        super().__init__()
        self.predictor: Predictor = (
            dspy.ChainOfThought(SupplierRiskSignature)
            if predictor is None
            else predictor
        )

    def forward(
        self, *, sql_context: str, faiss_context: str, graph_context: str
    ) -> object:
        """DSPy entry point; delegates to :meth:`generate`."""
        return self.generate(
            sql_context=sql_context,
            faiss_context=faiss_context,
            graph_context=graph_context,
        )

    def generate(
        self, *, sql_context: str, faiss_context: str, graph_context: str
    ) -> object:
        """Make exactly one prediction from the three evidence contexts.

        The contexts are passed through unchanged, so each one keeps its source
        label, error marker, and no-evidence marker.

        Raises:
            DecisionInputError: if a context is not a non-blank string.
            DecisionModelError: if the language-model call fails, times out, or
                returns nothing. The provider message is sanitized; credentials
                and large payloads are never logged.
        """
        sql_context = _require_context("sql_context", sql_context)
        faiss_context = _require_context("faiss_context", faiss_context)
        graph_context = _require_context("graph_context", graph_context)

        started = time.perf_counter()
        try:
            prediction = self.predictor(
                sql_context=sql_context,
                faiss_context=faiss_context,
                graph_context=graph_context,
            )
        except Exception as error:  # noqa: BLE001 - one LM failure, one project error
            logger.warning(
                "verdict generation failed: %s (sql=%d faiss=%d graph=%d chars)",
                type(error).__name__,
                len(sql_context),
                len(faiss_context),
                len(graph_context),
            )
            raise DecisionModelError(
                "language model call failed "
                f"({type(error).__name__}): {sanitize_error_message(str(error))}"
            ) from None
        logger.info("verdict generation completed in %.3fs", time.perf_counter() - started)
        if prediction is None:
            raise DecisionModelError("the language model returned no prediction")
        return prediction
