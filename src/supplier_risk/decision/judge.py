"""Deterministic, rule-based judge for Module 6 verdicts.

This is a business-rule component, not a model. It imports no DSPy module, no
provider SDK, opens no socket, and reads no configuration, so the same validated
verdict and the same retrieval result always produce the same judgment.

Rules enforced
--------------
The Step-by-Step Guide (Step 6, p.3) requires the judge to return True/False on
"valid structure, required fields present, values in valid ranges, and at least
one hard content rule". Structure, required fields, and ranges are enforced by
:class:`~supplier_risk.decision.verdict.SupplierRiskVerdict`; the judge reports a
payload that fails them as ``decision_invalid``. The hard content rule is the
banned-word rule the Guide names as its example and that the project's
traceability register already lists: a verdict may not use terms the three
evidence stores cannot establish (``BANNED_TERMS`` below).

Additional evidence rules keep the judge from accepting an approval that the
retrieval outcome does not support:

- ``unacknowledged_failed_source`` — every failed source must be named in
  ``missing_evidence``; a failed source is unknown evidence, not clean evidence.
  One acknowledgement never covers a second failed source.
- ``retrieval_failed`` — all three sources failed, so no verdict is verifiable.
- ``no_evidence_for_approval`` — APPROVE requires at least one source that
  actually returned evidence.
- ``supplier_identity_mismatch`` — the verdict must name the requested supplier.

The judge never upgrades a verdict: when it rejects an APPROVE, its final verdict
is FLAG. A rejected FLAG stays FLAG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from supplier_risk.decision.errors import DecisionError, sanitize_error_message
from supplier_risk.decision.verdict import (
    SupplierRiskVerdict,
    Verdict,
    parse_verdict_payload,
)
from supplier_risk.orchestration.result import SOURCE_ORDER, OrchestrationResult, SourceName

JudgeRule = Literal[
    "decision_invalid",
    "supplier_identity_mismatch",
    "unacknowledged_failed_source",
    "retrieval_failed",
    "no_evidence_for_approval",
    "banned_content",
]

# The hard content rule. Each term names a claim the SQL, FAISS/BM25, and graph
# stores cannot establish for this synthetic dataset, or an overclaim about
# absent evidence. Deliberately over-broad: a single occurrence rejects.
BANNED_TERMS: Final[tuple[str, ...]] = (
    "sanction",
    "blacklist",
    "guarantee",
    "no risk",
    "zero risk",
    "legal control",
    "clean record",
)

_BANNED_PATTERN: Final = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in BANNED_TERMS) + r")",
    re.IGNORECASE,
)

# A failed source is acknowledged when a ``missing_evidence`` item names that
# source. Module 5 names the sources ``sql``/``faiss``/``graph`` and tags their
# sections, including error sections, with the same word in brackets
# (``[GRAPH]``, ``[GRAPH ERROR]``), so one word-boundary pattern per source
# covers both forms. Matching never looks at the exception text: the error
# wording is incidental, the source identity is not.
_ACKNOWLEDGEMENT_PATTERNS: Final[dict[SourceName, re.Pattern[str]]] = {
    name: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE) for name in SOURCE_ORDER
}


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """The deterministic judgment for one model verdict.

    ``model_verdict`` is what the language model returned (``None`` when the
    output was unusable) and ``judge_verdict`` is the final rule-governed result.
    They are kept separate so a later UI or evaluation report can show both.
    """

    passed: bool
    model_verdict: Verdict | None
    judge_verdict: Verdict
    triggered_rules: tuple[JudgeRule, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def rule_triggered(self) -> bool:
        """True when at least one rule rejected the verdict."""
        return bool(self.triggered_rules)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible mapping of the judgment."""
        return {
            "passed": self.passed,
            "model_verdict": self.model_verdict,
            "judge_verdict": self.judge_verdict,
            "triggered_rules": list(self.triggered_rules),
            "reasons": list(self.reasons),
        }


def banned_terms_in(verdict: SupplierRiskVerdict) -> tuple[str, ...]:
    """Return the banned terms used by the verdict, in declared order.

    Matching is case-insensitive and prefix-based on word boundaries, so
    ``Sanctions``, ``blacklisted``, and ``guarantees`` are all caught.
    """
    text = "\n".join((verdict.rationale, *verdict.findings))
    return tuple(term for term in BANNED_TERMS if _BANNED_PATTERN.search(text))


def unacknowledged_failed_sources(
    missing_evidence: tuple[str, ...], failed_sources: tuple[SourceName, ...]
) -> tuple[SourceName, ...]:
    """Return the failed sources no ``missing_evidence`` item names.

    Each failed source is checked independently, so an item naming the SQL source
    acknowledges only SQL. Items that do not name a failed source at all — an
    absent complaint, an unavailable metric — acknowledge nothing, which is why a
    non-empty ``missing_evidence`` list is not by itself an acknowledgement.

    Args:
        missing_evidence: The verdict's missing-evidence items.
        failed_sources: ``OrchestrationResult.failed_sources``, in canonical
            order.

    Returns:
        The unacknowledged source names, in the same canonical order.
    """
    return tuple(
        name
        for name in failed_sources
        if not any(
            _ACKNOWLEDGEMENT_PATTERNS[name].search(item)
            for item in missing_evidence
        )
    )


def judge_verdict(
    verdict: SupplierRiskVerdict | None,
    retrieval: OrchestrationResult,
    *,
    requested_supplier_id: str | None = None,
) -> JudgeResult:
    """Apply the deterministic rules to one validated verdict.

    Args:
        verdict: The validated model verdict, or ``None`` when no usable verdict
            was produced. ``None`` is reported as ``decision_invalid`` rather
            than raised.
        retrieval: The Module 5 evidence result the verdict was based on.
        requested_supplier_id: Optional canonical supplier id the verdict must
            name. When omitted, identity is not re-checked here; the pipeline
            always supplies it.

    Returns:
        A :class:`JudgeResult`. The judge never raises for a bad verdict and
        never calls a model.

    Raises:
        TypeError: if ``verdict`` is neither a
            :class:`~supplier_risk.decision.verdict.SupplierRiskVerdict` nor
            ``None``. Raw payloads belong to :func:`judge_payload`.
    """
    if verdict is None:
        return JudgeResult(
            passed=False,
            model_verdict=None,
            judge_verdict="FLAG",
            triggered_rules=("decision_invalid",),
            reasons=("no validated verdict was supplied",),
        )
    if not isinstance(verdict, SupplierRiskVerdict):
        raise TypeError(
            "judge_verdict expects a SupplierRiskVerdict or None; "
            f"got {type(verdict).__name__}"
        )

    rules: list[JudgeRule] = []
    reasons: list[str] = []

    if requested_supplier_id is not None and verdict.supplier_id != requested_supplier_id:
        rules.append("supplier_identity_mismatch")
        reasons.append(
            f"verdict names {verdict.supplier_id}, expected {requested_supplier_id}"
        )

    unacknowledged = unacknowledged_failed_sources(
        verdict.missing_evidence, retrieval.failed_sources
    )
    if unacknowledged:
        rules.append("unacknowledged_failed_source")
        reasons.append(
            "failed sources are not acknowledged in missing_evidence: "
            + ", ".join(unacknowledged)
        )
    if retrieval.outcome == "failed":
        rules.append("retrieval_failed")
        reasons.append("every retrieval source failed; no verdict is verifiable")

    if verdict.verdict == "APPROVE" and not any(
        source.has_evidence for source in retrieval.sources
    ):
        rules.append("no_evidence_for_approval")
        reasons.append("APPROVE was returned although no source produced evidence")

    banned = banned_terms_in(verdict)
    if banned:
        rules.append("banned_content")
        reasons.append(
            "rationale or findings use terms the evidence cannot support: "
            + ", ".join(banned)
        )

    passed = not rules
    # Never upgrade: a rejected APPROVE is downgraded, a FLAG stays FLAG.
    final_verdict: Verdict = verdict.verdict if passed else "FLAG"
    return JudgeResult(
        passed=passed,
        model_verdict=verdict.verdict,
        judge_verdict=final_verdict,
        triggered_rules=tuple(rules),
        reasons=tuple(reasons),
    )


def judge_payload(
    payload: object,
    retrieval: OrchestrationResult,
    *,
    requested_supplier_id: str | None = None,
) -> JudgeResult:
    """Validate a raw model payload, then judge it.

    This is the entry point for the Module 7 metric, which receives unvalidated
    predictions. A payload that fails the verdict contract is reported as
    ``decision_invalid`` with ``model_verdict=None``; it is never judged as if it
    were well formed.
    """
    try:
        verdict = parse_verdict_payload(payload)
    except DecisionError as error:
        return JudgeResult(
            passed=False,
            model_verdict=None,
            judge_verdict="FLAG",
            triggered_rules=("decision_invalid",),
            reasons=(sanitize_error_message(str(error)),),
        )
    return judge_verdict(verdict, retrieval, requested_supplier_id=requested_supplier_id)
