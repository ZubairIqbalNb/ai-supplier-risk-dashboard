"""Typed structured supplier-risk verdict contract for Module 6.

The assignment defines the verdict vocabulary (``APPROVE`` / ``FLAG``) and asks
for "structured verdict fields", but does not name the fields. The contract here
is a project-level choice recorded in ``docs/schema.md`` section 15: supplier
identity, verdict, a concise evidence-grounded rationale, the material findings,
and the missing evidence that materially affects the decision. No risk score or
confidence value is added, because the assignment does not require one and an
invented scale could not be validated.

Free-form model output is never trusted directly. ``parse_verdict_payload``
normalizes the fields it is allowed to normalize, re-validates everything else
through Pydantic, and rejects anything that is missing, malformed, or describes
a different supplier.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from supplier_risk.decision.errors import (
    DecisionInputError,
    DecisionResponseError,
    SupplierMismatchError,
)

Verdict = Literal["APPROVE", "FLAG"]

"""The permitted verdict values, in assignment order."""
VERDICT_VALUES: Final[tuple[Verdict, ...]] = ("APPROVE", "FLAG")

MIN_RATIONALE_CHARS: Final = 20
MAX_RATIONALE_CHARS: Final = 800
MIN_FINDINGS: Final = 1
MAX_FINDINGS: Final = 8
MIN_FINDING_CHARS: Final = 3
MAX_FINDING_CHARS: Final = 400
MAX_MISSING_EVIDENCE: Final = 8

# Mirrors ``SupplierProfile.supplier_id`` in the Module 2 contracts. The decision
# layer accepts only a canonical supplier id so identity comparison stays exact;
# legal-name resolution remains the job of the three store wrappers.
_SUPPLIER_ID_PATTERN: Final = re.compile(r"^SUP-\d{3}$")

# The fields a decision payload must supply, in contract order.
VERDICT_FIELDS: Final[tuple[str, ...]] = (
    "supplier_id",
    "verdict",
    "rationale",
    "findings",
    "missing_evidence",
)


class SupplierRiskVerdict(BaseModel):
    """One validated supplier-risk decision produced by the DSPy module.

    The model is frozen, forbids unknown fields, and rejects type coercion, so a
    verdict object can only exist in a state that satisfies this contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    verdict: Verdict
    rationale: str = Field(min_length=MIN_RATIONALE_CHARS, max_length=MAX_RATIONALE_CHARS)
    findings: tuple[str, ...] = Field(min_length=MIN_FINDINGS, max_length=MAX_FINDINGS)
    missing_evidence: tuple[str, ...] = Field(max_length=MAX_MISSING_EVIDENCE)

    @field_validator("rationale")
    @classmethod
    def _rationale_is_trimmed_and_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale must not be blank")
        if value != value.strip():
            raise ValueError("rationale must not have leading or trailing whitespace")
        return value

    @field_validator("findings")
    @classmethod
    def _findings_are_usable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_items(value, "findings", MIN_FINDING_CHARS, MAX_FINDING_CHARS)

    @field_validator("missing_evidence")
    @classmethod
    def _missing_evidence_is_usable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_items(value, "missing_evidence", 1, MAX_FINDING_CHARS)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, JSON-compatible mapping of the verdict.

        Field order follows the declaration order above, tuples become lists, and
        no value depends on time, randomness, or the process environment.
        """
        return self.model_dump(mode="json")


def _validate_items(
    value: tuple[str, ...], field: str, min_chars: int, max_chars: int
) -> tuple[str, ...]:
    for index, item in enumerate(value):
        if not item.strip():
            raise ValueError(f"{field}[{index}] must not be blank")
        if item != item.strip():
            raise ValueError(f"{field}[{index}] must not have surrounding whitespace")
        if len(item) < min_chars:
            raise ValueError(f"{field}[{index}] must be at least {min_chars} characters")
        if len(item) > max_chars:
            raise ValueError(f"{field}[{index}] must be at most {max_chars} characters")
    return value


def parse_supplier_identity(entity: str) -> str:
    """Normalize and validate a canonical supplier id.

    Args:
        entity: A supplier id such as ``SUP-007``. Case and surrounding
            whitespace are normalized.

    Returns:
        The uppercase canonical supplier id.

    Raises:
        DecisionInputError: if ``entity`` is not a ``SUP-NNN`` identifier. The
            decision layer deliberately does not resolve legal names itself so
            that the verdict's supplier identity can be compared exactly.
    """
    if not isinstance(entity, str):
        raise DecisionInputError(
            f"supplier identity must be a string, got {type(entity).__name__}"
        )
    text = entity.strip().upper()
    if not _SUPPLIER_ID_PATTERN.fullmatch(text):
        raise DecisionInputError(
            "supplier identity must be a canonical supplier id such as SUP-007; "
            "resolve legal names through the store wrappers first"
        )
    return text


_MISSING: Final = object()


def _read_field(payload: object, name: str) -> object:
    if isinstance(payload, Mapping):
        return payload.get(name, _MISSING)
    return getattr(payload, name, _MISSING)


def _normalized_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise DecisionResponseError(f"{field} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise DecisionResponseError(f"{field} must not be blank")
    return text


def _normalized_items(value: object, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise DecisionResponseError(f"{field} must be a list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise DecisionResponseError(
                f"{field}[{index}] must be a string, got {type(item).__name__}"
            )
        text = item.strip()
        if not text:
            raise DecisionResponseError(f"{field}[{index}] must not be blank")
        items.append(text)
    if not items and not allow_empty:
        raise DecisionResponseError(f"{field} must contain at least one item")
    return tuple(items)


def _format_validation_error(error: ValidationError) -> str:
    """Return field locations only; never echo model text back to the caller."""
    locations = [
        ".".join(str(part) for part in detail.get("loc", ())) or "<verdict>"
        for detail in error.errors(include_input=False, include_url=False)
    ]
    return "invalid verdict contract at: " + ", ".join(dict.fromkeys(locations))


def parse_verdict_payload(payload: object) -> SupplierRiskVerdict:
    """Validate one raw model response against the verdict contract.

    ``payload`` may be a mapping or an object with the contract fields as
    attributes (a ``dspy.Prediction``). The verdict value is uppercased and the
    list items are trimmed before validation; every other rule is enforced by
    :class:`SupplierRiskVerdict`. String payloads, missing fields, unknown fields,
    blank values, out-of-range lengths, and invalid verdict values are rejected.

    Raises:
        DecisionResponseError: if the output cannot be used as a verdict.
    """
    if payload is None:
        raise DecisionResponseError("the model produced no structured output")
    if isinstance(payload, str):
        raise DecisionResponseError(
            "free-form model text is not a structured verdict; no validated fields"
        )

    if isinstance(payload, Mapping):
        unknown = [str(key) for key in payload if key not in VERDICT_FIELDS]
        if unknown:
            raise DecisionResponseError(
                "structured output has unknown field(s): " + ", ".join(unknown)
            )

    missing = [
        name for name in VERDICT_FIELDS if _read_field(payload, name) is _MISSING
    ]
    if missing:
        raise DecisionResponseError(
            "structured output is missing required field(s): " + ", ".join(missing)
        )

    raw_verdict = _read_field(payload, "verdict")
    verdict_text = _normalized_text(raw_verdict, "verdict").upper()
    if verdict_text not in VERDICT_VALUES:
        raise DecisionResponseError(
            f"verdict must be one of {', '.join(VERDICT_VALUES)}; "
            f"got {verdict_text[:40]!r}"
        )

    supplier_text = _normalized_text(_read_field(payload, "supplier_id"), "supplier_id")
    fields: dict[str, object] = {
        "supplier_id": supplier_text.upper(),
        "verdict": verdict_text,
        "rationale": _normalized_text(_read_field(payload, "rationale"), "rationale"),
        "findings": _normalized_items(
            _read_field(payload, "findings"), "findings", allow_empty=False
        ),
        "missing_evidence": _normalized_items(
            _read_field(payload, "missing_evidence"),
            "missing_evidence",
            allow_empty=True,
        ),
    }

    try:
        return SupplierRiskVerdict.model_validate(fields)
    except ValidationError as error:
        raise DecisionResponseError(_format_validation_error(error)) from None


def require_matching_supplier(
    verdict: SupplierRiskVerdict, requested_supplier_id: str
) -> SupplierRiskVerdict:
    """Reject a verdict that names a different supplier than the requested one.

    Raises:
        SupplierMismatchError: if the verdict's supplier id differs from the
            requested canonical id.
    """
    if verdict.supplier_id != requested_supplier_id:
        raise SupplierMismatchError(
            f"verdict names {verdict.supplier_id}, but {requested_supplier_id} was requested"
        )
    return verdict
