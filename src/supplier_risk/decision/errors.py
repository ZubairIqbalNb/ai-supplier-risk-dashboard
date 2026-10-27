"""Project-level errors for the Module 6 decision layer.

The decision layer sits between the Module 5 orchestration result (plain
evidence strings) and the Module 7 optimizer/evaluation workflow, so its
failures must stay distinguishable:

- :class:`DecisionInputError`   the caller supplied an unusable identity.
- :class:`DecisionModelError`   the language-model call itself failed.
- :class:`DecisionResponseError` the model answered, but the structured output
  is missing or does not satisfy the verdict contract.
- :class:`SupplierMismatchError` the answer (or retrieval result) describes a
  different supplier than the one that was requested.

None of these errors is ever translated into an ``APPROVE`` verdict.
"""

from __future__ import annotations

import re
from typing import Final

MAX_ERROR_MESSAGE_LENGTH: Final = 300

_CREDENTIAL_URI: Final = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)[^@/\s]+@")
_SECRET_ASSIGNMENT: Final = re.compile(
    r"(?i)(?P<key>[A-Za-z0-9_]*(?:password|secret|token|api[_-]?key|credential)"
    r"[A-Za-z0-9_]*)\s*[=:]\s*\S+"
)


class DecisionError(Exception):
    """Base class for every Module 6 decision-layer failure."""


class DecisionInputError(DecisionError):
    """Raised when the requested supplier identity or input is unusable."""


class DecisionModelError(DecisionError):
    """Raised when the language-model call fails, times out, or is unconfigured."""


class DecisionResponseError(DecisionError):
    """Raised when model output is missing, malformed, or outside the contract."""


class SupplierMismatchError(DecisionResponseError):
    """Raised when an answer or retrieval result describes another supplier."""


def sanitize_error_message(text: str) -> str:
    """Return a bounded, credential-free copy of an exception message.

    Provider errors can embed request details, including keys or connection
    URIs. Only the exception type and this sanitized text are ever put into a
    project error or a log record.
    """
    redacted = _CREDENTIAL_URI.sub(r"\g<scheme>***@", text)
    redacted = _SECRET_ASSIGNMENT.sub(r"\g<key>=***", redacted)
    return redacted.replace("\n", " ").strip()[:MAX_ERROR_MESSAGE_LENGTH]
