"""Shared validation for retrieval queries, result limits, and metadata filters."""

from __future__ import annotations

import re
from typing import Final

from supplier_risk.contracts import Partition
from supplier_risk.retrieval.errors import RetrievalQueryError

PARTITIONS: Final[tuple[Partition, ...]] = ("train", "dev", "held_out")
_SUPPLIER_ID_PATTERN: Final = re.compile(r"^SUP-\d{3}$")


def validate_query_text(value: str) -> str:
    """Return the stripped query text or raise for a blank or non-string query."""
    if not isinstance(value, str):
        raise RetrievalQueryError("query must be a string")
    text = value.strip()
    if not text:
        raise RetrievalQueryError("query must not be empty")
    return text


def validate_top_k(value: int) -> int:
    """Return ``value`` when it is a positive integer, otherwise raise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RetrievalQueryError("top_k must be a positive integer")
    return value


def validate_partition_filter(value: Partition | None) -> Partition | None:
    """Return a valid partition filter or raise."""
    if value is None:
        return None
    if value not in PARTITIONS:
        allowed = ", ".join(PARTITIONS)
        raise RetrievalQueryError(f"partition filter must be one of: {allowed}")
    return value


def validate_supplier_filter(value: str | None) -> str | None:
    """Return a valid supplier id filter or raise."""
    if value is None:
        return None
    if not isinstance(value, str) or _SUPPLIER_ID_PATTERN.fullmatch(value) is None:
        raise RetrievalQueryError("supplier_id filter must match the format SUP-NNN")
    return value
