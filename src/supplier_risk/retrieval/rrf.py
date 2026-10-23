"""Reciprocal Rank Fusion for merging ranked retrieval lists.

``RRF(d) = sum(1 / (k + rank_i(d)))`` with one-based ranks. Ranks are positions
in each input list, never raw engine scores, because BM25 and cosine similarity
are not on a common scale. ``k`` defaults to the standard 60 and must be a
positive integer.

Ties are broken by ascending ``document_id`` so the merged order is fully
deterministic. A document that is absent from an input list receives no
contribution from that source and no fabricated score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from supplier_risk.retrieval.errors import RetrievalQueryError

DEFAULT_RRF_K: Final = 60


@dataclass(frozen=True, slots=True)
class FusionHit:
    """One fused document with its score and the per-source rank that produced it."""

    document_id: str
    score: float
    ranks: Mapping[str, int]


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> tuple[FusionHit, ...]:
    """Fuse named ranked document-id lists into one deterministic ranking.

    Args:
        ranked_lists: Mapping of source name (for example ``"dense"``) to document
            ids ordered best first. Empty lists are allowed.
        k: Positive RRF constant added to each one-based rank.
        top_k: Optional limit on returned documents.

    Raises:
        RetrievalQueryError: for an invalid ``k``, ``top_k``, empty source name,
            empty document id, or a duplicate id inside one ranked list.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise RetrievalQueryError("rrf k must be a positive integer")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise RetrievalQueryError("top_k must be a positive integer")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for source, document_ids in ranked_lists.items():
        if not isinstance(source, str) or not source:
            raise RetrievalQueryError("each ranked list must have a non-empty name")
        seen: set[str] = set()
        for position, document_id in enumerate(document_ids, start=1):
            if not isinstance(document_id, str) or not document_id:
                raise RetrievalQueryError("ranked document ids must be non-empty strings")
            if document_id in seen:
                raise RetrievalQueryError(
                    f"ranked list {source!r} contains duplicate document id {document_id!r}"
                )
            seen.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(document_id, {})[source] = position

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    return tuple(
        FusionHit(document_id=document_id, score=score, ranks=dict(ranks[document_id]))
        for document_id, score in ordered
    )
