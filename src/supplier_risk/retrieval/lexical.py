"""Lexical retrieval with BM25 over the same document identities as FAISS.

The lexical corpus is persisted as the token lists produced by a documented,
deterministic tokenizer. ``BM25Okapi`` is rebuilt from those tokens on load, so
no pickle is involved and process restarts reproduce identical scores.

Tokenizer: lowercase, then keep maximal runs of ASCII letters and digits,
treating ``-``, ``_``, and ``.`` as joiners inside a token. Identifiers such as
``con-001-01``, ``sup-007``, ``lot-8399``, ``fy2025``, and ISO dates are
preserved as single tokens instead of being split into meaningless fragments.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rank_bm25 import BM25Okapi

from supplier_risk.retrieval.corpus import CorpusDocument
from supplier_risk.retrieval.errors import (
    CorpusError,
    RetrievalArtifactError,
    RetrievalQueryError,
)
from supplier_risk.retrieval.storage import (
    ARTIFACT_VERSION,
    BUILD_COMMAND,
    LEXICAL_FILENAME,
    ArtifactVersion,
    describe_validation_error,
    read_artifact_text,
    write_json,
)
from supplier_risk.retrieval.validation import validate_query_text, validate_top_k

TOKENIZER_VERSION: Final = "module3.v1"
DEFAULT_BM25_K1: Final = 1.5
DEFAULT_BM25_B: Final = 0.75

_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> tuple[str, ...]:
    """Return the deterministic BM25 token sequence for ``text``."""
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


@dataclass(frozen=True, slots=True)
class LexicalHit:
    """One BM25 result. Scores are BM25 values and are not comparable to cosine scores."""

    document_id: str
    score: float


class LexicalPayload(BaseModel):
    """Persisted lexical corpus: token lists in canonical document order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_version: ArtifactVersion
    corpus_fingerprint: str = Field(min_length=64, max_length=64)
    tokenizer_version: Literal["module3.v1"]
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
    document_ids: tuple[str, ...] = Field(min_length=1)
    tokens: tuple[tuple[str, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _document_ids_align_with_tokens(self) -> Self:
        if len(self.document_ids) != len(self.tokens):
            raise ValueError("document_ids and tokens must have the same length")
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("document_ids must be unique")
        if any(not document_tokens for document_tokens in self.tokens):
            raise ValueError("every document must have at least one token")
        return self


class LexicalIndex:
    """BM25 index over the persisted corpus, reconstructable from JSON."""

    def __init__(self, payload: LexicalPayload, bm25: BM25Okapi) -> None:
        """Wrap a rebuilt BM25 model; use :meth:`build` or :meth:`load_from` instead."""
        self._payload = payload
        self._bm25 = bm25
        self._token_sets = tuple(frozenset(tokens) for tokens in payload.tokens)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return self._payload.document_ids

    @property
    def size(self) -> int:
        return len(self._payload.document_ids)

    @property
    def corpus_fingerprint(self) -> str:
        return self._payload.corpus_fingerprint

    @property
    def tokenizer_version(self) -> str:
        return self._payload.tokenizer_version

    @property
    def bm25_k1(self) -> float:
        return self._payload.bm25_k1

    @property
    def bm25_b(self) -> float:
        return self._payload.bm25_b

    @classmethod
    def build(
        cls,
        documents: Sequence[CorpusDocument],
        *,
        corpus_fingerprint: str,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> LexicalIndex:
        """Tokenize ``documents`` in order and build a BM25 index."""
        if not documents:
            raise CorpusError("cannot build a lexical index from an empty corpus")
        if not math.isfinite(k1) or k1 <= 0:
            raise CorpusError("bm25 k1 must be a positive finite number")
        if not math.isfinite(b) or not 0.0 <= b <= 1.0:
            raise CorpusError("bm25 b must be between 0 and 1")

        tokens = tuple(tokenize(document.text) for document in documents)
        for document, document_tokens in zip(documents, tokens, strict=True):
            if not document_tokens:
                raise CorpusError(
                    f"document {document.document_id} contains no searchable tokens"
                )

        payload = LexicalPayload(
            artifact_version=ARTIFACT_VERSION,
            corpus_fingerprint=corpus_fingerprint,
            tokenizer_version=TOKENIZER_VERSION,
            bm25_k1=k1,
            bm25_b=b,
            document_ids=tuple(document.document_id for document in documents),
            tokens=tokens,
        )
        bm25 = BM25Okapi([list(document_tokens) for document_tokens in tokens], k1=k1, b=b)
        return cls(payload, bm25)

    def write_to(self, directory: Path) -> None:
        """Persist the lexical corpus and BM25 parameters."""
        write_json(directory / LEXICAL_FILENAME, self._payload.model_dump(mode="json"))

    @classmethod
    def load_from(cls, directory: Path) -> LexicalIndex:
        """Reload the lexical corpus and rebuild BM25 from its persisted tokens."""
        path = directory / LEXICAL_FILENAME
        text = read_artifact_text(path, description="lexical index")
        try:
            payload = LexicalPayload.model_validate_json(text)
        except ValidationError as error:
            detail = describe_validation_error(error)
            raise RetrievalArtifactError(
                f"lexical index at {path} is invalid ({detail}); rebuild it with "
                f"`{BUILD_COMMAND}`."
            ) from None

        bm25 = BM25Okapi(
            [list(document_tokens) for document_tokens in payload.tokens],
            k1=payload.bm25_k1,
            b=payload.bm25_b,
        )
        return cls(payload, bm25)

    def verify_tokens(self, documents: Sequence[CorpusDocument]) -> None:
        """Reject persisted tokens that do not match the canonical corpus text.

        ``documents`` must be in the same order as the persisted document ids.
        Recomputing the tokens is cheap and keeps the lexical artifact from
        silently diverging from the corpus it claims to represent.
        """
        expected = tuple(tokenize(document.text) for document in documents)
        if expected != self._payload.tokens:
            raise RetrievalArtifactError(
                "lexical index tokens do not match the persisted corpus text; "
                f"rebuild the artifacts with `{BUILD_COMMAND}`."
            )

    def search(self, query: str, *, top_k: int) -> tuple[LexicalHit, ...]:
        """Return BM25-ranked documents that contain at least one query token."""
        text = validate_query_text(query)
        limit = validate_top_k(top_k)
        query_tokens = tokenize(text)
        if not query_tokens:
            raise RetrievalQueryError("query contains no searchable terms")

        scores = np.asarray(self._bm25.get_scores(list(query_tokens)), dtype=np.float64)
        if scores.shape != (self.size,):
            raise RetrievalArtifactError(
                "lexical index scores do not match the persisted document count; "
                f"rebuild it with `{BUILD_COMMAND}`."
            )

        wanted = set(query_tokens)
        matches = [
            position
            for position, document_tokens in enumerate(self._token_sets)
            if wanted & document_tokens
        ]
        ordered = sorted(matches, key=lambda position: (-float(scores[position]), position))
        return tuple(
            LexicalHit(
                document_id=self._payload.document_ids[position],
                score=float(scores[position]),
            )
            for position in ordered[:limit]
        )
