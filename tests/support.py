"""Shared, offline test doubles for the Module 3 retrieval tests.

Importing this module never loads a model or touches the network. ``conftest``
re-exports the fixtures built on these doubles, while tests that need a class
directly import it from here so the suite works with any pytest import mode.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from supplier_risk.retrieval.lexical import tokenize

TEST_EMBEDDING_MODEL = "stub-embedder"
TEST_EMBEDDING_DIMENSION = 24


class StubEmbedder:
    """Deterministic hashed-bag-of-tokens embedder used in place of a real model."""

    def __init__(
        self,
        *,
        dimension: int = TEST_EMBEDDING_DIMENSION,
        model_name: str = TEST_EMBEDDING_MODEL,
    ) -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def declared_dimension(self) -> int:
        """Known without loading a model, so the retriever can check it at load time."""
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return np.asarray(self._vector(text), dtype=np.float32)

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self._dimension, dtype=np.float64)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return list(vector / norm) if norm else list(vector)


class MatrixEmbedder:
    """Embedder returning fixed vectors, for hand-computed similarity tests."""

    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        query: Sequence[float],
        model_name: str = TEST_EMBEDDING_MODEL,
    ) -> None:
        self._matrix = np.asarray(vectors, dtype=np.float32)
        self._query = np.asarray(query, dtype=np.float32)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._matrix.shape[1])

    @property
    def declared_dimension(self) -> int:
        return int(self._matrix.shape[1])

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if len(texts) != self._matrix.shape[0]:
            raise AssertionError("fixture vectors must match the number of documents")
        return self._matrix

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._query
