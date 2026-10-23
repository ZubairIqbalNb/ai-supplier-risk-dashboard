"""Offline tests for the lazy CPU embedding model wrapper.

No test here downloads or loads a real model: the failing model class and the
absent package are both simulated in ``sys.modules``.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from supplier_risk.retrieval.embeddings import SentenceTransformerEmbedder
from supplier_risk.retrieval.errors import EmbeddingModelError


@pytest.fixture
def broken_model_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Install a fake ``sentence_transformers`` whose model constructor fails."""

    class BrokenModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("weights unavailable")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = BrokenModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    yield


def test_model_loading_failure_raises_embedding_model_error(
    broken_model_module: None,
) -> None:
    embedder = SentenceTransformerEmbedder("some/local-model")

    with pytest.raises(EmbeddingModelError, match="failed to load embedding model"):
        embedder.embed_query("quality inspection records")


def test_missing_sentence_transformers_package_raises_embedding_model_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ``None`` entry makes ``from sentence_transformers import ...`` raise ImportError.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = SentenceTransformerEmbedder("some/local-model")

    with pytest.raises(EmbeddingModelError, match="not installed"):
        embedder.embed_documents(["quality inspection records"])
