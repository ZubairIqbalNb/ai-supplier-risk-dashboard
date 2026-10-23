"""Opt-in smoke tests for the real CPU sentence-transformer model.

These tests download ``sentence-transformers/all-MiniLM-L6-v2`` on first use and
are skipped unless ``SUPPLIER_RISK_REAL_MODEL_TESTS=1``. The required offline
suite never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from supplier_risk.data.generate import build_dataset
from supplier_risk.retrieval.corpus import build_corpus, corpus_fingerprint
from supplier_risk.retrieval.dense import FaissDenseIndex
from supplier_risk.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.environ.get("SUPPLIER_RISK_REAL_MODEL_TESTS") != "1",
        reason="set SUPPLIER_RISK_REAL_MODEL_TESTS=1 to run the local embedding model",
    ),
]


def test_real_model_returns_normalized_384_dimensional_vectors() -> None:
    embedder = SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)

    assert embedder.dimension == 384
    vectors = embedder.embed_documents(
        ["contract clause about quality inspection", "complaint about a late shipment"]
    )

    assert vectors.shape == (2, 384)
    assert bool(np.isfinite(vectors).all())
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_real_model_semantic_search_ranks_similar_documents_first() -> None:
    documents = build_corpus(build_dataset())
    embedder = SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)
    index = FaissDenseIndex.build(
        documents, embedder, corpus_fingerprint=corpus_fingerprint(documents)
    )

    hits = index.search(embedder.embed_query("late delivery of a shipment"), top_k=3)

    assert len(hits) == 3
    assert all(np.isfinite(hit.score) for hit in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert index.metadata.embedding_dimension == 384
    assert index.metadata.embedding_model == DEFAULT_EMBEDDING_MODEL
