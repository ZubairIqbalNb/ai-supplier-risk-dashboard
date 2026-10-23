from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import pytest
from numpy.typing import NDArray

from supplier_risk.retrieval.corpus import CorpusDocument, corpus_fingerprint
from supplier_risk.retrieval.dense import FaissDenseIndex
from supplier_risk.retrieval.errors import (
    CorpusError,
    EmbeddingModelError,
    RetrievalArtifactError,
    RetrievalQueryError,
)
from tests.support import MatrixEmbedder, StubEmbedder


def _build(documents: tuple[CorpusDocument, ...], embedder: StubEmbedder) -> FaissDenseIndex:
    return FaissDenseIndex.build(
        documents, embedder, corpus_fingerprint=corpus_fingerprint(documents)
    )


def test_index_dimension_vector_count_and_metadata(
    corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    index = _build(corpus_documents, stub_embedder)

    assert index.dimension == stub_embedder.dimension == 24
    assert index.size == len(corpus_documents) == 66
    assert index.embedding_model == "stub-embedder"
    assert index.metadata.index_type == "IndexFlatIP"
    assert index.metadata.metric == "cosine_inner_product"
    assert index.metadata.normalized is True
    assert index.document_ids == tuple(document.document_id for document in corpus_documents)


def test_similarity_ranking_matches_hand_computed_cosine(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    documents = corpus_documents[:3]
    embedder = MatrixEmbedder(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        query=[0.6, 0.8, 0.0],
    )
    index = _build(documents, embedder)  # type: ignore[arg-type]

    hits = index.search(embedder.embed_query("ignored"), top_k=3)

    assert [hit.document_id for hit in hits] == [
        documents[1].document_id,
        documents[0].document_id,
        documents[2].document_id,
    ]
    assert hits[0].score == pytest.approx(0.8, abs=1e-6)
    assert hits[1].score == pytest.approx(0.6, abs=1e-6)
    assert hits[2].score == pytest.approx(0.0, abs=1e-6)


def test_vector_ids_map_to_the_correct_documents(
    corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    index = _build(corpus_documents, stub_embedder)
    position = 5
    document = corpus_documents[position]

    hit = index.search(stub_embedder.embed_query(document.text), top_k=1)[0]

    assert hit.vector_id == position
    assert hit.document_id == document.document_id
    entry = index.metadata.vectors[position]
    assert entry.document_id == document.document_id
    assert entry.supplier_id == document.supplier_id
    assert entry.partition == document.partition
    assert entry.source_type == document.source_type


def test_persisted_index_reloads_with_matching_metadata(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    index = _build(corpus_documents, stub_embedder)
    index.write_to(tmp_path)

    loaded = FaissDenseIndex.load_from(tmp_path)
    query = stub_embedder.embed_query("quality procedures inspection records")

    assert loaded.document_ids == index.document_ids
    assert loaded.metadata == index.metadata
    assert loaded.search(query, top_k=5) == index.search(query, top_k=5)


def test_missing_dense_artifacts_raise(tmp_path: Path) -> None:
    with pytest.raises(RetrievalArtifactError, match="missing dense index metadata"):
        FaissDenseIndex.load_from(tmp_path)


def test_corrupt_index_raises(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    _build(corpus_documents, stub_embedder).write_to(tmp_path)
    (tmp_path / "dense.index.faiss").write_bytes(b"not a faiss index")

    with pytest.raises(RetrievalArtifactError, match="unreadable or corrupt"):
        FaissDenseIndex.load_from(tmp_path)


def test_non_inner_product_index_is_rejected(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    _build(corpus_documents, stub_embedder).write_to(tmp_path)
    faiss.write_index(
        faiss.IndexFlatL2(stub_embedder.dimension),
        str(tmp_path / "dense.index.faiss"),
    )

    with pytest.raises(RetrievalArtifactError, match="not an inner-product index"):
        FaissDenseIndex.load_from(tmp_path)


def test_index_vector_count_mismatch_raises(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    _build(corpus_documents, stub_embedder).write_to(tmp_path)
    smaller = tmp_path / "smaller"
    _build(corpus_documents[:-1], stub_embedder).write_to(smaller)
    shutil.copy(smaller / "dense.index.faiss", tmp_path / "dense.index.faiss")

    with pytest.raises(RetrievalArtifactError, match="contains 65 vectors"):
        FaissDenseIndex.load_from(tmp_path)


def test_metadata_vector_count_mismatch_raises(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    _build(corpus_documents, stub_embedder).write_to(tmp_path)
    path = tmp_path / "dense.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vectors"] = payload["vectors"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="lists 65 vectors"):
        FaissDenseIndex.load_from(tmp_path)


def test_dimension_mismatch_raises(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    _build(corpus_documents, stub_embedder).write_to(tmp_path)
    path = tmp_path / "dense.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["embedding_dimension"] = 8
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="dimension 24"):
        FaissDenseIndex.load_from(tmp_path)


def test_duplicate_document_ids_in_metadata_are_rejected(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    _build(corpus_documents, stub_embedder).write_to(tmp_path)
    path = tmp_path / "dense.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vectors"][1]["document_id"] = payload["vectors"][0]["document_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="same document"):
        FaissDenseIndex.load_from(tmp_path)


def test_vector_id_must_match_its_position(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    _build(corpus_documents, stub_embedder).write_to(tmp_path)
    path = tmp_path / "dense.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vectors"][1]["vector_id"] = 5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="vector ids must match"):
        FaissDenseIndex.load_from(tmp_path)


def test_non_finite_embeddings_are_rejected(corpus_documents: tuple[CorpusDocument, ...]) -> None:
    embedder = MatrixEmbedder([[np.nan, 0.0]], query=[1.0, 0.0])

    with pytest.raises(EmbeddingModelError, match="non-finite"):
        _build(corpus_documents[:1], embedder)  # type: ignore[arg-type]


class _WrongShapeEmbedder:
    model_name = "stub-embedder"
    dimension = 3

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return np.ones(2, dtype=np.float32)


def test_embedding_shape_mismatch_is_rejected(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    with pytest.raises(EmbeddingModelError, match="expected"):
        _build(corpus_documents[:2], _WrongShapeEmbedder())  # type: ignore[arg-type]


def test_zero_length_embedding_is_rejected(corpus_documents: tuple[CorpusDocument, ...]) -> None:
    embedder = MatrixEmbedder([[0.0, 0.0]], query=[1.0, 0.0])

    with pytest.raises(EmbeddingModelError, match="zero-length"):
        _build(corpus_documents[:1], embedder)  # type: ignore[arg-type]


def test_empty_corpus_is_rejected(stub_embedder: StubEmbedder) -> None:
    with pytest.raises(CorpusError, match="empty corpus"):
        FaissDenseIndex.build((), stub_embedder, corpus_fingerprint=corpus_fingerprint(()))


def test_search_rejects_a_query_with_the_wrong_dimension(
    corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    index = _build(corpus_documents, stub_embedder)

    with pytest.raises(RetrievalQueryError, match="the index expects 24"):
        index.search(np.ones(7, dtype=np.float32), top_k=3)


def test_search_rejects_a_zero_query_vector(
    corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    index = _build(corpus_documents, stub_embedder)

    with pytest.raises(RetrievalQueryError, match="zero length"):
        index.search(np.zeros(24, dtype=np.float32), top_k=3)


@pytest.mark.parametrize("top_k", [0, -3, True])
def test_search_rejects_invalid_top_k(
    corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder, top_k: object
) -> None:
    index = _build(corpus_documents, stub_embedder)

    with pytest.raises(RetrievalQueryError, match="top_k"):
        index.search(stub_embedder.embed_query("quality"), top_k=top_k)  # type: ignore[arg-type]


def test_search_returns_at_most_the_corpus_size(
    corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    index = _build(corpus_documents, stub_embedder)

    hits = index.search(stub_embedder.embed_query("quality"), top_k=10_000)

    assert len(hits) == len(corpus_documents)


def test_copy_of_artifacts_can_be_reloaded_independently(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...], stub_embedder: StubEmbedder
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _build(corpus_documents, stub_embedder).write_to(source)
    shutil.copytree(source, target)

    assert FaissDenseIndex.load_from(target).document_ids == (
        FaissDenseIndex.load_from(source).document_ids
    )
