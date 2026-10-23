from __future__ import annotations

import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from supplier_risk.data.generate import build_dataset, write_dataset
from supplier_risk.retrieval import pipeline as pipeline_module
from supplier_risk.retrieval.build import build_retrieval_artifacts
from supplier_risk.retrieval.build import main as build_main
from supplier_risk.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from supplier_risk.retrieval.errors import (
    RetrievalArtifactError,
    RetrievalQueryError,
    UnknownSupplierError,
)
from supplier_risk.retrieval.manifest import load_manifest
from supplier_risk.retrieval.pipeline import (
    HybridRetriever,
    get_default_retriever,
    query_faiss,
    reset_default_retriever,
    resolve_supplier_id,
)
from tests.support import StubEmbedder


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("retrieval")
    dataset_path = write_dataset(build_dataset(), root / "suppliers.json")
    build_retrieval_artifacts(
        dataset_path=dataset_path,
        artifacts_dir=root / "artifacts",
        embedder=StubEmbedder(),
    )
    return root / "artifacts"


@pytest.fixture
def retriever(artifact_dir: Path) -> HybridRetriever:
    return HybridRetriever.load(artifact_dir, embedder=StubEmbedder())


def test_reloaded_retriever_exposes_corpus_and_index_identity(
    retriever: HybridRetriever,
) -> None:
    assert retriever.document_count == 66
    assert retriever.embedding_model == "stub-embedder"
    assert retriever.rrf_k == 60
    assert {document.source_type for document in retriever.documents} == {
        "contract",
        "complaint",
    }
    assert all(document.supplier_id for document in retriever.documents)
    assert all(
        document.partition in {"train", "dev", "held_out"}
        for document in retriever.documents
    )


def test_hybrid_query_returns_documents_with_ranking_metadata(
    retriever: HybridRetriever,
) -> None:
    hits = retriever.query("quality procedures inspection records", top_k=5)

    assert hits
    for hit in hits:
        assert hit.fusion_score > 0
        assert hit.dense_rank is not None or hit.lexical_rank is not None
        assert hit.dense_rank is None or hit.dense_rank >= 1
        assert hit.lexical_rank is None or hit.lexical_rank >= 1
        assert hit.document.document_id in {
            document.document_id for document in retriever.documents
        }
    scores = [hit.fusion_score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_query_merges_both_rankers(retriever: HybridRetriever) -> None:
    hits = retriever.query("quality procedures inspection records", top_k=10)

    assert any(hit.dense_rank is not None for hit in hits)
    assert any(hit.lexical_rank is not None for hit in hits)


def test_supplier_filter_excludes_other_suppliers(retriever: HybridRetriever) -> None:
    for supplier_id in ("SUP-001", "SUP-014", "SUP-024"):
        hits = retriever.query("quality delivery complaint audit", top_k=3, supplier_id=supplier_id)

        assert hits, supplier_id
        assert {hit.document.supplier_id for hit in hits} == {supplier_id}


def test_partition_filter_excludes_other_partitions(retriever: HybridRetriever) -> None:
    for partition in ("train", "dev", "held_out"):
        hits = retriever.query("quality delivery complaint audit", top_k=3, partition=partition)

        assert hits, partition
        assert {hit.document.partition for hit in hits} == {partition}


def test_combined_filters_are_applied_together(retriever: HybridRetriever) -> None:
    supplier_id = "SUP-019"
    partition = next(
        document.partition for document in retriever.documents if document.supplier_id == supplier_id
    )

    hits = retriever.query(
        "packaging shipment", top_k=5, supplier_id=supplier_id, partition=partition
    )

    assert hits
    assert all(
        hit.document.supplier_id == supplier_id and hit.document.partition == partition
        for hit in hits
    )


def test_contradictory_filters_return_no_documents(retriever: HybridRetriever) -> None:
    assert retriever.query("quality", top_k=3, supplier_id="SUP-001", partition="held_out") == ()


def test_supplier_filter_returns_every_supplier_document(retriever: HybridRetriever) -> None:
    supplier_id = "SUP-024"
    expected = {
        document.document_id
        for document in retriever.documents
        if document.supplier_id == supplier_id
    }

    hits = retriever.query("supplier", top_k=50, supplier_id=supplier_id)

    assert {hit.document.document_id for hit in hits} == expected


def test_no_cross_supplier_leakage(retriever: HybridRetriever) -> None:
    for supplier_id in sorted({document.supplier_id for document in retriever.documents}):
        hits = retriever.query(
            "supplier quality complaint delivery audit", top_k=5, supplier_id=supplier_id
        )

        assert hits, supplier_id
        assert all(hit.document.supplier_id == supplier_id for hit in hits)


def test_filters_are_stable_across_independent_reloads(artifact_dir: Path) -> None:
    first = HybridRetriever.load(artifact_dir, embedder=StubEmbedder())
    second = HybridRetriever.load(artifact_dir, embedder=StubEmbedder())

    first_hits = first.query("quality inspection", top_k=3, supplier_id="SUP-005")
    second_hits = second.query("quality inspection", top_k=3, supplier_id="SUP-005")

    assert [hit.document.document_id for hit in first_hits] == [
        hit.document.document_id for hit in second_hits
    ]
    assert {hit.document.supplier_id for hit in second_hits} == {"SUP-005"}


@pytest.mark.parametrize("query", ["", "   ", "!!!"])
def test_invalid_queries_are_rejected(retriever: HybridRetriever, query: str) -> None:
    with pytest.raises(RetrievalQueryError):
        retriever.query(query, top_k=3)


@pytest.mark.parametrize("top_k", [0, -1, True, 2.5])
def test_invalid_top_k_is_rejected(retriever: HybridRetriever, top_k: object) -> None:
    with pytest.raises(RetrievalQueryError, match="top_k"):
        retriever.query("quality", top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("supplier_id", ["SUP-1", "sup-001", "001", ""])
def test_invalid_supplier_filter_is_rejected(
    retriever: HybridRetriever, supplier_id: str
) -> None:
    with pytest.raises(RetrievalQueryError, match="supplier_id"):
        retriever.query("quality", supplier_id=supplier_id)


def test_invalid_partition_filter_is_rejected(retriever: HybridRetriever) -> None:
    with pytest.raises(RetrievalQueryError, match="partition"):
        retriever.query("quality", partition="validation")  # type: ignore[arg-type]


def test_missing_artifacts_raise(tmp_path: Path) -> None:
    with pytest.raises(RetrievalArtifactError, match="missing retrieval manifest"):
        HybridRetriever.load(tmp_path)


def test_manifest_fingerprint_mismatch_is_detected(artifact_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, copied)
    path = copied / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["corpus_fingerprint"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="fingerprints do not match"):
        HybridRetriever.load(copied, embedder=StubEmbedder())


def test_lexical_document_order_mismatch_is_detected(artifact_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, copied)
    path = copied / "lexical.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["document_ids"][0], payload["document_ids"][1] = (
        payload["document_ids"][1],
        payload["document_ids"][0],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="document order does not match"):
        HybridRetriever.load(copied, embedder=StubEmbedder())


def test_matching_injected_embedder_loads_and_queries(artifact_dir: Path) -> None:
    retriever = HybridRetriever.load(artifact_dir, embedder=StubEmbedder())

    hits = retriever.query("quality inspection", top_k=3)

    assert hits
    assert retriever.embedding_model == "stub-embedder"


def test_injected_embedder_with_a_different_model_is_rejected(artifact_dir: Path) -> None:
    with pytest.raises(RetrievalArtifactError, match="does not match the persisted index model"):
        HybridRetriever.load(artifact_dir, embedder=StubEmbedder(model_name="other-model"))


def test_injected_embedder_with_a_different_dimension_is_rejected(artifact_dir: Path) -> None:
    with pytest.raises(
        RetrievalArtifactError, match="does not match the persisted index dimension"
    ):
        HybridRetriever.load(artifact_dir, embedder=StubEmbedder(dimension=7))


def test_injected_embedder_without_a_declared_dimension_fails_at_query_time(
    artifact_dir: Path,
) -> None:
    class NoDeclaredDimensionEmbedder:
        model_name = "stub-embedder"

        def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
            raise AssertionError("query-only embedder")

        def embed_query(self, text: str) -> NDArray[np.float32]:
            return np.ones(7, dtype=np.float32)

    retriever = HybridRetriever.load(artifact_dir, embedder=NoDeclaredDimensionEmbedder())

    with pytest.raises(RetrievalQueryError, match="the index expects 24"):
        retriever.query("quality inspection", top_k=3)


def test_default_embedder_is_lazy_and_uses_the_manifest_model(
    artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []

    class SpyEmbedder(SentenceTransformerEmbedder):
        def __init__(
            self, model_name: str = DEFAULT_EMBEDDING_MODEL, **kwargs: object
        ) -> None:
            requested.append(model_name)
            super().__init__(model_name)

    monkeypatch.setattr(pipeline_module, "SentenceTransformerEmbedder", SpyEmbedder)

    retriever = HybridRetriever.load(artifact_dir)

    assert requested == [load_manifest(artifact_dir).embedding_model]
    # The model is constructed but not loaded until the first query.
    assert retriever._embedder._model is None  # type: ignore[attr-defined]


def test_lexical_token_mismatch_is_detected(artifact_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, copied)
    path = copied / "lexical.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tokens"][0][0] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="tokens do not match"):
        HybridRetriever.load(copied, embedder=StubEmbedder())


@pytest.mark.parametrize(
    ("field", "tampered"),
    [("supplier_id", "SUP-999"), ("partition", "held_out"), ("source_type", "complaint")],
)
def test_dense_metadata_identity_mismatch_is_detected(
    artifact_dir: Path, tmp_path: Path, field: str, tampered: str
) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, copied)
    path = copied / "dense.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["vectors"][0][field] != tampered
    payload["vectors"][0][field] = tampered
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="identity fields do not match"):
        HybridRetriever.load(copied, embedder=StubEmbedder())


def test_reset_default_retriever_clears_the_cache(artifact_dir: Path) -> None:
    first = get_default_retriever(artifact_dir)

    assert get_default_retriever(artifact_dir) is first

    reset_default_retriever()

    assert get_default_retriever(artifact_dir) is not first
    reset_default_retriever()


def test_build_cli_reports_a_missing_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts_dir = tmp_path / "artifacts"

    exit_code = build_main(
        ["--dataset", str(tmp_path / "absent.json"), "--artifacts-dir", str(artifacts_dir)]
    )

    assert exit_code == 1
    assert "supplier dataset not found" in capsys.readouterr().err
    assert not artifacts_dir.exists()


def test_build_cli_reports_an_invalid_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_path = tmp_path / "suppliers.json"
    dataset_path.write_text('{"seed": 1}', encoding="utf-8")
    artifacts_dir = tmp_path / "artifacts"

    exit_code = build_main(
        ["--dataset", str(dataset_path), "--artifacts-dir", str(artifacts_dir)]
    )

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err
    assert not artifacts_dir.exists()


def test_embedding_model_mismatch_is_detected(artifact_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, copied)
    path = copied / "dense.metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["embedding_model"] = "some-other-model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="embedding model"):
        HybridRetriever.load(copied, embedder=StubEmbedder())


def test_supplier_resolution_accepts_ids_and_names(retriever: HybridRetriever) -> None:
    assert resolve_supplier_id("SUP-007", retriever.documents) == "SUP-007"
    assert resolve_supplier_id(" sup-007 ", retriever.documents) == "SUP-007"
    assert (
        resolve_supplier_id("sakurajima precision kk", retriever.documents) == "SUP-007"
    )


@pytest.mark.parametrize("entity", ["SUP-099", "Unknown Holdings", "", "   "])
def test_unknown_supplier_identity_is_rejected(
    retriever: HybridRetriever, entity: str
) -> None:
    with pytest.raises(UnknownSupplierError):
        resolve_supplier_id(entity, retriever.documents)


def test_query_faiss_returns_labeled_supplier_context(retriever: HybridRetriever) -> None:
    context = query_faiss("SUP-001", retriever=retriever, top_k=3)

    assert context.startswith("[FAISS] ")
    assert "supplier=SUP-001" in context
    assert "Norvend Components Ltd" in context
    assert "partition=train" in context
    assert "scope=contracts+complaints" in context
    assert "document=contract:CON-001-01" in context
    assert "source=contract:CON-001-01" in context
    lowered = context.lower()
    assert "sql" not in lowered
    assert "neo4j" not in lowered
    assert "graph" not in lowered


def test_query_faiss_accepts_a_legal_name_case_insensitively(
    retriever: HybridRetriever,
) -> None:
    context = query_faiss("norvend components ltd", retriever=retriever)

    assert "supplier=SUP-001" in context


def test_query_faiss_only_returns_the_requested_supplier(
    retriever: HybridRetriever,
) -> None:
    context = query_faiss("SUP-019", retriever=retriever, top_k=5)

    assert "supplier=SUP-019" in context
    document_lines = [
        line for line in context.splitlines() if re.match(r"^\d+\. document=", line)
    ]
    assert document_lines
    for line in document_lines:
        assert "document=contract:CON-019-" in line or "document=complaint:CMP-019-" in line
    assert "SUP-001" not in context


def test_query_faiss_reports_unknown_suppliers_without_raising(
    retriever: HybridRetriever,
) -> None:
    context = query_faiss("Unknown Holdings", retriever=retriever)

    assert context.startswith("[FAISS ERROR]")
    assert "unknown supplier identity" in context


def test_query_faiss_reports_missing_artifacts_without_raising(tmp_path: Path) -> None:
    context = query_faiss("SUP-001", artifacts_dir=tmp_path / "absent")

    assert context.startswith("[FAISS ERROR]")
    assert "document retrieval unavailable" in context
