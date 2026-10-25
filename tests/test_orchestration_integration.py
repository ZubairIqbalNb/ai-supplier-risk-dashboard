"""Integration tests for Module 5 orchestration.

The offline tests here run the real ``query_sql``, ``query_faiss`` (hybrid FAISS
+ BM25 over real artifacts built with the deterministic stub embedder), and
``query_graph`` wrappers together. No model download, network access, or Neo4j
service is required. The live Neo4j test is opt-in with
``SUPPLIER_RISK_NEO4J_TESTS=1``.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path

import pytest

from supplier_risk.config import load_neo4j_settings
from supplier_risk.contracts import SupplierDataset
from supplier_risk.data.generate import write_dataset
from supplier_risk.graph.errors import GraphConnectionError
from supplier_risk.graph.ingest import ingest_dataset
from supplier_risk.graph.pipeline import query_graph
from supplier_risk.graph.schema import (
    OwnershipEvidenceRow,
    SupplierIdentity,
)
from supplier_risk.graph.store import Neo4jGraphStore
from supplier_risk.orchestration import query_all_sources
from supplier_risk.retrieval.build import build_retrieval_artifacts
from supplier_risk.retrieval.corpus import CorpusDocument
from supplier_risk.retrieval.pipeline import HybridRetriever, RetrievalHit, query_faiss
from supplier_risk.sql.build import build_from_dataset
from supplier_risk.sql.pipeline import query_sql
from tests.support import StubEmbedder

DATASET_PATH = Path("data/generated/suppliers.json")


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory, module2_dataset: SupplierDataset) -> Path:
    path = tmp_path_factory.mktemp("orchestration-sql") / "suppliers.sqlite"
    return build_from_dataset(module2_dataset, path).database_path


@pytest.fixture(scope="module")
def retriever(
    tmp_path_factory: pytest.TempPathFactory, module2_dataset: SupplierDataset
) -> HybridRetriever:
    root = tmp_path_factory.mktemp("orchestration-retrieval")
    dataset_path = write_dataset(module2_dataset, root / "suppliers.json")
    artifacts_dir = root / "artifacts"
    build_retrieval_artifacts(
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        embedder=StubEmbedder(),
    )
    return HybridRetriever.load(artifacts_dir, embedder=StubEmbedder())


class FakeGraphStore:
    """Minimal store double for the real :func:`query_graph` wrapper."""

    def __init__(
        self,
        identity: SupplierIdentity | None,
        rows: Sequence[OwnershipEvidenceRow] = (),
        error: Exception | None = None,
    ) -> None:
        self._identity = identity
        self._rows = tuple(rows)
        self._error = error

    def resolve_supplier(self, entity: str) -> SupplierIdentity | None:
        if self._error is not None:
            raise self._error
        return self._identity

    def fetch_ownership_evidence(self, supplier_id: str) -> tuple[OwnershipEvidenceRow, ...]:
        if self._error is not None:
            raise self._error
        return self._rows


class NoHitRetriever:
    """Real-retriever surface that returns no fused documents."""

    def __init__(self, documents: Sequence[CorpusDocument], *, rrf_k: int = 60) -> None:
        self._documents = tuple(documents)
        self._rrf_k = rrf_k

    @property
    def documents(self) -> tuple[CorpusDocument, ...]:
        return self._documents

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        supplier_id: str | None = None,
        partition: str | None = None,
    ) -> tuple[RetrievalHit, ...]:
        return ()


def _identity(module2_dataset: SupplierDataset, supplier_id: str) -> SupplierIdentity:
    record = next(
        item for item in module2_dataset.suppliers if item.profile.supplier_id == supplier_id
    )
    return SupplierIdentity(
        supplier_id=record.profile.supplier_id,
        legal_name=record.profile.legal_name,
        partition=record.partition,
        sector=record.profile.sector,
        country_code=record.profile.country_code,
    )


def _ownership_row(module2_dataset: SupplierDataset, supplier_id: str) -> OwnershipEvidenceRow:
    record = next(
        item for item in module2_dataset.suppliers if item.profile.supplier_id == supplier_id
    )
    link = record.ownership
    assert link is not None
    return OwnershipEvidenceRow(
        child_key=link.child_supplier_id,
        child_name=record.profile.legal_name,
        parent_key=link.parent_company_id,
        parent_name=link.parent_company_name,
        relationship_type=link.relationship_type,
        ownership_percentage=link.ownership_percentage,
        hops=1,
    )


def test_orchestrator_combines_the_three_real_engine_contexts(
    database: Path, retriever: HybridRetriever, module2_dataset: SupplierDataset
) -> None:
    result = query_all_sources(
        "SUP-001",
        sources={
            "sql": partial(query_sql, database_path=database),
            "faiss": partial(query_faiss, retriever=retriever),
            "graph": partial(
                query_graph,
                store=FakeGraphStore(
                    _identity(module2_dataset, "SUP-001"),
                    (_ownership_row(module2_dataset, "SUP-001"),),
                ),
            ),
        },
    )

    assert result.outcome == "complete"
    assert [source.status for source in result.sources] == ["ok", "ok", "ok"]

    # Each source contributes different facts for the same entity.
    assert "on_time_delivery_rate=" in result.sql.context
    assert "document=" in result.faiss.context
    assert "ownership_relationships=1" in result.graph.context
    assert "ORG-" in result.graph.context

    # Provenance is preserved and no source claims another source's evidence.
    assert "ORG-" not in result.sql.context
    assert "document=" not in result.sql.context
    assert "document=" not in result.graph.context
    assert "on_time_delivery_rate" not in result.faiss.context
    assert "on_time_delivery_rate" not in result.graph.context
    assert "SUP-002" not in result.faiss.context


def test_graph_failure_keeps_real_sql_and_faiss_evidence(
    database: Path, retriever: HybridRetriever, module2_dataset: SupplierDataset
) -> None:
    result = query_all_sources(
        "SUP-002",
        sources={
            "sql": partial(query_sql, database_path=database),
            "faiss": partial(query_faiss, retriever=retriever),
            "graph": partial(
                query_graph,
                store=FakeGraphStore(
                    None, error=GraphConnectionError("Neo4j is unavailable")
                ),
            ),
        },
    )

    assert result.outcome == "partial"
    assert result.failed_sources == ("graph",)
    assert result.sql.status == "ok"
    assert result.faiss.status == "ok"
    assert result.graph.status == "error"
    assert result.graph.context.startswith("[GRAPH ERROR] graph retrieval unavailable")
    assert "on_time_delivery_rate=" in result.sql.context
    assert "document=" in result.faiss.context


def test_unknown_supplier_is_reported_by_every_real_source(
    database: Path, retriever: HybridRetriever
) -> None:
    result = query_all_sources(
        "SUP-999",
        sources={
            "sql": partial(query_sql, database_path=database),
            "faiss": partial(query_faiss, retriever=retriever),
            "graph": partial(query_graph, store=FakeGraphStore(None)),
        },
    )

    assert result.outcome == "failed"
    assert all(source.status == "error" for source in result.sources)
    assert all(
        "unknown supplier identity" in (source.error or "") for source in result.sources
    )
    assert result.entity == "SUP-999"


def test_real_sources_distinguish_no_evidence_from_failure(
    retriever: HybridRetriever, module2_dataset: SupplierDataset, tmp_path: Path
) -> None:
    # SQL: a real database with the supplier's observation removed.
    database = tmp_path / "no-metrics.sqlite"
    build_from_dataset(module2_dataset, database)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM delivery_metrics WHERE supplier_id = 'SUP-003'")

    # FAISS: the real wrapper over a retriever surface with no fused hits.
    supplier_documents = tuple(
        document for document in retriever.documents if document.supplier_id == "SUP-003"
    )
    no_hits = NoHitRetriever(supplier_documents)

    # GRAPH: the real wrapper over a store with no ownership rows.
    graph_store = FakeGraphStore(_identity(module2_dataset, "SUP-003"), ())

    result = query_all_sources(
        "SUP-003",
        sources={
            "sql": partial(query_sql, database_path=database),
            "faiss": partial(query_faiss, retriever=no_hits),
            "graph": partial(query_graph, store=graph_store),
        },
    )

    assert result.outcome == "complete"
    assert [source.status for source in result.sources] == ["empty", "empty", "empty"]
    assert all(source.error is None for source in result.sources)
    assert "delivery_metrics=0" in result.sql.context
    assert "No contract or complaint document matched" in result.faiss.context
    assert "ownership_relationships=0" in result.graph.context


def test_orchestration_preserves_the_requested_entity_and_canonical_resolution(
    database: Path, retriever: HybridRetriever, module2_dataset: SupplierDataset
) -> None:
    result = query_all_sources(
        "sup-007",
        sources={
            "sql": partial(query_sql, database_path=database),
            "faiss": partial(query_faiss, retriever=retriever),
            "graph": partial(
                query_graph,
                store=FakeGraphStore(
                    _identity(module2_dataset, "SUP-007"),
                    (_ownership_row(module2_dataset, "SUP-007"),),
                ),
            ),
        },
    )

    assert result.entity == "sup-007"
    assert "supplier=SUP-007" in result.sql.context
    assert "supplier=SUP-007" in result.faiss.context
    assert "supplier=SUP-007" in result.graph.context


@pytest.fixture(scope="module")
def live_graph_store(module2_dataset: SupplierDataset) -> Iterator[Neo4jGraphStore]:
    settings = load_neo4j_settings()
    with Neo4jGraphStore.from_settings(settings) as store:
        store.verify_connectivity()
        ingest_dataset(store, module2_dataset, dataset_path=DATASET_PATH)
        yield store


@pytest.mark.neo4j
@pytest.mark.skipif(
    os.environ.get("SUPPLIER_RISK_NEO4J_TESTS") != "1",
    reason="set SUPPLIER_RISK_NEO4J_TESTS=1 to run against the local Neo4j service",
)
def test_live_graph_source_joins_the_orchestration(
    live_graph_store: Neo4jGraphStore,
    database: Path,
    retriever: HybridRetriever,
    module2_dataset: SupplierDataset,
) -> None:
    result = query_all_sources(
        "SUP-007",
        sources={
            "sql": partial(query_sql, database_path=database),
            "faiss": partial(query_faiss, retriever=retriever),
            "graph": partial(query_graph, store=live_graph_store),
        },
    )

    assert result.outcome == "complete"
    assert [source.status for source in result.sources] == ["ok", "ok", "ok"]
    assert "on_time_delivery_rate=" in result.sql.context
    assert "document=" in result.faiss.context
    assert "ownership_relationships=" in result.graph.context
    assert "ORG-" in result.graph.context
