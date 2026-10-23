"""Hybrid document retrieval interface for Module 3.

``HybridRetriever`` loads a persisted corpus, FAISS index, and BM25 index, and
answers text queries with Reciprocal Rank Fusion. Metadata filters are applied to
both ranked lists before fusion. When a filter is active, both engines search the
whole corpus, so a supplier's or partition's documents cannot be missed because
they fell outside a small unfiltered candidate window; without filters, each
engine returns ``top_k * oversample`` candidates.

``query_faiss(entity)`` is the assignment-facing supplier wrapper. It resolves a
supplier id or legal name and returns one labeled context string covering
contract and complaint documents only; it never implies that SQLite or Neo4j was
queried. Infrastructure failures are reported as a tagged error string, while
the generic :meth:`HybridRetriever.query` API raises typed project errors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from supplier_risk.contracts import Partition
from supplier_risk.retrieval.corpus import (
    CorpusDocument,
    CorpusPayload,
    load_corpus,
)
from supplier_risk.retrieval.dense import FaissDenseIndex
from supplier_risk.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from supplier_risk.retrieval.errors import (
    EmbeddingModelError,
    RetrievalArtifactError,
    RetrievalQueryError,
    UnknownSupplierError,
)
from supplier_risk.retrieval.lexical import LexicalIndex
from supplier_risk.retrieval.manifest import RetrievalManifest, load_manifest
from supplier_risk.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion
from supplier_risk.retrieval.storage import BUILD_COMMAND, DEFAULT_ARTIFACTS_DIR
from supplier_risk.retrieval.validation import (
    validate_partition_filter,
    validate_query_text,
    validate_supplier_filter,
    validate_top_k,
)

DEFAULT_TOP_K: Final = 5
DEFAULT_OVERSAMPLE: Final = 4


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A fused document result with the source ranks that produced it."""

    document: CorpusDocument
    fusion_score: float
    dense_rank: int | None
    lexical_rank: int | None


def _validate_rrf_k(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RetrievalQueryError("rrf k must be a positive integer")
    return value


class HybridRetriever:
    """Dense + lexical retrieval with RRF, over one persisted corpus."""

    def __init__(
        self,
        *,
        documents: Mapping[str, CorpusDocument],
        dense: FaissDenseIndex,
        lexical: LexicalIndex,
        embedder: Embedder,
        rrf_k: int = DEFAULT_RRF_K,
        oversample: int = DEFAULT_OVERSAMPLE,
    ) -> None:
        """Assemble a retriever from already-validated components.

        ``documents`` must be in canonical corpus order, the same order used by
        the dense and lexical indexes.
        """
        if not documents:
            raise RetrievalArtifactError("retrieval corpus is empty")
        _validate_rrf_k(rrf_k)
        if isinstance(oversample, bool) or not isinstance(oversample, int) or oversample < 1:
            raise RetrievalQueryError("oversample must be a positive integer")

        canonical_ids = tuple(documents)
        if dense.document_ids != canonical_ids or lexical.document_ids != canonical_ids:
            raise RetrievalArtifactError(
                "corpus, dense index, and lexical index document order do not match; "
                "rebuild the retrieval artifacts together."
            )

        self._documents = dict(documents)
        self._dense = dense
        self._lexical = lexical
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._oversample = oversample

    @property
    def documents(self) -> tuple[CorpusDocument, ...]:
        """Corpus documents in canonical order."""
        return tuple(self._documents.values())

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def embedding_model(self) -> str:
        """Model recorded by the persisted dense index.

        The active query embedder is validated against this model at load time.
        """
        return self._dense.embedding_model

    @property
    def rrf_k(self) -> int:
        return self._rrf_k

    @classmethod
    def load(
        cls,
        artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
        *,
        embedder: Embedder | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        oversample: int = DEFAULT_OVERSAMPLE,
    ) -> HybridRetriever:
        """Load persisted artifacts and validate them against each other.

        The embedding model is not loaded here; an omitted ``embedder`` is
        created from the manifest model name and only touches the model on the
        first query.
        """
        directory = Path(artifacts_dir)
        manifest = load_manifest(directory)
        corpus = load_corpus(directory)
        _validate_manifest_against_corpus(manifest, corpus)

        dense = FaissDenseIndex.load_from(directory)
        lexical = LexicalIndex.load_from(directory)
        _validate_dense_against_manifest(manifest, dense, corpus)
        _validate_lexical_against_manifest(manifest, corpus, lexical)

        active_embedder: Embedder = (
            embedder
            if embedder is not None
            else SentenceTransformerEmbedder(manifest.embedding_model)
        )
        _validate_embedder_compatibility(manifest, active_embedder)
        return cls(
            documents={document.document_id: document for document in corpus.documents},
            dense=dense,
            lexical=lexical,
            embedder=active_embedder,
            rrf_k=rrf_k,
            oversample=oversample,
        )

    def _matching_documents(
        self, *, supplier_id: str | None, partition: Partition | None
    ) -> set[str]:
        return {
            document.document_id
            for document in self._documents.values()
            if (supplier_id is None or document.supplier_id == supplier_id)
            and (partition is None or document.partition == partition)
        }

    def _candidate_limit(self, top_k: int, *, filtered: bool) -> int:
        if filtered:
            return self._dense.size
        return min(max(top_k * self._oversample, top_k), self._dense.size)

    def query(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        supplier_id: str | None = None,
        partition: Partition | None = None,
    ) -> tuple[RetrievalHit, ...]:
        """Return fused hybrid results, optionally restricted by metadata filters.

        Raises:
            RetrievalQueryError: for a blank query, invalid ``top_k``, or an
                invalid metadata filter.
            RetrievalError: for embedding or artifact-integrity failures.
        """
        text = validate_query_text(query)
        limit = validate_top_k(top_k)
        supplier = validate_supplier_filter(supplier_id)
        partition_filter = validate_partition_filter(partition)

        allowed = self._matching_documents(supplier_id=supplier, partition=partition_filter)
        if not allowed:
            return ()

        filtered = supplier is not None or partition_filter is not None
        candidate_limit = self._candidate_limit(limit, filtered=filtered)

        query_vector = self._embedder.embed_query(text)
        dense_ids = [
            hit.document_id
            for hit in self._dense.search(query_vector, top_k=candidate_limit)
            if hit.document_id in allowed
        ]
        lexical_ids = [
            hit.document_id
            for hit in self._lexical.search(text, top_k=candidate_limit)
            if hit.document_id in allowed
        ]

        fused = reciprocal_rank_fusion(
            {"dense": dense_ids, "lexical": lexical_ids},
            k=self._rrf_k,
            top_k=limit,
        )
        return tuple(
            RetrievalHit(
                document=self._documents[hit.document_id],
                fusion_score=hit.score,
                dense_rank=hit.ranks.get("dense"),
                lexical_rank=hit.ranks.get("lexical"),
            )
            for hit in fused
        )


def resolve_supplier_id(entity: str, documents: Sequence[CorpusDocument]) -> str:
    """Resolve a supplier id or exact legal name (case-insensitive) to a supplier id."""
    if not isinstance(entity, str):
        raise UnknownSupplierError("supplier identity must be a string")
    text = entity.strip()
    if not text:
        raise UnknownSupplierError("supplier identity must not be empty")

    known_ids = {document.supplier_id for document in documents}
    candidate = text.upper()
    if candidate.startswith("SUP-"):
        if candidate in known_ids:
            return candidate
        raise UnknownSupplierError(f"unknown supplier identity {text!r}")

    matches = {
        document.supplier_id
        for document in documents
        if document.supplier_name.casefold() == text.casefold()
    }
    if len(matches) == 1:
        return matches.pop()
    if not matches:
        raise UnknownSupplierError(
            f"unknown supplier identity {text!r}; pass a supplier id such as SUP-007 "
            "or an exact supplier legal name"
        )
    raise UnknownSupplierError(f"supplier identity {text!r} is ambiguous")


def _render_supplier_context(
    retriever: HybridRetriever, supplier_id: str, top_k: int
) -> str:
    supplier_documents = [
        document
        for document in retriever.documents
        if document.supplier_id == supplier_id
    ]
    supplier_name = supplier_documents[0].supplier_name
    partition = supplier_documents[0].partition
    hits = retriever.query(
        f"{supplier_name} {supplier_id}",
        top_k=top_k,
        supplier_id=supplier_id,
    )

    header = (
        f"[FAISS] supplier={supplier_id} name={supplier_name} partition={partition} "
        f"documents={len(supplier_documents)} scope=contracts+complaints "
        f"ranking=hybrid(rrf_k={retriever.rrf_k})"
    )
    lines = [header]
    if not hits:
        lines.append("No contract or complaint document matched this supplier query.")
    for rank, hit in enumerate(hits, start=1):
        document = hit.document
        dense_rank = "-" if hit.dense_rank is None else str(hit.dense_rank)
        lexical_rank = "-" if hit.lexical_rank is None else str(hit.lexical_rank)
        lines.append(
            f"{rank}. document={document.document_id} "
            f"source={document.source_type}:{document.source_id} "
            f"rrf={hit.fusion_score:.6f} dense_rank={dense_rank} "
            f"lexical_rank={lexical_rank}"
        )
        lines.append(f"   {document.text}")
    return "\n".join(lines)


def _supplier_context(entity: str, retriever: HybridRetriever, top_k: int) -> str:
    try:
        supplier_id = resolve_supplier_id(entity, retriever.documents)
        return _render_supplier_context(retriever, supplier_id, top_k)
    except UnknownSupplierError as error:
        return f"[FAISS ERROR] {error}"
    except (RetrievalArtifactError, EmbeddingModelError) as error:
        return f"[FAISS ERROR] document retrieval unavailable: {error}"


@lru_cache(maxsize=4)
def _load_default_retriever(artifacts_dir: Path) -> HybridRetriever:
    return HybridRetriever.load(artifacts_dir)


def get_default_retriever(
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> HybridRetriever:
    """Return the process-wide retriever for ``artifacts_dir``, loaded once.

    The result is cached per resolved artifact path. After rebuilding artifacts
    in a long-running process, call :func:`reset_default_retriever` so the next
    call loads the new set, or restart the process.
    """
    return _load_default_retriever(Path(artifacts_dir).resolve())


def reset_default_retriever() -> None:
    """Clear the cached default retriever after rebuilding artifacts in-process."""
    _load_default_retriever.cache_clear()


def query_faiss(
    entity: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    retriever: HybridRetriever | None = None,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> str:
    """Return one labeled document-context string for a supplier entity.

    The result covers contract and complaint documents only. Unknown suppliers
    and unavailable or corrupt artifacts are reported as ``[FAISS ERROR]``
    strings rather than exceptions, matching the engine-level contract expected
    by the assignment; other errors propagate.
    """
    if retriever is not None:
        return _supplier_context(entity, retriever, top_k)
    try:
        return _supplier_context(entity, get_default_retriever(artifacts_dir), top_k)
    except (RetrievalArtifactError, EmbeddingModelError) as error:
        return f"[FAISS ERROR] document retrieval unavailable: {error}"


def _validate_manifest_against_corpus(
    manifest: RetrievalManifest, corpus: CorpusPayload
) -> None:
    if manifest.corpus_fingerprint != corpus.corpus_fingerprint:
        raise RetrievalArtifactError(
            "retrieval manifest and corpus fingerprints do not match; "
            f"rebuild the artifacts with `{BUILD_COMMAND}`."
        )
    if manifest.document_count != corpus.document_count:
        raise RetrievalArtifactError(
            "retrieval manifest and corpus document counts do not match; rebuild the artifacts."
        )
    if manifest.corpus_schema_version != corpus.corpus_schema_version:
        raise RetrievalArtifactError(
            "retrieval manifest and corpus schema versions do not match; rebuild the artifacts."
        )
    contracts = sum(1 for document in corpus.documents if document.source_type == "contract")
    complaints = corpus.document_count - contracts
    if (contracts, complaints) != (manifest.contract_count, manifest.complaint_count):
        raise RetrievalArtifactError(
            "retrieval manifest source-record counts do not match the corpus; "
            "rebuild the artifacts."
        )


def _validate_embedder_compatibility(
    manifest: RetrievalManifest, embedder: Embedder
) -> None:
    """Reject a query embedder that does not match the persisted dense index.

    ``declared_dimension`` is an optional capability: embedders that can report
    their dimension without loading a model are checked at load time, while
    lazily loaded models keep the existing query-time dimension check.
    """
    if embedder.model_name != manifest.embedding_model:
        raise RetrievalArtifactError(
            f"query embedder model {embedder.model_name!r} does not match the persisted index "
            f"model {manifest.embedding_model!r}; load the matching model or rebuild the index."
        )

    declared_dimension = getattr(embedder, "declared_dimension", None)
    if declared_dimension is None:
        return
    if isinstance(declared_dimension, bool) or not isinstance(declared_dimension, int):
        raise RetrievalArtifactError(
            "query embedder declared an invalid dimension; expected a positive integer"
        )
    if declared_dimension != manifest.embedding_dimension:
        raise RetrievalArtifactError(
            f"query embedder dimension {declared_dimension} does not match the persisted index "
            f"dimension {manifest.embedding_dimension}; load the matching model or rebuild the index."
        )


def _validate_dense_against_manifest(
    manifest: RetrievalManifest, dense: FaissDenseIndex, corpus: CorpusPayload
) -> None:
    metadata = dense.metadata
    if metadata.corpus_fingerprint != manifest.corpus_fingerprint:
        raise RetrievalArtifactError(
            "dense index metadata was built from a different corpus; rebuild the artifacts."
        )
    if metadata.embedding_model != manifest.embedding_model:
        raise RetrievalArtifactError(
            f"dense index was built with embedding model {metadata.embedding_model!r} but the "
            f"manifest records {manifest.embedding_model!r}; rebuild the index."
        )
    if metadata.embedding_dimension != manifest.embedding_dimension:
        raise RetrievalArtifactError(
            f"dense index dimension {metadata.embedding_dimension} does not match the manifest "
            f"dimension {manifest.embedding_dimension}; rebuild the index."
        )
    if metadata.document_count != manifest.document_count:
        raise RetrievalArtifactError(
            "dense index document count does not match the manifest; rebuild the index."
        )
    corpus_ids = tuple(document.document_id for document in corpus.documents)
    if dense.document_ids != corpus_ids:
        raise RetrievalArtifactError(
            "dense index document order does not match the corpus; rebuild the artifacts."
        )
    for entry, document in zip(metadata.vectors, corpus.documents, strict=True):
        if (
            entry.document_id != document.document_id
            or entry.supplier_id != document.supplier_id
            or entry.partition != document.partition
            or entry.source_type != document.source_type
        ):
            raise RetrievalArtifactError(
                "dense index metadata identity fields do not match the corpus for "
                f"{document.document_id}; rebuild the artifacts."
            )


def _validate_lexical_against_manifest(
    manifest: RetrievalManifest, corpus: CorpusPayload, lexical: LexicalIndex
) -> None:
    if lexical.corpus_fingerprint != manifest.corpus_fingerprint:
        raise RetrievalArtifactError(
            "lexical index was built from a different corpus; rebuild the artifacts."
        )
    if lexical.tokenizer_version != manifest.tokenizer_version:
        raise RetrievalArtifactError(
            "lexical index tokenizer version does not match the manifest; rebuild the index."
        )
    if (lexical.bm25_k1, lexical.bm25_b) != (manifest.bm25_k1, manifest.bm25_b):
        raise RetrievalArtifactError(
            "lexical index BM25 parameters do not match the manifest; rebuild the index."
        )
    if lexical.document_ids != tuple(document.document_id for document in corpus.documents):
        raise RetrievalArtifactError(
            "lexical index document order does not match the corpus; rebuild the artifacts."
        )
    lexical.verify_tokens(corpus.documents)
