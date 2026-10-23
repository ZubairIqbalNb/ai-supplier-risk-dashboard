"""Module 3 document retrieval: deterministic corpus, FAISS, BM25, and RRF.

Importing this package loads NumPy and FAISS but never downloads or loads an
embedding model; :class:`SentenceTransformerEmbedder` loads its model lazily on
first use.
"""

from supplier_risk.retrieval.corpus import (
    CorpusDocument,
    CorpusPayload,
    build_corpus,
    build_corpus_payload,
    corpus_fingerprint,
    document_id_for,
    load_corpus,
    validate_corpus,
    write_corpus,
)
from supplier_risk.retrieval.dense import DenseHit, FaissDenseIndex
from supplier_risk.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    SentenceTransformerEmbedder,
)
from supplier_risk.retrieval.errors import (
    CorpusError,
    EmbeddingModelError,
    RetrievalArtifactError,
    RetrievalError,
    RetrievalQueryError,
    UnknownSupplierError,
)
from supplier_risk.retrieval.lexical import LexicalHit, LexicalIndex, tokenize
from supplier_risk.retrieval.pipeline import (
    DEFAULT_TOP_K,
    HybridRetriever,
    RetrievalHit,
    query_faiss,
    reset_default_retriever,
    resolve_supplier_id,
)
from supplier_risk.retrieval.rrf import (
    DEFAULT_RRF_K,
    FusionHit,
    reciprocal_rank_fusion,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RRF_K",
    "DEFAULT_TOP_K",
    "CorpusDocument",
    "CorpusError",
    "CorpusPayload",
    "DenseHit",
    "Embedder",
    "EmbeddingModelError",
    "FaissDenseIndex",
    "FusionHit",
    "HybridRetriever",
    "LexicalHit",
    "LexicalIndex",
    "RetrievalArtifactError",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalQueryError",
    "SentenceTransformerEmbedder",
    "UnknownSupplierError",
    "build_corpus",
    "build_corpus_payload",
    "corpus_fingerprint",
    "document_id_for",
    "load_corpus",
    "query_faiss",
    "reciprocal_rank_fusion",
    "reset_default_retriever",
    "resolve_supplier_id",
    "tokenize",
    "validate_corpus",
    "write_corpus",
]
