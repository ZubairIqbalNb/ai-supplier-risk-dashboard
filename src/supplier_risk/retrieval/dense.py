"""Dense retrieval: persisted FAISS exact cosine-similarity index.

Documents are embedded with a local sentence-transformer model, L2-normalized,
and added to a FAISS ``IndexFlatIP`` index, so inner product equals cosine
similarity. Vector ids are the index positions ``0..n-1``; ``dense.metadata.json``
maps each vector id to its document, supplier, partition, and source type.

Building the index is an explicit operation. Loading never rebuilds or replaces
an index, and querying without persisted artifacts raises a clear error.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import faiss
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from supplier_risk.contracts import Partition
from supplier_risk.retrieval.corpus import CorpusDocument, DocumentSourceType
from supplier_risk.retrieval.embeddings import Embedder
from supplier_risk.retrieval.errors import (
    CorpusError,
    EmbeddingModelError,
    RetrievalArtifactError,
    RetrievalQueryError,
)
from supplier_risk.retrieval.storage import (
    ARTIFACT_VERSION,
    BUILD_COMMAND,
    DENSE_INDEX_FILENAME,
    DENSE_METADATA_FILENAME,
    ArtifactVersion,
    describe_validation_error,
    read_artifact_text,
    write_json,
)
from supplier_risk.retrieval.validation import validate_top_k

METRIC: Final = "cosine_inner_product"
INDEX_TYPE: Final = "IndexFlatIP"


@dataclass(frozen=True, slots=True)
class DenseHit:
    """One dense search result, identified by its FAISS vector id."""

    vector_id: int
    document_id: str
    score: float


class VectorEntry(BaseModel):
    """Mapping of one FAISS vector id to the document it represents."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    vector_id: int = Field(ge=0)
    document_id: str = Field(min_length=1)
    supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    partition: Partition
    source_type: DocumentSourceType


class DenseIndexMetadata(BaseModel):
    """Persisted metadata that reconstructs dense search results without embeddings."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_version: ArtifactVersion
    corpus_fingerprint: str = Field(min_length=64, max_length=64)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    metric: Literal["cosine_inner_product"]
    index_type: Literal["IndexFlatIP"]
    normalized: Literal[True]
    document_count: int = Field(ge=1)
    vectors: tuple[VectorEntry, ...] = Field(min_length=1)


def _validated_embeddings(
    matrix: object,
    *,
    expected_rows: int,
    expected_dimension: int,
) -> NDArray[np.float32]:
    """Validate an embedding matrix and return a writable float32 copy."""
    try:
        array = np.array(matrix, dtype=np.float32, copy=True, order="C")
    except (TypeError, ValueError):
        raise EmbeddingModelError("embedding output is not a numeric matrix") from None
    if array.ndim != 2:
        raise EmbeddingModelError("embedding output must be a two-dimensional matrix")
    if array.shape != (expected_rows, expected_dimension):
        raise EmbeddingModelError(
            f"embedding output has shape {array.shape}; expected "
            f"({expected_rows}, {expected_dimension})"
        )
    if not bool(np.isfinite(array).all()):
        raise EmbeddingModelError("embedding output contains non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if bool((norms == 0.0).any()):
        raise EmbeddingModelError("embedding output contains a zero-length vector")
    return array


class FaissDenseIndex:
    """Exact cosine-similarity FAISS index with document-level metadata."""

    def __init__(self, index: faiss.Index, metadata: DenseIndexMetadata) -> None:
        """Wrap an existing FAISS index; use :meth:`build` or :meth:`load_from` instead."""
        self._index = index
        self._metadata = metadata

    @property
    def metadata(self) -> DenseIndexMetadata:
        return self._metadata

    @property
    def embedding_model(self) -> str:
        return self._metadata.embedding_model

    @property
    def dimension(self) -> int:
        return self._metadata.embedding_dimension

    @property
    def size(self) -> int:
        return self._metadata.document_count

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(entry.document_id for entry in self._metadata.vectors)

    @classmethod
    def build(
        cls,
        documents: Sequence[CorpusDocument],
        embedder: Embedder,
        *,
        corpus_fingerprint: str,
    ) -> FaissDenseIndex:
        """Embed ``documents`` in order and build a flat inner-product index."""
        if not documents:
            raise CorpusError("cannot build a dense index from an empty corpus")

        texts = [document.text for document in documents]
        vectors = _validated_embeddings(
            embedder.embed_documents(texts),
            expected_rows=len(texts),
            expected_dimension=embedder.dimension,
        )
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(embedder.dimension)
        index.add(vectors)

        metadata = DenseIndexMetadata(
            artifact_version=ARTIFACT_VERSION,
            corpus_fingerprint=corpus_fingerprint,
            embedding_model=embedder.model_name,
            embedding_dimension=embedder.dimension,
            metric=METRIC,
            index_type=INDEX_TYPE,
            normalized=True,
            document_count=len(documents),
            vectors=tuple(
                VectorEntry(
                    vector_id=position,
                    document_id=document.document_id,
                    supplier_id=document.supplier_id,
                    partition=document.partition,
                    source_type=document.source_type,
                )
                for position, document in enumerate(documents)
            ),
        )
        return cls(index, metadata)

    def write_to(self, directory: Path) -> None:
        """Persist the FAISS index and its vector/document metadata."""
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / DENSE_INDEX_FILENAME
        temporary = index_path.with_name(f".{index_path.name}.tmp")
        try:
            faiss.write_index(self._index, str(temporary))
            os.replace(temporary, index_path)
        except (OSError, RuntimeError) as error:
            temporary.unlink(missing_ok=True)
            raise RetrievalArtifactError(
                f"could not write the dense index to {index_path}: {type(error).__name__}"
            ) from None
        write_json(directory / DENSE_METADATA_FILENAME, self._metadata.model_dump(mode="json"))

    @classmethod
    def load_from(cls, directory: Path) -> FaissDenseIndex:
        """Load and validate a persisted dense index; never rebuilds it."""
        metadata_path = directory / DENSE_METADATA_FILENAME
        index_path = directory / DENSE_INDEX_FILENAME

        text = read_artifact_text(metadata_path, description="dense index metadata")
        try:
            metadata = DenseIndexMetadata.model_validate_json(text)
        except ValidationError as error:
            detail = describe_validation_error(error)
            raise RetrievalArtifactError(
                f"dense index metadata at {metadata_path} is invalid ({detail})"
            ) from None
        _validate_metadata(metadata, metadata_path)

        if not index_path.is_file():
            raise RetrievalArtifactError(
                f"missing dense index: {index_path}. Build it with `{BUILD_COMMAND}`."
            )
        try:
            index = faiss.read_index(str(index_path))
        except (RuntimeError, OSError, ValueError):
            raise RetrievalArtifactError(
                f"dense index at {index_path} is unreadable or corrupt; "
                f"rebuild it with `{BUILD_COMMAND}`."
            ) from None

        _validate_index(index, metadata, index_path)
        return cls(index, metadata)

    def search(
        self, query_vector: NDArray[np.float32], *, top_k: int
    ) -> tuple[DenseHit, ...]:
        """Return the nearest documents for a query vector, best first."""
        limit = validate_top_k(top_k)
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise RetrievalQueryError("query embedding must be a one-dimensional vector")
        if vector.shape[0] != self.dimension:
            raise RetrievalQueryError(
                f"query embedding has {vector.shape[0]} dimensions; "
                f"the index expects {self.dimension}"
            )
        if not bool(np.isfinite(vector).all()):
            raise RetrievalQueryError("query embedding contains non-finite values")

        matrix = np.array(vector, dtype=np.float32, copy=True, order="C").reshape(1, -1)
        if float(np.linalg.norm(matrix)) == 0.0:
            raise RetrievalQueryError("query embedding has zero length")
        faiss.normalize_L2(matrix)

        scores, ids = self._index.search(matrix, limit)
        hits: list[DenseHit] = []
        for score, vector_id in zip(scores[0], ids[0], strict=True):
            identifier = int(vector_id)
            if identifier < 0:
                continue
            hits.append(
                DenseHit(
                    vector_id=identifier,
                    document_id=self._metadata.vectors[identifier].document_id,
                    score=float(score),
                )
            )
        return tuple(hits)


def _validate_metadata(metadata: DenseIndexMetadata, path: Path) -> None:
    if metadata.document_count != len(metadata.vectors):
        raise RetrievalArtifactError(
            f"dense index metadata at {path} declares {metadata.document_count} documents "
            f"but lists {len(metadata.vectors)} vectors; rebuild the index with `{BUILD_COMMAND}`."
        )
    for position, entry in enumerate(metadata.vectors):
        if entry.vector_id != position:
            raise RetrievalArtifactError(
                f"dense index metadata at {path} has vector_id {entry.vector_id} at "
                f"position {position}; vector ids must match index positions."
            )
    document_ids = [entry.document_id for entry in metadata.vectors]
    if len(set(document_ids)) != len(document_ids):
        raise RetrievalArtifactError(
            f"dense index metadata at {path} maps several vectors to the same document."
        )


def _validate_index(index: faiss.Index, metadata: DenseIndexMetadata, path: Path) -> None:
    if not isinstance(index, faiss.IndexFlatIP):
        raise RetrievalArtifactError(
            f"dense index at {path} is not an inner-product index; "
            f"rebuild it with `{BUILD_COMMAND}`."
        )
    if index.d != metadata.embedding_dimension:
        raise RetrievalArtifactError(
            f"dense index at {path} has dimension {index.d} but its metadata declares "
            f"{metadata.embedding_dimension}; rebuild the index."
        )
    if index.ntotal != metadata.document_count:
        raise RetrievalArtifactError(
            f"dense index at {path} contains {index.ntotal} vectors but its metadata declares "
            f"{metadata.document_count} documents; rebuild the index."
        )
