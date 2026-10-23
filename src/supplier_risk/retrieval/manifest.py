"""Persisted manifest describing one consistent set of retrieval artifacts.

``manifest.json`` is written last during a build and records the corpus
fingerprint, document counts, embedding model and dimension, index type, and
tokenizer/BM25 configuration. Loading validates the corpus, dense, and lexical
artifacts against it, so a stale or partially replaced artifact set is rejected
instead of silently used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from supplier_risk.contracts import SchemaVersion
from supplier_risk.retrieval.errors import RetrievalArtifactError
from supplier_risk.retrieval.storage import (
    MANIFEST_FILENAME,
    ArtifactVersion,
    describe_validation_error,
    read_artifact_text,
    write_json,
)


class RetrievalManifest(BaseModel):
    """Identity and compatibility record for the persisted retrieval artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_version: ArtifactVersion
    corpus_schema_version: SchemaVersion
    corpus_fingerprint: str = Field(min_length=64, max_length=64)
    document_count: int = Field(ge=1)
    contract_count: int = Field(ge=0)
    complaint_count: int = Field(ge=0)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    index_type: Literal["IndexFlatIP"]
    metric: Literal["cosine_inner_product"]
    tokenizer_version: Literal["module3.v1"]
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _source_counts_add_up(self) -> Self:
        if self.contract_count + self.complaint_count != self.document_count:
            raise ValueError("contract_count + complaint_count must equal document_count")
        return self


def write_manifest(manifest: RetrievalManifest, directory: Path) -> Path:
    """Persist the manifest, which marks an artifact set as complete."""
    path = directory / MANIFEST_FILENAME
    write_json(path, manifest.model_dump(mode="json"))
    return path


def load_manifest(directory: Path) -> RetrievalManifest:
    """Load and validate the manifest for an artifact directory."""
    path = directory / MANIFEST_FILENAME
    text = read_artifact_text(path, description="retrieval manifest")
    try:
        return RetrievalManifest.model_validate_json(text)
    except ValidationError as error:
        detail = describe_validation_error(error)
        raise RetrievalArtifactError(
            f"retrieval manifest at {path} is invalid ({detail})"
        ) from None
