"""Build the persisted retrieval artifacts from the generated supplier dataset.

Run::

    uv run --locked python -m supplier_risk.retrieval.build

The build reads and re-validates ``data/generated/suppliers.json``, creates one
document per contract and complaint, embeds them locally, writes the FAISS index,
the BM25 lexical corpus, and the corpus metadata to ``artifacts/retrieval/``, and
writes ``manifest.json`` last. Building is explicit; loading an absent artifact
set raises instead of building on demand.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from supplier_risk.contracts import DatasetValidationError, SupplierDataset
from supplier_risk.data.generate import DEFAULT_OUTPUT as DEFAULT_DATASET_PATH
from supplier_risk.data.generate import load_dataset
from supplier_risk.retrieval.corpus import (
    build_corpus,
    build_corpus_payload,
    corpus_fingerprint,
    write_corpus,
)
from supplier_risk.retrieval.dense import INDEX_TYPE, FaissDenseIndex
from supplier_risk.retrieval.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    SentenceTransformerEmbedder,
)
from supplier_risk.retrieval.errors import CorpusError, RetrievalError
from supplier_risk.retrieval.lexical import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    TOKENIZER_VERSION,
    LexicalIndex,
)
from supplier_risk.retrieval.manifest import RetrievalManifest, write_manifest
from supplier_risk.retrieval.storage import ARTIFACT_VERSION, DEFAULT_ARTIFACTS_DIR

GENERATE_COMMAND: Final = "uv run --locked python -m supplier_risk.data.generate"


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """Counts and configuration of one completed artifact build."""

    artifacts_dir: Path
    dataset_path: Path
    document_count: int
    contract_count: int
    complaint_count: int
    embedding_model: str
    embedding_dimension: int
    corpus_fingerprint: str


def _load_validated_dataset(path: Path) -> SupplierDataset:
    try:
        return load_dataset(path)
    except FileNotFoundError:
        raise CorpusError(
            f"supplier dataset not found at {path}; run `{GENERATE_COMMAND}` first."
        ) from None
    except OSError:
        raise CorpusError(f"could not read the supplier dataset at {path}") from None
    except DatasetValidationError as error:
        raise CorpusError(str(error)) from None


def build_retrieval_artifacts(
    *,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    embedder: Embedder | None = None,
    bm25_k1: float = DEFAULT_BM25_K1,
    bm25_b: float = DEFAULT_BM25_B,
) -> BuildSummary:
    """Build and persist the corpus, FAISS index, BM25 corpus, and manifest."""
    directory = Path(artifacts_dir)
    source = Path(dataset_path)
    dataset = _load_validated_dataset(source)
    documents = build_corpus(dataset)
    fingerprint = corpus_fingerprint(documents)

    active_embedder: Embedder = (
        embedder if embedder is not None else SentenceTransformerEmbedder()
    )
    dense = FaissDenseIndex.build(
        documents, active_embedder, corpus_fingerprint=fingerprint
    )
    lexical = LexicalIndex.build(
        documents, corpus_fingerprint=fingerprint, k1=bm25_k1, b=bm25_b
    )
    payload = build_corpus_payload(documents, generator_seed=dataset.seed)
    contract_count = sum(1 for document in documents if document.source_type == "contract")
    manifest = RetrievalManifest(
        artifact_version=ARTIFACT_VERSION,
        corpus_schema_version=payload.corpus_schema_version,
        corpus_fingerprint=fingerprint,
        document_count=payload.document_count,
        contract_count=contract_count,
        complaint_count=payload.document_count - contract_count,
        embedding_model=dense.embedding_model,
        embedding_dimension=dense.dimension,
        index_type=INDEX_TYPE,
        metric="cosine_inner_product",
        tokenizer_version=TOKENIZER_VERSION,
        bm25_k1=lexical.bm25_k1,
        bm25_b=lexical.bm25_b,
    )

    directory.mkdir(parents=True, exist_ok=True)
    write_corpus(payload, directory)
    dense.write_to(directory)
    lexical.write_to(directory)
    write_manifest(manifest, directory)

    return BuildSummary(
        artifacts_dir=directory,
        dataset_path=source,
        document_count=manifest.document_count,
        contract_count=manifest.contract_count,
        complaint_count=manifest.complaint_count,
        embedding_model=manifest.embedding_model,
        embedding_dimension=manifest.embedding_dimension,
        corpus_fingerprint=fingerprint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the artifact build CLI."""
    parser = argparse.ArgumentParser(
        description="Build the Module 3 document retrieval artifacts."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Validated supplier dataset (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Artifact output directory (default: {DEFAULT_ARTIFACTS_DIR}).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Local sentence-transformer model (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    args = parser.parse_args(argv)

    try:
        summary = build_retrieval_artifacts(
            dataset_path=args.dataset,
            artifacts_dir=args.artifacts_dir,
            embedder=SentenceTransformerEmbedder(args.embedding_model),
        )
    except RetrievalError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Wrote {summary.document_count} documents "
        f"({summary.contract_count} contracts, {summary.complaint_count} complaints) "
        f"from {summary.dataset_path} to {summary.artifacts_dir}"
    )
    print(
        f"Dense: {INDEX_TYPE} dim={summary.embedding_dimension} "
        f"model={summary.embedding_model}"
    )
    print(f"Lexical: BM25Okapi documents={summary.document_count} tokenizer={TOKENIZER_VERSION}")
    print(f"Corpus fingerprint: {summary.corpus_fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
