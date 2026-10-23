from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from supplier_risk.retrieval.corpus import CorpusDocument, corpus_fingerprint
from supplier_risk.retrieval.errors import (
    CorpusError,
    RetrievalArtifactError,
    RetrievalQueryError,
)
from supplier_risk.retrieval.lexical import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    TOKENIZER_VERSION,
    LexicalIndex,
    tokenize,
)


def _build(documents: tuple[CorpusDocument, ...]) -> LexicalIndex:
    return LexicalIndex.build(documents, corpus_fingerprint=corpus_fingerprint(documents))


def test_tokenizer_preserves_identifiers_and_numbers() -> None:
    tokens = tokenize("Complaint CMP-001-01 for SUP-001, lot LOT-8399, FY2025, 12,500 USD.")

    assert tokens[:6] == ("complaint", "cmp-001-01", "for", "sup-001", "lot", "lot-8399")
    assert "fy2025" in tokens
    assert "12" in tokens
    assert "500" in tokens
    assert tokens == tokenize("Complaint CMP-001-01 for SUP-001, lot LOT-8399, FY2025, 12,500 USD.")


def test_tokenizer_handles_iso_dates_as_single_tokens() -> None:
    assert tokenize("effective 2023-01-01 expires") == ("effective", "2023-01-01", "expires")


def test_exact_identifier_search_finds_the_contract(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    index = _build(corpus_documents)

    hits = index.search("CON-001-02", top_k=3)

    assert hits
    assert hits[0].document_id == "contract:CON-001-02"


def test_exact_lot_search_finds_the_complaint(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    index = _build(corpus_documents)
    lots = sorted({lot for document in corpus_documents for lot in re.findall(r"LOT-\d+", document.text)})
    unique_lot = next(
        lot
        for lot in lots
        if sum(1 for document in corpus_documents if lot in document.text) == 1
    )
    complaint = next(document for document in corpus_documents if unique_lot in document.text)

    hits = index.search(unique_lot, top_k=3)

    assert hits
    assert hits[0].document_id == complaint.document_id


def test_query_without_matching_terms_returns_no_results(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    index = _build(corpus_documents)

    assert index.search("zzzznonexistentterm", top_k=5) == ()


def test_scores_are_descending_and_results_deterministic(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    index = _build(corpus_documents)

    hits = index.search("quality inspection records", top_k=10)

    assert hits
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert [hit.document_id for hit in hits] == list(dict.fromkeys(hit.document_id for hit in hits))
    assert index.search("quality inspection records", top_k=10) == hits


def test_document_ids_align_with_the_corpus(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    index = _build(corpus_documents)

    assert index.document_ids == tuple(document.document_id for document in corpus_documents)
    assert index.size == len(corpus_documents)
    assert index.tokenizer_version == TOKENIZER_VERSION
    assert index.bm25_k1 == DEFAULT_BM25_K1
    assert index.bm25_b == DEFAULT_BM25_B


def test_persisted_lexical_index_reloads_with_identical_scores(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    index = _build(corpus_documents)
    index.write_to(tmp_path)

    loaded = LexicalIndex.load_from(tmp_path)

    assert loaded.document_ids == index.document_ids
    assert loaded.corpus_fingerprint == index.corpus_fingerprint
    assert loaded.tokenizer_version == index.tokenizer_version
    assert loaded.search("quality inspection", top_k=5) == index.search(
        "quality inspection", top_k=5
    )


def test_lexical_artifact_is_readable_json(tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...]) -> None:
    _build(corpus_documents).write_to(tmp_path)

    payload = json.loads((tmp_path / "lexical.json").read_text(encoding="utf-8"))

    assert payload["tokenizer_version"] == TOKENIZER_VERSION
    assert payload["document_ids"][0] == "contract:CON-001-01"
    assert payload["tokens"][0][0] == "contract"


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(CorpusError, match="empty corpus"):
        LexicalIndex.build((), corpus_fingerprint=corpus_fingerprint(()))


@pytest.mark.parametrize("query", ["", "   ", "!!!"])
def test_blank_or_tokenless_queries_are_rejected(
    corpus_documents: tuple[CorpusDocument, ...], query: str
) -> None:
    index = _build(corpus_documents)

    with pytest.raises(RetrievalQueryError):
        index.search(query, top_k=3)


@pytest.mark.parametrize("top_k", [0, -1, True, 2.5])
def test_invalid_top_k_is_rejected(
    corpus_documents: tuple[CorpusDocument, ...], top_k: object
) -> None:
    index = _build(corpus_documents)

    with pytest.raises(RetrievalQueryError, match="top_k"):
        index.search("quality", top_k=top_k)  # type: ignore[arg-type]


def test_missing_lexical_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(RetrievalArtifactError, match="missing lexical index"):
        LexicalIndex.load_from(tmp_path)


def test_corrupt_lexical_artifact_raises(tmp_path: Path) -> None:
    (tmp_path / "lexical.json").write_text('{"document_ids": []}', encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="lexical index"):
        LexicalIndex.load_from(tmp_path)


def test_misaligned_lexical_artifact_raises(
    tmp_path: Path, corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    _build(corpus_documents).write_to(tmp_path)
    path = tmp_path / "lexical.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tokens"] = payload["tokens"][:-1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="same length"):
        LexicalIndex.load_from(tmp_path)
