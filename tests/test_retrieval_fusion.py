from __future__ import annotations

import pytest

from supplier_risk.retrieval.errors import RetrievalQueryError
from supplier_risk.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_rrf_formula_uses_one_based_ranks() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ["a", "b", "c"], "lexical": ["c", "a"]}, k=60
    )

    scores = {hit.document_id: hit.score for hit in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert [hit.document_id for hit in fused] == ["a", "c", "b"]


def test_rrf_records_the_original_ranks() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ["a", "b", "c"], "lexical": ["c", "a"]}, k=60
    )

    ranks = {hit.document_id: dict(hit.ranks) for hit in fused}
    assert ranks["a"] == {"dense": 1, "lexical": 2}
    assert ranks["b"] == {"dense": 2}
    assert ranks["c"] == {"dense": 3, "lexical": 1}


def test_document_present_in_both_lists_gets_summed() -> None:
    fused = reciprocal_rank_fusion({"dense": ["only"], "lexical": ["only"]}, k=60)

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_ties_break_on_document_id() -> None:
    fused = reciprocal_rank_fusion({"dense": ["b"], "lexical": ["a"]}, k=60)

    assert [hit.document_id for hit in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(fused[1].score)


def test_empty_inputs_return_no_results() -> None:
    assert reciprocal_rank_fusion({}) == ()
    assert reciprocal_rank_fusion({"dense": [], "lexical": []}) == ()
    assert reciprocal_rank_fusion({"dense": [], "lexical": ["a"]})[0].document_id == "a"


def test_top_k_limits_a_stable_order() -> None:
    ranked = {"dense": ["a", "b", "c"], "lexical": []}

    assert [hit.document_id for hit in reciprocal_rank_fusion(ranked, top_k=2)] == ["a", "b"]
    assert [hit.document_id for hit in reciprocal_rank_fusion(ranked, top_k=99)] == [
        "a",
        "b",
        "c",
    ]
    assert reciprocal_rank_fusion(ranked, top_k=1)[0].document_id == "a"


def test_default_k_is_the_documented_constant() -> None:
    assert DEFAULT_RRF_K == 60
    assert reciprocal_rank_fusion({"dense": ["a"]})[0].score == pytest.approx(1 / 61)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "60"])
def test_invalid_rrf_constant_is_rejected(k: object) -> None:
    with pytest.raises(RetrievalQueryError, match="rrf k"):
        reciprocal_rank_fusion({"dense": ["a"]}, k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, True, 2.5])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(RetrievalQueryError, match="top_k"):
        reciprocal_rank_fusion({"dense": ["a"]}, top_k=top_k)  # type: ignore[arg-type]


def test_duplicate_ids_within_one_list_are_rejected() -> None:
    with pytest.raises(RetrievalQueryError, match="duplicate document id"):
        reciprocal_rank_fusion({"dense": ["a", "a"]})


def test_blank_document_id_is_rejected() -> None:
    with pytest.raises(RetrievalQueryError, match="non-empty strings"):
        reciprocal_rank_fusion({"dense": [""]})
