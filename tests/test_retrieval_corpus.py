from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from supplier_risk.contracts import SupplierDataset, partition_for_supplier_id
from supplier_risk.data.generate import build_dataset
from supplier_risk.retrieval.corpus import (
    CorpusDocument,
    build_corpus,
    build_corpus_payload,
    corpus_fingerprint,
    document_id_for,
    load_corpus,
    validate_corpus,
    write_corpus,
)
from supplier_risk.retrieval.errors import CorpusError, RetrievalArtifactError


def _fabricate_document(**overrides: Any) -> CorpusDocument:
    values: dict[str, Any] = {
        "document_id": "contract:CON-001-01",
        "source_type": "contract",
        "source_id": "CON-001-01",
        "supplier_id": "SUP-001",
        "supplier_name": "Norvend Components Ltd",
        "partition": "train",
        "text": "Contract CON-001-01 for supplier Norvend Components Ltd.",
        "contract_type": "MASTER_SUPPLY",
        "effective_date": date(2024, 1, 1),
        "expiry_date": date(2025, 1, 1),
        "annual_value_usd": 1_000_000,
    }
    values.update(overrides)
    return CorpusDocument(**values)


def test_corpus_covers_every_contract_and_complaint(module2_dataset: SupplierDataset) -> None:
    documents = build_corpus(module2_dataset)

    contracts = [document for document in documents if document.source_type == "contract"]
    complaints = [document for document in documents if document.source_type == "complaint"]
    expected_contracts = sum(len(record.contracts) for record in module2_dataset.suppliers)
    expected_complaints = sum(len(record.complaints) for record in module2_dataset.suppliers)

    assert len(contracts) == expected_contracts == 32
    assert len(complaints) == expected_complaints == 34
    assert len(documents) == 66


def test_document_ids_are_stable_and_unique(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    rebuilt = build_corpus(build_dataset())

    assert rebuilt == corpus_documents
    assert corpus_fingerprint(rebuilt) == corpus_fingerprint(corpus_documents)
    assert len({document.document_id for document in corpus_documents}) == 66
    assert len({(document.source_type, document.source_id) for document in corpus_documents}) == 66
    for document in corpus_documents:
        assert document.document_id == document_id_for(document.source_type, document.source_id)
        assert document.document_id.startswith(f"{document.source_type}:")


def test_documents_map_to_their_supplier_and_partition(
    module2_dataset: SupplierDataset, corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    records = {record.profile.supplier_id: record for record in module2_dataset.suppliers}

    for document in corpus_documents:
        record = records[document.supplier_id]
        assert document.supplier_name == record.profile.legal_name
        assert document.partition == record.partition
        assert document.partition == partition_for_supplier_id(document.supplier_id)
        if document.source_type == "contract":
            assert any(contract.contract_id == document.source_id for contract in record.contracts)
        else:
            assert any(
                complaint.complaint_id == document.source_id for complaint in record.complaints
            )


def test_contract_documents_keep_contract_metadata(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    contracts = [document for document in corpus_documents if document.source_type == "contract"]

    assert contracts
    for document in contracts:
        assert document.contract_type in {
            "MASTER_SUPPLY",
            "QUALITY_ASSURANCE",
            "LOGISTICS_SERVICES",
        }
        assert document.effective_date is not None
        assert document.expiry_date is not None
        assert document.expiry_date > document.effective_date
        assert document.annual_value_usd is not None
        assert document.received_date is None
        assert document.severity is None
        assert document.category is None


def test_complaint_documents_keep_complaint_metadata(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    complaints = [document for document in corpus_documents if document.source_type == "complaint"]

    assert complaints
    for document in complaints:
        assert document.severity in {"LOW", "MEDIUM", "HIGH"}
        assert document.category in {"DELIVERY", "QUALITY", "DOCUMENTATION", "COMPLIANCE"}
        assert document.received_date is not None
        assert document.contract_type is None
        assert document.effective_date is None
        assert document.expiry_date is None
        assert document.annual_value_usd is None


def test_document_text_contains_identifiers_names_and_source_text(
    module2_dataset: SupplierDataset, corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    source_texts: dict[tuple[str, str], str] = {}
    for record in module2_dataset.suppliers:
        for contract in record.contracts:
            source_texts[("contract", contract.contract_id)] = contract.contract_text
        for complaint in record.complaints:
            source_texts[("complaint", complaint.complaint_id)] = complaint.complaint_text

    for document in corpus_documents:
        raw_text = source_texts[(document.source_type, document.source_id)]
        assert document.source_id in document.text
        assert document.supplier_id in document.text
        assert document.supplier_name in document.text
        assert raw_text in document.text
        assert "{'" not in document.text
        assert "None" not in document.text


def test_duplicate_documents_are_rejected(
    corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    with pytest.raises(CorpusError, match="duplicate document_id"):
        validate_corpus([corpus_documents[0], corpus_documents[0]])


def test_unknown_source_record_is_rejected(module2_dataset: SupplierDataset) -> None:
    fabricated = _fabricate_document(
        document_id="contract:CON-999-01",
        source_id="CON-999-01",
    )

    with pytest.raises(CorpusError, match="unknown source record"):
        validate_corpus([fabricated], dataset=module2_dataset)


def test_source_record_must_belong_to_the_document_supplier(
    module2_dataset: SupplierDataset,
) -> None:
    fabricated = _fabricate_document(
        document_id="contract:CON-002-01",
        source_id="CON-002-01",
    )

    with pytest.raises(CorpusError, match="belongs to SUP-002, not SUP-001"):
        validate_corpus([fabricated], dataset=module2_dataset)


def test_supplier_outside_the_manifest_is_rejected() -> None:
    with pytest.raises(ValidationError, match="manifest numbering range"):
        _fabricate_document(supplier_id="SUP-099")


def test_invalid_partition_for_supplier_is_rejected() -> None:
    with pytest.raises(ValidationError, match="partition must be"):
        _fabricate_document(partition="dev")


def test_document_id_must_match_the_source_record() -> None:
    with pytest.raises(ValidationError, match="document_id must equal"):
        _fabricate_document(document_id="contract:CON-001-02")


def test_source_id_format_must_match_the_source_type() -> None:
    with pytest.raises(ValidationError, match="not a valid contract identifier"):
        _fabricate_document(source_id="CMP-001-01", document_id="contract:CMP-001-01")


def test_contract_fields_are_rejected_on_complaint_documents() -> None:
    with pytest.raises(ValidationError, match="must not carry contract metadata"):
        _fabricate_document(
            document_id="complaint:CMP-001-01",
            source_type="complaint",
            source_id="CMP-001-01",
            text="Complaint CMP-001-01 for supplier Norvend Components Ltd.",
            received_date=date(2024, 5, 1),
            severity="MEDIUM",
            category="QUALITY",
        )


def test_blank_document_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="text"):
        _fabricate_document(text="   ")


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(CorpusError, match="corpus is empty"):
        validate_corpus([])


def test_corpus_payload_round_trips_through_disk(
    tmp_path: Path, module2_dataset: SupplierDataset, corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    payload = build_corpus_payload(corpus_documents, generator_seed=module2_dataset.seed)
    write_corpus(payload, tmp_path)

    loaded = load_corpus(tmp_path)

    assert loaded == payload
    assert loaded.documents == corpus_documents
    assert loaded.generator_seed == module2_dataset.seed
    assert loaded.corpus_schema_version == "module2.v1"


def test_missing_corpus_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(RetrievalArtifactError, match="missing retrieval corpus"):
        load_corpus(tmp_path)


def test_tampered_corpus_is_rejected(
    tmp_path: Path, module2_dataset: SupplierDataset, corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    payload = build_corpus_payload(corpus_documents, generator_seed=module2_dataset.seed)
    path = write_corpus(payload, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["documents"][0]["text"] = data["documents"][0]["text"] + " tampered"

    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="corpus_fingerprint"):
        load_corpus(tmp_path)


def test_duplicate_document_ids_in_a_reloaded_corpus_are_rejected(
    tmp_path: Path, module2_dataset: SupplierDataset, corpus_documents: tuple[CorpusDocument, ...]
) -> None:
    payload = build_corpus_payload(corpus_documents, generator_seed=module2_dataset.seed)
    path = write_corpus(payload, tmp_path)
    duplicates = list(corpus_documents)
    duplicates[1] = duplicates[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["documents"][1] = data["documents"][0]
    data["document_count"] = len(duplicates)
    # Recompute the fingerprint so duplicate-id validation is what rejects the file.
    data["corpus_fingerprint"] = corpus_fingerprint(duplicates)

    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="must be unique"):
        load_corpus(tmp_path)


def test_corrupt_corpus_json_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "corpus.json").write_text('{"documents": [}', encoding="utf-8")

    with pytest.raises(RetrievalArtifactError, match="retrieval corpus"):
        load_corpus(tmp_path)
