"""Deterministic document corpus built from the validated Module 2 dataset.

One searchable document is created per contract or complaint record, because the
generated texts are short (a few hundred characters) and a whole agreement or
complaint is a useful retrieval unit. Documents are ordered by ``supplier_id``,
then contracts, then complaints, then source-record id, so repeated builds from
unchanged input produce an identical corpus and fingerprint.

Document identity is derived from the source record, not from a process-random
hash: ``document_id`` is ``"<source_type>:<source_id>"``, for example
``contract:CON-001-01`` or ``complaint:CMP-001-01``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from supplier_risk.contracts import (
    SCHEMA_VERSION,
    Complaint,
    ComplaintCategory,
    ComplaintSeverity,
    Contract,
    ContractType,
    Partition,
    SchemaVersion,
    SupplierDataset,
    SupplierRecord,
    partition_for_supplier_id,
)
from supplier_risk.retrieval.errors import CorpusError, RetrievalArtifactError
from supplier_risk.retrieval.storage import (
    ARTIFACT_VERSION,
    CORPUS_FILENAME,
    ArtifactVersion,
    describe_validation_error,
    read_artifact_text,
    write_json,
)

type DocumentSourceType = Literal["contract", "complaint"]
DocumentSourceTypes: Final[tuple[DocumentSourceType, ...]] = ("contract", "complaint")

_CONTRACT_SOURCE_ID = re.compile(r"^CON-\d{3}-\d{2}$")
_COMPLAINT_SOURCE_ID = re.compile(r"^CMP-\d{3}-\d{2}$")


def document_id_for(source_type: DocumentSourceType, source_id: str) -> str:
    """Return the stable document identifier for one source record."""
    return f"{source_type}:{source_id}"


class CorpusDocument(BaseModel):
    """One searchable document mapped to exactly one source record and supplier.

    Contract-only fields are populated for contract documents and are ``None``
    for complaints, and the reverse holds for complaint-only fields. The model
    validates that identity, the partition, and the source-type fields agree, so
    a reloaded artifact cannot silently carry a mismatched supplier or partition.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    document_id: str = Field(min_length=1)
    source_type: DocumentSourceType
    source_id: str = Field(min_length=1)
    supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    supplier_name: str = Field(min_length=2, max_length=120)
    partition: Partition
    text: str = Field(min_length=1)

    contract_type: ContractType | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    annual_value_usd: int | None = None

    received_date: date | None = None
    severity: ComplaintSeverity | None = None
    category: ComplaintCategory | None = None

    @field_validator("text")
    @classmethod
    def _text_has_no_surrounding_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        if value != value.strip():
            raise ValueError("text must not have leading or trailing whitespace")
        return value

    @model_validator(mode="after")
    def _identity_and_source_fields_agree(self) -> Self:
        if self.document_id != document_id_for(self.source_type, self.source_id):
            raise ValueError("document_id must equal '<source_type>:<source_id>'")

        source_pattern = (
            _CONTRACT_SOURCE_ID if self.source_type == "contract" else _COMPLAINT_SOURCE_ID
        )
        if source_pattern.fullmatch(self.source_id) is None:
            raise ValueError(f"source_id is not a valid {self.source_type} identifier")

        contract_fields = (
            self.contract_type,
            self.effective_date,
            self.expiry_date,
            self.annual_value_usd,
        )
        complaint_fields = (self.received_date, self.severity, self.category)
        if self.source_type == "contract":
            if any(value is None for value in contract_fields):
                raise ValueError("contract documents require contract metadata")
            if any(value is not None for value in complaint_fields):
                raise ValueError("contract documents must not carry complaint metadata")
        else:
            if any(value is None for value in complaint_fields):
                raise ValueError("complaint documents require complaint metadata")
            if any(value is not None for value in contract_fields):
                raise ValueError("complaint documents must not carry contract metadata")

        try:
            expected_partition = partition_for_supplier_id(self.supplier_id)
        except ValueError:
            raise ValueError("supplier_id is outside the fixed manifest numbering range") from None
        if self.partition != expected_partition:
            raise ValueError(
                f"partition must be {expected_partition} for supplier_id {self.supplier_id}"
            )
        return self


def _contract_document(record: SupplierRecord, contract: Contract) -> CorpusDocument:
    profile = record.profile
    header = (
        f"Contract {contract.contract_id} for supplier {profile.legal_name} "
        f"({profile.supplier_id}), contract type {contract.contract_type}, "
        f"annual value USD {contract.annual_value_usd}."
    )
    return CorpusDocument(
        document_id=document_id_for("contract", contract.contract_id),
        source_type="contract",
        source_id=contract.contract_id,
        supplier_id=profile.supplier_id,
        supplier_name=profile.legal_name,
        partition=record.partition,
        text=f"{header} {contract.contract_text}",
        contract_type=contract.contract_type,
        effective_date=contract.effective_date,
        expiry_date=contract.expiry_date,
        annual_value_usd=contract.annual_value_usd,
    )


def _complaint_document(record: SupplierRecord, complaint: Complaint) -> CorpusDocument:
    profile = record.profile
    header = (
        f"Complaint {complaint.complaint_id} for supplier {profile.legal_name} "
        f"({profile.supplier_id}), category {complaint.category}, "
        f"severity {complaint.severity}, received {complaint.received_date.isoformat()}."
    )
    return CorpusDocument(
        document_id=document_id_for("complaint", complaint.complaint_id),
        source_type="complaint",
        source_id=complaint.complaint_id,
        supplier_id=profile.supplier_id,
        supplier_name=profile.legal_name,
        partition=record.partition,
        text=f"{header} {complaint.complaint_text}",
        received_date=complaint.received_date,
        severity=complaint.severity,
        category=complaint.category,
    )


def build_corpus(dataset: SupplierDataset) -> tuple[CorpusDocument, ...]:
    """Build the deterministic document corpus for a validated dataset.

    Raises:
        CorpusError: if documents are duplicated or reference unknown records.
    """
    documents: list[CorpusDocument] = []
    for record in sorted(dataset.suppliers, key=lambda item: item.profile.supplier_id):
        contracts = sorted(record.contracts, key=lambda item: item.contract_id)
        complaints = sorted(record.complaints, key=lambda item: item.complaint_id)
        documents.extend(_contract_document(record, contract) for contract in contracts)
        documents.extend(_complaint_document(record, complaint) for complaint in complaints)

    validate_corpus(documents, dataset=dataset)
    return tuple(documents)


def validate_corpus(
    documents: Sequence[CorpusDocument],
    *,
    dataset: SupplierDataset | None = None,
) -> None:
    """Validate document uniqueness and the mapping back to source records.

    When ``dataset`` is supplied, every document must name a source record that
    exists in that dataset and belongs to the document's supplier.
    """
    if not documents:
        raise CorpusError("document corpus is empty; at least one contract or complaint is required")

    issues: list[str] = []
    document_ids: set[str] = set()
    source_records: set[tuple[str, str]] = set()

    expected_suppliers: dict[tuple[str, str], str] = {}
    known_suppliers: set[str] = set()
    if dataset is not None:
        for record in dataset.suppliers:
            supplier_id = record.profile.supplier_id
            known_suppliers.add(supplier_id)
            for contract in record.contracts:
                expected_suppliers[("contract", contract.contract_id)] = supplier_id
            for complaint in record.complaints:
                expected_suppliers[("complaint", complaint.complaint_id)] = supplier_id

    for document in documents:
        if not document.text.strip():
            issues.append(f"empty text for document {document.document_id}")

        if document.document_id in document_ids:
            issues.append(f"duplicate document_id: {document.document_id}")
        document_ids.add(document.document_id)

        source_key = (document.source_type, document.source_id)
        if source_key in source_records:
            issues.append(
                f"duplicate source record: {document.source_type} {document.source_id}"
            )
        source_records.add(source_key)

        if dataset is None:
            continue
        if document.supplier_id not in known_suppliers:
            issues.append(
                f"unknown supplier_id {document.supplier_id} for {document.document_id}"
            )
            continue
        owning_supplier = expected_suppliers.get(source_key)
        if owning_supplier is None:
            issues.append(
                f"unknown source record {document.source_type} {document.source_id} "
                f"for {document.document_id}"
            )
        elif owning_supplier != document.supplier_id:
            issues.append(
                f"source record {document.source_type} {document.source_id} belongs to "
                f"{owning_supplier}, not {document.supplier_id}"
            )

    if issues:
        raise CorpusError("; ".join(issues))


def corpus_fingerprint(documents: Sequence[CorpusDocument]) -> str:
    """Return a stable SHA-256 fingerprint of the canonical document payload."""
    canonical = json.dumps(
        [document.model_dump(mode="json") for document in documents],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CorpusPayload(BaseModel):
    """Persisted corpus artifact: documents plus the identity of the source dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_version: ArtifactVersion
    corpus_schema_version: SchemaVersion
    generator_seed: int = Field(ge=0)
    document_count: int = Field(ge=1)
    corpus_fingerprint: str = Field(min_length=64, max_length=64)
    documents: tuple[CorpusDocument, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _counts_fingerprint_and_identity_match(self) -> Self:
        if self.document_count != len(self.documents):
            raise ValueError("document_count does not match the number of documents")
        if self.corpus_fingerprint != corpus_fingerprint(self.documents):
            raise ValueError("corpus_fingerprint does not match the persisted documents")
        document_ids = [document.document_id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document_id values must be unique")
        return self


def build_corpus_payload(
    documents: Sequence[CorpusDocument], *, generator_seed: int
) -> CorpusPayload:
    """Wrap validated documents in a persistable payload."""
    return CorpusPayload(
        artifact_version=ARTIFACT_VERSION,
        corpus_schema_version=SCHEMA_VERSION,
        generator_seed=generator_seed,
        document_count=len(documents),
        corpus_fingerprint=corpus_fingerprint(documents),
        documents=tuple(documents),
    )


def write_corpus(payload: CorpusPayload, directory: Path) -> Path:
    """Persist the corpus payload to ``directory`` and return the artifact path."""
    path = directory / CORPUS_FILENAME
    write_json(path, payload.model_dump(mode="json"))
    return path


def load_corpus(directory: Path) -> CorpusPayload:
    """Reload and fully validate a persisted corpus payload."""
    path = directory / CORPUS_FILENAME
    text = read_artifact_text(path, description="retrieval corpus")
    try:
        return CorpusPayload.model_validate_json(text)
    except ValidationError as error:
        detail = describe_validation_error(error)
        raise RetrievalArtifactError(
            f"retrieval corpus at {path} is invalid ({detail})"
        ) from None
