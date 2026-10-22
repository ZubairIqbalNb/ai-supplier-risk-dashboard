"""Canonical supplier data contracts for Module 2.

The models here describe *raw* supplier source data only. Risk scores,
verdicts, evidence status, and other derived values belong to later modules
and are intentionally absent. See ``docs/schema.md`` for the documented
schema, units, and provisional design decisions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from typing import Any, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

Partition = Literal["train", "dev", "held_out"]
SchemaVersion = Literal["module2.v1"]
Sector = Literal[
    "electronics",
    "industrial_components",
    "logistics",
    "packaging",
    "raw_materials",
]
ComplaintSeverity = Literal["LOW", "MEDIUM", "HIGH"]
ComplaintCategory = Literal["DELIVERY", "QUALITY", "DOCUMENTATION", "COMPLIANCE"]
ContractType = Literal["MASTER_SUPPLY", "QUALITY_ASSURANCE", "LOGISTICS_SERVICES"]
OwnershipType = Literal["SUBSIDIARY_OF", "JOINT_VENTURE_WITH"]

SCHEMA_VERSION: Final = "module2.v1"
SEED_MAX = 2**32 - 1

# Fixed manifest numbering. This mapping is the single source of truth for
# supplier-to-partition membership; the validator and the generator both use it.
PARTITION_ID_RANGES: dict[Partition, range] = {
    "train": range(1, 15),
    "dev": range(15, 19),
    "held_out": range(19, 25),
}

# Provisional dataset design. The assignment only requires approximately
# 20-30 fake entities and a held-out entity outside the training manifest.
PARTITION_SIZES: dict[Partition, int] = {
    partition: len(numbers) for partition, numbers in PARTITION_ID_RANGES.items()
}

# Only Pydantic field-level errors are truncated at this bound. Dataset-level
# integrity issues are aggregated into a single message and are not truncated.
_MAX_REPORTED_FIELD_ERRORS = 20


class DatasetValidationError(ValueError):
    """Raised when a dataset payload fails canonical validation."""


def partition_for_supplier_id(supplier_id: str) -> Partition:
    """Return the partition fixed by the supplier manifest numbering.

    Raises:
        ValueError: if ``supplier_id`` is not a manifest identifier.
    """
    match = re.fullmatch(r"SUP-(\d{3})", supplier_id)
    if match is not None:
        number = int(match.group(1))
        for partition, numbers in PARTITION_ID_RANGES.items():
            if number in numbers:
                return partition
    raise ValueError(f"supplier_id {supplier_id!r} is outside the manifest numbering range")


class _ContractModel(BaseModel):
    """Shared strict, immutable configuration for canonical records."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class SupplierProfile(_ContractModel):
    """Stable identity and categorical facts about a synthetic supplier."""

    supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    legal_name: str = Field(min_length=2, max_length=120)
    country_code: str = Field(pattern=r"^[A-Z]{2}$")
    sector: Sector
    founded_year: int = Field(ge=1800, le=2025)
    employee_count: int = Field(ge=1, le=1_000_000)

    @field_validator("legal_name")
    @classmethod
    def _legal_name_has_no_surrounding_whitespace(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("legal_name must not have leading or trailing whitespace")
        return value


class DeliveryMetrics(_ContractModel):
    """One annual structured observation per supplier (SQLite source data)."""

    supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    reporting_period: str = Field(pattern=r"^FY\d{4}$")
    on_time_delivery_rate: float = Field(ge=0.0, le=1.0)
    defect_rate: float = Field(ge=0.0, le=1.0)
    average_lead_time_days: float = Field(gt=0.0, le=365.0)
    orders_fulfilled: int = Field(ge=0, le=10_000_000)


class Contract(_ContractModel):
    """A synthetic contract whose text is a future semantic-search source."""

    contract_id: str = Field(pattern=r"^CON-\d{3}-\d{2}$")
    supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    contract_type: ContractType
    effective_date: date
    expiry_date: date
    annual_value_usd: int = Field(ge=0, le=1_000_000_000_000)
    contract_text: str = Field(min_length=40, max_length=4000)

    @model_validator(mode="after")
    def _expiry_after_effective(self) -> Self:
        if self.expiry_date <= self.effective_date:
            raise ValueError("expiry_date must be later than effective_date")
        return self


class Complaint(_ContractModel):
    """A synthetic complaint whose text is a future semantic-search source."""

    complaint_id: str = Field(pattern=r"^CMP-\d{3}-\d{2}$")
    supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    received_date: date
    severity: ComplaintSeverity
    category: ComplaintCategory
    complaint_text: str = Field(min_length=20, max_length=4000)


class OwnershipLink(_ContractModel):
    """A directed ownership edge for the future Neo4j graph."""

    parent_company_id: str = Field(pattern=r"^ORG-\d{3}$")
    parent_company_name: str = Field(min_length=2, max_length=120)
    child_supplier_id: str = Field(pattern=r"^SUP-\d{3}$")
    ownership_percentage: float = Field(gt=0.0, le=100.0)
    relationship_type: OwnershipType

    @model_validator(mode="after")
    def _percentage_matches_relationship(self) -> Self:
        if self.relationship_type == "SUBSIDIARY_OF" and self.ownership_percentage <= 50.0:
            raise ValueError("SUBSIDIARY_OF requires ownership_percentage greater than 50")
        if self.relationship_type == "JOINT_VENTURE_WITH" and self.ownership_percentage > 50.0:
            raise ValueError("JOINT_VENTURE_WITH requires ownership_percentage at most 50")
        return self


class SupplierRecord(_ContractModel):
    """All raw records that belong to one supplier identity."""

    profile: SupplierProfile
    partition: Partition
    delivery_metrics: DeliveryMetrics
    contracts: tuple[Contract, ...] = Field(min_length=1, max_length=10)
    complaints: tuple[Complaint, ...] = Field(default=(), max_length=20)
    ownership: OwnershipLink | None = None

    @model_validator(mode="after")
    def _nested_records_match_supplier(self) -> Self:
        supplier_id = self.profile.supplier_id
        related_ids = [self.delivery_metrics.supplier_id]
        related_ids.extend(contract.supplier_id for contract in self.contracts)
        related_ids.extend(complaint.supplier_id for complaint in self.complaints)
        if self.ownership is not None:
            related_ids.append(self.ownership.child_supplier_id)

        if any(related_id != supplier_id for related_id in related_ids):
            raise ValueError(f"nested records must all reference supplier_id {supplier_id}")
        if len({contract.contract_id for contract in self.contracts}) != len(self.contracts):
            raise ValueError("contract_id values must be unique within a supplier record")
        if len({complaint.complaint_id for complaint in self.complaints}) != len(self.complaints):
            raise ValueError("complaint_id values must be unique within a supplier record")
        return self


class SupplierDataset(_ContractModel):
    """The complete synthetic dataset, including partition assignment."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    seed: int = Field(ge=0, le=SEED_MAX)
    suppliers: tuple[SupplierRecord, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _dataset_integrity(self) -> Self:
        issues: list[str] = []
        counts: dict[Partition, int] = {partition: 0 for partition in PARTITION_SIZES}
        supplier_ids: set[str] = set()
        legal_names: set[str] = set()
        contract_ids: set[str] = set()
        complaint_ids: set[str] = set()
        parent_names: dict[str, str] = {}

        for record in self.suppliers:
            profile = record.profile
            counts[record.partition] += 1

            try:
                expected_partition = partition_for_supplier_id(profile.supplier_id)
            except ValueError:
                issues.append(f"supplier_id is outside the fixed manifest: {profile.supplier_id}")
            else:
                if record.partition != expected_partition:
                    issues.append(
                        f"supplier_id {profile.supplier_id} must be in partition "
                        f"{expected_partition}; found {record.partition}"
                    )

            if profile.supplier_id in supplier_ids:
                issues.append(f"duplicate supplier_id: {profile.supplier_id}")
            supplier_ids.add(profile.supplier_id)

            name_key = profile.legal_name.casefold()
            if name_key in legal_names:
                issues.append(f"duplicate supplier legal_name for {profile.supplier_id}")
            legal_names.add(name_key)

            for contract in record.contracts:
                if contract.contract_id in contract_ids:
                    issues.append(f"duplicate contract_id: {contract.contract_id}")
                contract_ids.add(contract.contract_id)

            for complaint in record.complaints:
                if complaint.complaint_id in complaint_ids:
                    issues.append(f"duplicate complaint_id: {complaint.complaint_id}")
                complaint_ids.add(complaint.complaint_id)

            if record.ownership is not None:
                parent_id = record.ownership.parent_company_id
                parent_name = record.ownership.parent_company_name
                known_name = parent_names.get(parent_id)
                if known_name is None:
                    parent_names[parent_id] = parent_name
                elif known_name != parent_name:
                    issues.append(f"inconsistent parent_company_name for {parent_id}")

        for parent_id in parent_names:
            if parent_id in supplier_ids:
                issues.append(f"parent_company_id collides with a supplier_id: {parent_id}")

        if counts != PARTITION_SIZES:
            issues.append(f"partition counts must be {PARTITION_SIZES}; found {counts}")

        if issues:
            raise ValueError("; ".join(issues))
        return self


def format_validation_error(error: ValidationError) -> str:
    """Return a concise, input-free validation report.

    Rejected input values are omitted so callers cannot accidentally log an
    entire record or a sensitive value. Field-level entries are truncated at
    ``_MAX_REPORTED_FIELD_ERRORS``; an aggregated dataset-integrity error is a
    single entry and may therefore list more individual issues.
    """
    entries: list[str] = []
    for detail in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in detail["loc"]) or "<dataset>"
        entries.append(f"{location}: {detail['msg']}")

    shown = entries[:_MAX_REPORTED_FIELD_ERRORS]
    if len(entries) > _MAX_REPORTED_FIELD_ERRORS:
        shown.append(f"... and {len(entries) - _MAX_REPORTED_FIELD_ERRORS} more error(s)")
    return "Invalid supplier dataset:\n" + "\n".join(shown)


def parse_dataset_payload(payload: Mapping[str, Any]) -> SupplierDataset:
    """Validate a Python-typed payload and raise a safe error on failure."""
    try:
        return SupplierDataset.model_validate(payload)
    except ValidationError as error:
        raise DatasetValidationError(format_validation_error(error)) from None


def parse_dataset_json(payload: str) -> SupplierDataset:
    """Validate a JSON document and raise a safe error on failure."""
    try:
        return SupplierDataset.model_validate_json(payload)
    except ValidationError as error:
        raise DatasetValidationError(format_validation_error(error)) from None
