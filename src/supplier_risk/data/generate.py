"""Deterministic generator for the synthetic supplier dataset.

The dataset is entirely fictional and is reproduced from a fixed seed plus a
fixed supplier manifest. Generate it with::

    uv run --locked python -m supplier_risk.data.generate

The default output, ``data/generated/suppliers.json``, is Git-ignored. See
``docs/schema.md`` for field definitions, units, and partition rules.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from supplier_risk.contracts import (
    PARTITION_SIZES,
    SEED_MAX,
    Complaint,
    ComplaintCategory,
    ComplaintSeverity,
    Contract,
    ContractType,
    DeliveryMetrics,
    OwnershipLink,
    OwnershipType,
    Sector,
    SupplierDataset,
    SupplierProfile,
    SupplierRecord,
    parse_dataset_json,
    partition_for_supplier_id,
)

DEFAULT_SEED = 20261019
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "generated" / "suppliers.json"


@dataclass(frozen=True, slots=True)
class _SupplierSpec:
    """Hand-written, stable identity and categorical facts for one supplier."""

    supplier_id: str
    legal_name: str
    country_code: str
    sector: Sector
    founded_year: int
    parent_company_id: str


# The manifest is explicitly ordered and carries stable identifiers. Changing
# an entry is a dataset-version change and must be documented.
_SUPPLIER_SPECS: tuple[_SupplierSpec, ...] = (
    _SupplierSpec("SUP-001", "Norvend Components Ltd", "GB", "electronics", 1954, "ORG-001"),
    _SupplierSpec("SUP-002", "Halcyon Circuitry GmbH", "DE", "electronics", 1987, "ORG-002"),
    _SupplierSpec("SUP-003", "Cedarline Packaging Co", "CA", "packaging", 1972, "ORG-003"),
    _SupplierSpec("SUP-004", "Tamarind Logistics Partners", "SG", "logistics", 2001, "ORG-004"),
    _SupplierSpec("SUP-005", "Ironvale Metals Pty Ltd", "AU", "raw_materials", 1968, "ORG-005"),
    _SupplierSpec(
        "SUP-006", "Blue Harbor Industrial Inc", "US", "industrial_components", 1994, "ORG-006"
    ),
    _SupplierSpec("SUP-007", "Sakurajima Precision KK", "JP", "electronics", 1959, "ORG-007"),
    _SupplierSpec("SUP-008", "Verde Polymer Solutions SpA", "IT", "raw_materials", 2005, "ORG-008"),
    _SupplierSpec("SUP-009", "Maple Ridge Packaging Inc", "CA", "packaging", 1983, "ORG-009"),
    _SupplierSpec("SUP-010", "Atlas Meridian Freight BV", "NL", "logistics", 2011, "ORG-010"),
    _SupplierSpec("SUP-011", "Quarrystone Minerals Ltd", "ZA", "raw_materials", 1976, "ORG-002"),
    _SupplierSpec("SUP-012", "Brightline Semiconductors Co", "TW", "electronics", 1999, "ORG-003"),
    _SupplierSpec(
        "SUP-013", "Orion Mechanical Works LLC", "US", "industrial_components", 1962, "ORG-006"
    ),
    _SupplierSpec("SUP-014", "Lakeshore Carton Supply", "US", "packaging", 1988, "ORG-001"),
    _SupplierSpec("SUP-015", "Fjordline Logistics AS", "NO", "logistics", 2008, "ORG-004"),
    _SupplierSpec("SUP-016", "Copperfield Alloys SpA", "CL", "raw_materials", 1970, "ORG-005"),
    _SupplierSpec(
        "SUP-017", "Pinecrest Components AB", "SE", "industrial_components", 1991, "ORG-007"
    ),
    _SupplierSpec("SUP-018", "Aurelia Circuits SA", "FR", "electronics", 2014, "ORG-008"),
    _SupplierSpec("SUP-019", "DeltaPort Shipping LLC", "AE", "logistics", 2003, "ORG-009"),
    _SupplierSpec("SUP-020", "Granite Peak Packaging Ltd", "NZ", "packaging", 1979, "ORG-010"),
    _SupplierSpec(
        "SUP-021", "Helix Industrial Supply Pvt Ltd", "IN", "industrial_components", 1996, "ORG-001"
    ),
    _SupplierSpec("SUP-022", "Solstice Materials SA", "BR", "raw_materials", 1965, "ORG-005"),
    _SupplierSpec("SUP-023", "Cobalt Ridge Electronics Co", "KR", "electronics", 1985, "ORG-003"),
    _SupplierSpec("SUP-024", "Westmoor Logistics Ltd", "IE", "logistics", 2016, "ORG-004"),
)

_PARENT_COMPANIES: dict[str, str] = {
    "ORG-001": "Meridian Industrial Holdings",
    "ORG-002": "Kestrel Capital Partners",
    "ORG-003": "Northstar Equity Group",
    "ORG-004": "Aurora Consortium",
    "ORG-005": "Pacific Crest Holdings",
    "ORG-006": "Redwood Manufacturing Group",
    "ORG-007": "Zenith Supply Group",
    "ORG-008": "Adriatic Partners",
    "ORG-009": "Summit Resource Holdings",
    "ORG-010": "Lakeside Commercial Trust",
}

_SECTOR_LABELS: dict[Sector, str] = {
    "electronics": "electronic components",
    "industrial_components": "industrial components",
    "logistics": "logistics services",
    "packaging": "packaging materials",
    "raw_materials": "processed raw materials",
}

_CONTRACT_TYPES: tuple[ContractType, ...] = (
    "MASTER_SUPPLY",
    "QUALITY_ASSURANCE",
    "LOGISTICS_SERVICES",
)
_CONTRACT_TYPE_LABELS: dict[ContractType, str] = {
    "MASTER_SUPPLY": "Master Supply Agreement",
    "QUALITY_ASSURANCE": "Quality Assurance Agreement",
    "LOGISTICS_SERVICES": "Logistics Services Agreement",
}
_CONTRACT_EFFECTIVE_DATES: tuple[date, ...] = (
    date(2022, 1, 1),
    date(2022, 7, 1),
    date(2023, 1, 1),
    date(2023, 7, 1),
    date(2024, 1, 1),
)
_CONTRACT_CLAUSES: tuple[str, ...] = (
    (
        "The supplier must maintain written quality procedures and retain inspection records "
        "for three years."
    ),
    (
        "Delivery schedules may be revised by mutual written agreement with at least fourteen "
        "days' notice."
    ),
    (
        "The supplier must notify the buyer within two business days of any material change "
        "in ownership or production location."
    ),
    (
        "Confidential information must be protected and used only for the purpose of performing "
        "this agreement."
    ),
    (
        "Either party may terminate for material breach after thirty days' written notice and "
        "an opportunity to cure."
    ),
    (
        "The supplier must comply with applicable labour, environmental, and restricted-substance "
        "requirements."
    ),
)

_COMPLAINT_SEVERITIES: tuple[ComplaintSeverity, ...] = ("LOW", "MEDIUM", "HIGH")
_SEVERITY_WEIGHTS: tuple[float, ...] = (0.35, 0.45, 0.20)
_COMPLAINT_CATEGORIES: tuple[ComplaintCategory, ...] = (
    "DELIVERY",
    "QUALITY",
    "DOCUMENTATION",
    "COMPLIANCE",
)
_COMPLAINT_TEMPLATES: dict[ComplaintCategory, tuple[str, ...]] = {
    "DELIVERY": (
        (
            "A scheduled shipment from {name} arrived {days} days after the agreed delivery "
            "date, which delayed downstream production."
        ),
        (
            "Tracking data showed that {name} held lot {lot} at a regional hub for {days} extra "
            "days without notifying the buyer."
        ),
        (
            "A partial shipment from {name} arrived with {days} units missing from the confirmed "
            "packing list."
        ),
    ),
    "QUALITY": (
        (
            "Incoming inspection of lot {lot} from {name} found dimensional measurements "
            "outside the approved tolerance band."
        ),
        (
            "A sample from lot {lot} supplied by {name} failed the agreed moisture test after "
            "{days} days in storage."
        ),
        (
            "Visual inspection of lot {lot} from {name} found surface defects on several units, "
            "requiring rework before use."
        ),
    ),
    "DOCUMENTATION": (
        (
            "The certificate of conformity for lot {lot} from {name} was missing when the "
            "shipment arrived and was supplied {days} days later."
        ),
        (
            "The safety data sheet provided by {name} for lot {lot} listed an outdated revision "
            "and could not be accepted."
        ),
        (
            "Customs paperwork for lot {lot} from {name} contained an inconsistent country of "
            "origin and delayed clearance by {days} days."
        ),
    ),
    "COMPLIANCE": (
        (
            "A declaration from {name} concerning restricted substances for lot {lot} was "
            "{days} days overdue."
        ),
        (
            "An audit of {name} found that lot {lot} had been stored outside the agreed "
            "temperature range."
        ),
        (
            "A labour-practice questionnaire from {name} was returned incomplete after a "
            "{days}-day reminder period."
        ),
    ),
}


def _supplier_number(supplier_id: str) -> int:
    return int(supplier_id.rsplit("-", maxsplit=1)[1])


def _build_contracts(rng: random.Random, spec: _SupplierSpec) -> tuple[Contract, ...]:
    contract_count = rng.choice((1, 1, 2))
    contracts: list[Contract] = []
    for sequence in range(1, contract_count + 1):
        contract_type = rng.choice(_CONTRACT_TYPES)
        effective_date = rng.choice(_CONTRACT_EFFECTIVE_DATES)
        term_years = rng.choice((1, 2, 3))
        expiry_date = effective_date.replace(year=effective_date.year + term_years)
        clauses = rng.sample(_CONTRACT_CLAUSES, k=2)
        contract_text = (
            f"{_CONTRACT_TYPE_LABELS[contract_type]} between the buyer and {spec.legal_name}, "
            f"a supplier of {_SECTOR_LABELS[spec.sector]} registered in {spec.country_code}. "
            f"The agreement takes effect on {effective_date.isoformat()} and expires on "
            f"{expiry_date.isoformat()}. " + " ".join(clauses)
        )
        contracts.append(
            Contract(
                contract_id=f"CON-{_supplier_number(spec.supplier_id):03d}-{sequence:02d}",
                supplier_id=spec.supplier_id,
                contract_type=contract_type,
                effective_date=effective_date,
                expiry_date=expiry_date,
                annual_value_usd=rng.randrange(250, 40_000) * 1000,
                contract_text=contract_text,
            )
        )
    return tuple(contracts)


def _build_complaints(rng: random.Random, spec: _SupplierSpec) -> tuple[Complaint, ...]:
    count = rng.choices((0, 1, 2, 3), weights=(15, 40, 30, 15), k=1)[0]
    complaints: list[Complaint] = []
    for sequence in range(1, count + 1):
        category = rng.choice(_COMPLAINT_CATEGORIES)
        severity = rng.choices(_COMPLAINT_SEVERITIES, weights=_SEVERITY_WEIGHTS, k=1)[0]
        template = rng.choice(_COMPLAINT_TEMPLATES[category])
        complaint_text = template.format(
            name=spec.legal_name,
            days=rng.randrange(1, 22),
            lot=f"LOT-{rng.randrange(1000, 10000)}",
        )
        complaints.append(
            Complaint(
                complaint_id=f"CMP-{_supplier_number(spec.supplier_id):03d}-{sequence:02d}",
                supplier_id=spec.supplier_id,
                received_date=date(2024, 1, 1) + timedelta(days=rng.randrange(0, 730)),
                severity=severity,
                category=category,
                complaint_text=complaint_text,
            )
        )
    return tuple(complaints)


def _build_ownership(rng: random.Random, spec: _SupplierSpec) -> OwnershipLink:
    parent_name = _PARENT_COMPANIES[spec.parent_company_id]
    relationship: OwnershipType
    percentage: float
    if rng.random() < 0.25:
        relationship = "JOINT_VENTURE_WITH"
        percentage = rng.choice((25.0, 40.0, 50.0))
    else:
        relationship = "SUBSIDIARY_OF"
        percentage = rng.choice((55.0, 60.0, 70.0, 80.0, 100.0))
    return OwnershipLink(
        parent_company_id=spec.parent_company_id,
        parent_company_name=parent_name,
        child_supplier_id=spec.supplier_id,
        ownership_percentage=percentage,
        relationship_type=relationship,
    )


def _build_record(rng: random.Random, spec: _SupplierSpec) -> SupplierRecord:
    profile = SupplierProfile(
        supplier_id=spec.supplier_id,
        legal_name=spec.legal_name,
        country_code=spec.country_code,
        sector=spec.sector,
        founded_year=spec.founded_year,
        employee_count=rng.randrange(80, 12_000),
    )
    delivery_metrics = DeliveryMetrics(
        supplier_id=spec.supplier_id,
        reporting_period="FY2025",
        on_time_delivery_rate=round(rng.uniform(0.72, 0.995), 3),
        defect_rate=round(rng.uniform(0.0005, 0.04), 4),
        average_lead_time_days=round(rng.uniform(5.0, 75.0), 1),
        orders_fulfilled=rng.randrange(120, 6000),
    )
    return SupplierRecord(
        profile=profile,
        partition=partition_for_supplier_id(spec.supplier_id),
        delivery_metrics=delivery_metrics,
        contracts=_build_contracts(rng, spec),
        complaints=_build_complaints(rng, spec),
        ownership=_build_ownership(rng, spec),
    )


def build_dataset(seed: int = DEFAULT_SEED) -> SupplierDataset:
    """Build and validate the complete synthetic dataset for ``seed``."""
    rng = random.Random(seed)
    records = tuple(_build_record(rng, spec) for spec in _SUPPLIER_SPECS)
    return SupplierDataset(seed=seed, suppliers=records)


def serialize_dataset(dataset: SupplierDataset) -> str:
    """Serialize a dataset to stable, human-readable JSON."""
    payload = dataset.model_dump(mode="json")
    return json.dumps(payload, indent=2) + "\n"


def write_dataset(dataset: SupplierDataset, path: str | Path = DEFAULT_OUTPUT) -> Path:
    """Write a validated dataset to ``path`` and return the output path.

    The destination is written directly, in place; this helper is not atomic.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize_dataset(dataset), encoding="utf-8")
    return output


def load_dataset(path: str | Path = DEFAULT_OUTPUT) -> SupplierDataset:
    """Load and validate a dataset JSON file."""
    return parse_dataset_json(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator CLI."""
    parser = argparse.ArgumentParser(description="Generate the synthetic supplier dataset.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Integer seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    if not 0 <= args.seed <= SEED_MAX:
        parser.error(f"--seed must be between 0 and {SEED_MAX}")

    dataset = build_dataset(seed=args.seed)
    output = write_dataset(dataset, args.output)
    counts = {partition: 0 for partition in PARTITION_SIZES}
    for record in dataset.suppliers:
        counts[record.partition] += 1

    print(f"Wrote {len(dataset.suppliers)} synthetic suppliers to {output}")
    print(
        f"Partitions: train={counts['train']} dev={counts['dev']} "
        f"held_out={counts['held_out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
