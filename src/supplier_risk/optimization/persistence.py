"""Persistence for the compiled Module 7 student.

The compiled program is saved with DSPy's own state API
(``dspy.Module.save``/``load``), which writes a JSON state file plus a
``metadata`` block recording dependency versions. Alongside it, this module
writes a separate, human-readable metadata document describing how the artifact
was produced.

What is stored
--------------
- ``optimized_student.json`` — the compiled DSPy program state (demonstrations and
  signature), written by :meth:`dspy.Module.save`.
- ``optimized_student.metadata.json`` — artifact version, DSPy version, teacher and
  student model identifiers, optimizer configuration, label-policy version,
  dataset schema version/seed/fingerprint, partition membership, compilation
  timestamp, and (optionally) the evaluation metrics.

What is never stored
--------------------
No API key, authorization header, or secret value is written. Only the *model
identifiers* are recorded, through
:meth:`~supplier_risk.decision.lm.RoleLanguageModels.masked`. The DSPy state file
excludes LM API keys by construction (``BaseLM.dump_state`` drops ``api_key``),
and this project never sets an LM on the predictor, so no LM state is embedded at
all.

Compatibility
-------------
The artifact is portable only within a compatible DSPy and model environment.
``dspy.Module.load`` warns on a dependency-version mismatch and the metadata
records the versions it was produced under; neither guarantees portability across
unrelated DSPy or provider versions. This is documented, not asserted away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel, ConfigDict, ValidationError

from supplier_risk.contracts import SupplierDataset
from supplier_risk.decision.lm import RoleLanguageModels
from supplier_risk.decision.module import VerdictGenerator
from supplier_risk.decision.verdict import SupplierRiskVerdict, parse_verdict_payload
from supplier_risk.optimization.errors import ArtifactError
from supplier_risk.optimization.examples import LabeledDecisionExample
from supplier_risk.optimization.labels import label_policy_description
from supplier_risk.optimization.optimizer import DEFAULT_OPTIMIZER_NAME, OptimizerSettings

ARTIFACT_VERSION = "module7.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "optimization"
PROGRAM_FILENAME = "optimized_student.json"
METADATA_FILENAME = "optimized_student.metadata.json"


class ArtifactMetadata(BaseModel):
    """The safe, secret-free description that accompanies a compiled student."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_version: str
    dspy_version: str
    optimizer: str
    optimizer_settings: dict[str, Any]
    label_policy: dict[str, Any]
    teacher_model: str
    student_model: str
    dataset_schema_version: str
    dataset_seed: int
    dataset_fingerprint: str
    partitions: dict[str, list[str]]
    program_filename: str
    compiled_at: str
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping (sorted by key when serialized)."""
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class SavedStudentArtifact:
    """The two files written by :func:`save_compiled_student`."""

    program_path: Path
    metadata_path: Path


def build_artifact_metadata(
    *,
    lms: RoleLanguageModels,
    settings: OptimizerSettings,
    dataset: SupplierDataset,
    dataset_fingerprint: str,
    partitions: dict[str, list[str]],
    metrics: dict[str, Any] | None = None,
    compiled_at: str | None = None,
    program_filename: str = PROGRAM_FILENAME,
) -> ArtifactMetadata:
    """Assemble artifact metadata without any secret value.

    ``compiled_at`` defaults to the current UTC time; it is metadata only and is
    deliberately excluded from any reproducibility claim.
    """
    timestamp = compiled_at if compiled_at is not None else datetime.now(UTC).isoformat()
    masked = lms.masked()
    return ArtifactMetadata(
        artifact_version=ARTIFACT_VERSION,
        dspy_version=str(getattr(dspy, "__version__", "unknown")),
        optimizer=DEFAULT_OPTIMIZER_NAME,
        optimizer_settings=settings.to_dict(),
        label_policy=label_policy_description(),
        teacher_model=masked["teacher_model"],
        student_model=masked["student_model"],
        dataset_schema_version=dataset.schema_version,
        dataset_seed=dataset.seed,
        dataset_fingerprint=dataset_fingerprint,
        partitions=partitions,
        program_filename=program_filename,
        compiled_at=timestamp,
        metrics=metrics,
    )


def save_compiled_student(
    program: dspy.Module,
    metadata: ArtifactMetadata,
    *,
    directory: Path = DEFAULT_ARTIFACT_DIR,
) -> SavedStudentArtifact:
    """Persist the compiled program and its metadata.

    Raises:
        ArtifactError: if the program or metadata cannot be written.
    """
    directory = Path(directory)
    program_path = directory / metadata.program_filename
    metadata_path = directory / METADATA_FILENAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        program.save(program_path)
        metadata_path.write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except ArtifactError:
        raise
    except Exception as error:  # noqa: BLE001 - saving must never leak a raw error
        raise ArtifactError(
            f"could not save the compiled student: {type(error).__name__}"
        ) from None
    return SavedStudentArtifact(program_path=program_path, metadata_path=metadata_path)


def read_artifact_metadata(
    metadata_path: Path = DEFAULT_ARTIFACT_DIR / METADATA_FILENAME,
) -> ArtifactMetadata:
    """Read and validate an artifact metadata document.

    Raises:
        ArtifactError: if the file is missing, unreadable, or invalid.
    """
    path = Path(metadata_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ArtifactError(f"artifact metadata not found: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(
            f"artifact metadata is unreadable ({type(error).__name__}): {path}"
        ) from None
    try:
        return ArtifactMetadata.model_validate(payload)
    except ValidationError as error:
        fields = ", ".join(
            ".".join(str(part) for part in detail.get("loc", ())) or "<metadata>"
            for detail in error.errors(include_input=False, include_url=False)
        )
        raise ArtifactError(f"artifact metadata is invalid at: {fields}") from None


def load_compiled_student(
    program_path: Path = DEFAULT_ARTIFACT_DIR / PROGRAM_FILENAME,
    *,
    program: dspy.Module | None = None,
) -> dspy.Module:
    """Load a compiled student without recompiling it.

    The program instance is created fresh (or supplied) and its state is restored
    from disk. The optimizer is never invoked.

    Raises:
        ArtifactError: if the file is missing, corrupt, or a version-incompatible
            state cannot be restored.
    """
    path = Path(program_path)
    if not path.is_file():
        raise ArtifactError(f"compiled student artifact not found: {path}")
    target: dspy.Module = VerdictGenerator() if program is None else program
    try:
        target.load(path)
    except Exception as error:  # noqa: BLE001 - a corrupt artifact must fail clearly
        raise ArtifactError(
            f"could not load the compiled student ({type(error).__name__}): {path}"
        ) from None
    return target


def load_and_verify(
    example: LabeledDecisionExample,
    *,
    lm: object,
    program_path: Path = DEFAULT_ARTIFACT_DIR / PROGRAM_FILENAME,
    program: dspy.Module | None = None,
) -> SupplierRiskVerdict:
    """Load a compiled student, run one representative input, and validate the output.

    The loaded program is called under ``dspy.context(lm=lm)`` so the intended
    student model is used. Loading never triggers compilation.

    Raises:
        ArtifactError: if the artifact cannot be loaded.
        DecisionResponseError: if the loaded program's output violates the verdict
            contract.
    """
    loaded = load_compiled_student(program_path, program=program)
    retrieval = example.retrieval
    with dspy.context(lm=lm):
        prediction = loaded(
            sql_context=retrieval.sql.context,
            faiss_context=retrieval.faiss.context,
            graph_context=retrieval.graph.context,
        )
    return parse_verdict_payload(prediction)
