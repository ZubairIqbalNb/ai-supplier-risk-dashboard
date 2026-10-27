"""Persistence tests for Module 7.

These tests save and reload a real compiled-shaped program (a
:class:`~supplier_risk.decision.module.VerdictGenerator` with demonstrations) using
DSPy's own state API, offline. They assert round-tripping, secret-free metadata,
that reloading never recompiles, and that corrupt or missing artifacts fail
clearly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from supplier_risk.data.generate import build_dataset
from supplier_risk.decision.lm import RoleLanguageModels
from supplier_risk.decision.module import VerdictGenerator
from supplier_risk.optimization.errors import ArtifactError
from supplier_risk.optimization.examples import (
    SIGNATURE_INPUTS,
    LabeledDecisionExample,
    dataset_fingerprint,
)
from supplier_risk.optimization.optimizer import OptimizerSettings
from supplier_risk.optimization.persistence import (
    ARTIFACT_VERSION,
    build_artifact_metadata,
    load_and_verify,
    load_compiled_student,
    read_artifact_metadata,
    save_compiled_student,
)
from tests.optimization_support import retrieval_for

SENTINEL = "sk-live-secret-should-never-be-written"
DATASET = build_dataset()


class _KeyedLM(SimpleNamespace):
    """LM stand-in exposing a model id and a secret, like a real provider client."""

    def __init__(self, model: str) -> None:
        super().__init__(model=model, api_key=SENTINEL)


def _lms() -> RoleLanguageModels:
    return RoleLanguageModels(teacher=_KeyedLM("gemini/teacher"), student=_KeyedLM("gemini/student"))


def _demo() -> dspy.Example:
    return dspy.Example(
        sql_context="[SQL] supplier=SUP-019 evidence present",
        faiss_context="[FAISS] supplier=SUP-019 documents present",
        graph_context="[GRAPH] supplier=SUP-019 ownership present",
        supplier_id="SUP-019",
        verdict="FLAG",
        rationale="The supplied evidence supports this verdict for the reviewed supplier.",
        findings=["[SQL] metrics reviewed"],
        missing_evidence=[],
    ).with_inputs(*SIGNATURE_INPUTS)


def _compiled_program() -> VerdictGenerator:
    program = VerdictGenerator()
    for _, predictor in program.named_predictors():
        predictor.demos = [_demo()]
    return program


def _metadata(directory: Path) -> object:
    return build_artifact_metadata(
        lms=_lms(),
        settings=OptimizerSettings(seed=3),
        dataset=DATASET,
        dataset_fingerprint=dataset_fingerprint(DATASET),
        partitions={
            "train": ["SUP-001"],
            "dev": ["SUP-015"],
            "held_out": ["SUP-019"],
        },
        metrics={"outcome": "unchanged"},
        compiled_at="2026-10-27T00:00:00+00:00",
    )


def test_save_and_read_metadata_round_trip(tmp_path: Path) -> None:
    program = _compiled_program()
    metadata = _metadata(tmp_path)

    saved = save_compiled_student(program, metadata, directory=tmp_path)

    assert saved.program_path.exists()
    assert saved.metadata_path.exists()
    loaded_metadata = read_artifact_metadata(saved.metadata_path)
    assert loaded_metadata == metadata
    assert loaded_metadata.artifact_version == ARTIFACT_VERSION
    assert loaded_metadata.dataset_fingerprint == dataset_fingerprint(DATASET)
    assert loaded_metadata.partitions["held_out"] == ["SUP-019"]


def test_artifact_files_contain_no_secret(tmp_path: Path) -> None:
    program = _compiled_program()
    metadata = _metadata(tmp_path)

    saved = save_compiled_student(program, metadata, directory=tmp_path)

    combined = saved.program_path.read_text() + saved.metadata_path.read_text()
    assert SENTINEL not in combined
    assert "api_key" not in combined.lower()
    # The demo text is present, so the artifact is not silently empty.
    assert "SUP-019" in saved.program_path.read_text()


def test_lm_state_saved_on_the_predictor_drops_the_api_key(tmp_path: Path) -> None:
    program = VerdictGenerator()
    live_lm = dspy.LM("gemini/gemini-2.0-flash", api_key=SENTINEL)
    for _, predictor in program.named_predictors():
        predictor.lm = live_lm

    saved = save_compiled_student(program, _metadata(tmp_path), directory=tmp_path)

    assert SENTINEL not in saved.program_path.read_text()


def test_reload_restores_demonstrations(tmp_path: Path) -> None:
    program = _compiled_program()
    saved = save_compiled_student(program, _metadata(tmp_path), directory=tmp_path)

    reloaded = load_compiled_student(saved.program_path)

    assert isinstance(reloaded, VerdictGenerator)
    demos = [predictor.demos for _, predictor in reloaded.named_predictors()]
    assert len(demos) == 1
    assert len(demos[0]) == 1
    assert demos[0][0]["supplier_id"] == "SUP-019"


def test_reload_does_not_recompile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved = save_compiled_student(_compiled_program(), _metadata(tmp_path), directory=tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("load must not invoke the optimizer")

    monkeypatch.setattr(dspy.BootstrapFewShot, "compile", fail)

    reloaded = load_compiled_student(saved.program_path)

    assert isinstance(reloaded, VerdictGenerator)


def test_missing_artifact_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        load_compiled_student(tmp_path / "missing.json")


def test_corrupt_artifact_fails_clearly(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ this is not valid json", encoding="utf-8")

    with pytest.raises(ArtifactError, match="could not load"):
        load_compiled_student(corrupt)


def test_invalid_metadata_fails_clearly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.metadata.json"
    bad.write_text(json.dumps({"artifact_version": "module7.v1"}), encoding="utf-8")

    with pytest.raises(ArtifactError, match="invalid"):
        read_artifact_metadata(bad)


def test_missing_metadata_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        read_artifact_metadata(tmp_path / "nope.json")


def test_loaded_program_runs_a_representative_input(tmp_path: Path) -> None:
    saved = save_compiled_student(_compiled_program(), _metadata(tmp_path), directory=tmp_path)
    example = LabeledDecisionExample(
        supplier_id="SUP-019",
        partition="held_out",
        expected_verdict="FLAG",
        retrieval=retrieval_for("SUP-019"),
    )
    lm = DummyLM(
        {
            "SUP-019": {
                "reasoning": "reviewed",
                "supplier_id": "SUP-019",
                "verdict": "FLAG",
                "rationale": "The loaded program produced a valid structured verdict offline.",
                "findings": ["[SQL] metrics reviewed"],
                "missing_evidence": [],
            }
        }
    )

    verdict = load_and_verify(example, lm=lm, program_path=saved.program_path)

    assert verdict.supplier_id == "SUP-019"
    assert verdict.verdict == "FLAG"
