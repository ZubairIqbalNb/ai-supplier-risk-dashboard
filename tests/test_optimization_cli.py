"""CLI tests for Module 7.

The configuration path and argument handling are tested directly. One test runs
the whole CLI offline by substituting only the credential loader, the LM builder,
and the retrieval engine, so the real DSPy ``BootstrapFewShot`` compile, save, and
evaluation all execute without a network call.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from supplier_risk.config import ConfigError, LanguageModelSettings
from supplier_risk.data.generate import build_dataset
from supplier_risk.decision.lm import RoleLanguageModels
from supplier_risk.optimization import cli
from supplier_risk.optimization.errors import ArtifactError
from tests.optimization_support import dummy_lm, retrieval_for

DATASET = build_dataset()


def test_parse_args_defaults() -> None:
    args = cli._parse_args([])

    assert args.seed == 20261019
    assert args.check_config is False
    assert args.skip_dev is False
    assert args.max_bootstrapped_demos == 4
    assert args.max_labeled_demos == 0
    assert args.train_limit is None


def _write_env(path: Path) -> None:
    path.write_text(
        "GEMINI_API_KEY=test-key\n"
        "GEMINI_TEACHER_MODEL=gemini-2.5-pro\n"
        "GEMINI_STUDENT_MODEL=gemini-2.0-flash\n",
        encoding="utf-8",
    )


def test_check_config_returns_zero_without_touching_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file)

    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("check-config must not run retrieval or the dataset")

    monkeypatch.setattr(cli, "build_dataset", fail)
    monkeypatch.setattr(cli, "query_all_sources", fail)

    assert cli.main(["--check-config", "--env-file", str(env_file)]) == 0


def test_missing_configuration_returns_two(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.env"

    assert cli.main(["--check-config", "--env-file", str(missing)]) == 2


def test_invalid_optimizer_settings_return_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file)

    assert cli.main(["--check-config", "--env-file", str(env_file), "--train-limit", "0"]) == 2


def test_config_error_from_loader_returns_two(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_config(*args: object, **kwargs: object) -> LanguageModelSettings:
        raise ConfigError("Missing or blank required environment variable(s): GEMINI_API_KEY")

    monkeypatch.setattr(cli, "load_language_model_settings", raise_config)

    assert cli.main(["--check-config"]) == 2


def test_compilation_failure_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = LanguageModelSettings(
        api_key=SecretStr("test-key"),
        teacher_model="gemini-2.5-pro",
        student_model="gemini-2.0-flash",
    )
    monkeypatch.setattr(cli, "load_language_model_settings", lambda *a, **k: settings)
    monkeypatch.setattr(
        cli,
        "build_role_language_models",
        lambda *a, **k: RoleLanguageModels(
            teacher=SimpleNamespace(model="gemini/teacher"),
            student=SimpleNamespace(model="gemini/student"),
        ),
    )
    monkeypatch.setattr(
        cli, "run_workflow", lambda **kwargs: (_ for _ in ()).throw(ArtifactError("disk full"))
    )

    assert cli.main(["--artifact-dir", str(tmp_path)]) == 1


def test_full_cli_run_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = LanguageModelSettings(
        api_key=SecretStr("test-key"),
        teacher_model="dummy-teacher",
        student_model="dummy-student",
    )
    lm = dummy_lm(DATASET)
    monkeypatch.setattr(cli, "load_language_model_settings", lambda *a, **k: settings)
    monkeypatch.setattr(
        cli,
        "build_role_language_models",
        lambda *a, **k: RoleLanguageModels(teacher=lm, student=lm),
    )
    monkeypatch.setattr(cli, "query_all_sources", retrieval_for)

    exit_code = cli.main(
        [
            "--artifact-dir",
            str(tmp_path),
            "--train-limit",
            "3",
            "--dev-limit",
            "2",
            "--held-out-limit",
            "2",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / cli.RESULTS_FILENAME).exists()
    assert (tmp_path / "optimized_student.json").exists()
    assert (tmp_path / "optimized_student.metadata.json").exists()
    assert "api_key" not in (tmp_path / "optimized_student.json").read_text().lower()
