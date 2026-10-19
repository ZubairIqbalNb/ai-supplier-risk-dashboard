from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

import supplier_risk
import supplier_risk.config as config_module
from supplier_risk.config import ConfigError, Settings, load_settings

SENTINEL = "test-secret-value"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_KEYS = {
    "GEMINI_API_KEY",
    "GEMINI_TEACHER_MODEL",
    "GEMINI_STUDENT_MODEL",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "LOG_LEVEL",
}


def _base_env(**overrides: str) -> dict[str, str]:
    env = {"GEMINI_API_KEY": SENTINEL}
    env.update(overrides)
    return env


def _read_example_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def test_package_imports() -> None:
    assert supplier_risk.__version__ == "0.1.0"


def test_load_settings_from_mapping() -> None:
    settings = load_settings(environ=_base_env())

    assert settings.gemini_api_key.get_secret_value() == SENTINEL
    assert settings.gemini_teacher_model is None
    assert settings.gemini_student_model is None
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password is None
    assert settings.log_level == "INFO"


def test_missing_api_key_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_settings(environ={})


def test_blank_api_key_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_settings(environ={"GEMINI_API_KEY": "   "})


def test_optional_values_are_trimmed_and_blank_password_becomes_none() -> None:
    settings = load_settings(
        environ=_base_env(
            GEMINI_TEACHER_MODEL="  teacher-model  ",
            GEMINI_STUDENT_MODEL="  student-model  ",
            NEO4J_URI="  bolt://example:7687  ",
            NEO4J_USER="  graph-user  ",
            NEO4J_PASSWORD="   ",
        )
    )

    assert settings.gemini_teacher_model == "teacher-model"
    assert settings.gemini_student_model == "student-model"
    assert settings.neo4j_uri == "bolt://example:7687"
    assert settings.neo4j_user == "graph-user"
    assert settings.neo4j_password is None


def test_unknown_variables_are_ignored() -> None:
    settings = load_settings(environ=_base_env(GEMINI_MODEL="legacy-value"))

    assert settings.gemini_teacher_model is None
    assert not hasattr(settings, "gemini_model")


def test_invalid_log_level_is_rejected_without_leaking_secret() -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_settings(environ=_base_env(LOG_LEVEL="VERBOSE"))

    message = str(exc_info.value)
    assert "log_level" in message
    assert SENTINEL not in message


def test_secret_values_are_masked_in_representations() -> None:
    settings = load_settings(environ=_base_env(NEO4J_PASSWORD=SENTINEL))
    masked = settings.masked()

    assert masked["gemini_api_key"] == "***"
    assert masked["neo4j_password"] == "***"
    for rendered in (repr(settings), str(settings), repr(masked), str(masked)):
        assert SENTINEL not in rendered


def test_direct_settings_construction_rejects_blank_api_key() -> None:
    with pytest.raises(ValidationError):
        Settings(gemini_api_key=SecretStr("   "))


def test_environ_mapping_does_not_read_dotenv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> dict[str, str | None]:
        raise AssertionError("dotenv_values must not be called when environ is supplied")

    monkeypatch.setattr(config_module, "dotenv_values", fail_if_called)

    settings = load_settings(environ=_base_env(), env_file=".env")

    assert settings.gemini_api_key.get_secret_value() == SENTINEL


def test_env_example_contains_only_expected_placeholders() -> None:
    values = _read_example_values()

    assert set(values) == EXAMPLE_KEYS
    assert values["GEMINI_API_KEY"] == "replace-me"
    assert values["GEMINI_TEACHER_MODEL"] == ""
    assert values["GEMINI_STUDENT_MODEL"] == ""
    assert values["NEO4J_URI"] == "bolt://localhost:7687"
    assert values["NEO4J_USER"] == "neo4j"
    assert values["NEO4J_PASSWORD"] == "replace-me"
    assert values["LOG_LEVEL"] == "INFO"
