"""Teacher/student language-model configuration tests for Module 6.

The tests use injected environment mappings and a fake LM factory, so no dotenv
file is read, no credential is used, and no provider call is made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import supplier_risk.config as config_module
from supplier_risk.config import ConfigError, load_language_model_settings
from supplier_risk.decision import (
    LM_ROLES,
    RoleLanguageModels,
    build_role_language_models,
    dspy_model_name,
)

SENTINEL = "test-gemini-key"
BASE_ENV = {
    "GEMINI_API_KEY": SENTINEL,
    "GEMINI_TEACHER_MODEL": "gemini-2.5-pro",
    "GEMINI_STUDENT_MODEL": "gemini-2.0-flash",
}


class RecordingFactory:
    """Fake ``dspy.LM`` constructor that records its keyword arguments."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.models: list[object] = []

    def __call__(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        model = SimpleNamespace(model=kwargs["model"])
        self.models.append(model)
        return model


def test_loads_teacher_and_student_models() -> None:
    settings = load_language_model_settings(environ=BASE_ENV)

    assert settings.teacher_model == "gemini-2.5-pro"
    assert settings.student_model == "gemini-2.0-flash"
    assert settings.api_key.get_secret_value() == SENTINEL
    assert settings.masked()["api_key"] == "***"


@pytest.mark.parametrize(
    "name", ["GEMINI_API_KEY", "GEMINI_TEACHER_MODEL", "GEMINI_STUDENT_MODEL"]
)
def test_missing_variable_is_reported_by_name(name: str) -> None:
    env = dict(BASE_ENV)
    env[name] = "   "

    with pytest.raises(ConfigError, match=name):
        load_language_model_settings(environ=env)


def test_secret_never_appears_in_representations() -> None:
    settings = load_language_model_settings(environ=BASE_ENV)

    for rendered in (
        repr(settings),
        str(settings),
        repr(settings.masked()),
        str(settings.masked()),
    ):
        assert SENTINEL not in rendered


def test_environ_mapping_does_not_read_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> dict[str, str | None]:
        raise AssertionError("dotenv_values must not be called when environ is supplied")

    monkeypatch.setattr(config_module, "dotenv_values", fail_if_called)

    settings = load_language_model_settings(environ=BASE_ENV, env_file=".env")

    assert settings.student_model == "gemini-2.0-flash"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("gemini-2.0-flash", "gemini/gemini-2.0-flash"),
        ("gemini/gemini-2.5-pro", "gemini/gemini-2.5-pro"),
        ("openai/gpt-4o-mini", "openai/gpt-4o-mini"),
    ],
)
def test_model_names_keep_explicit_providers(configured: str, expected: str) -> None:
    assert dspy_model_name(configured) == expected


def test_blank_model_name_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must not be blank"):
        dspy_model_name("   ")


def test_builds_distinct_teacher_and_student_roles() -> None:
    factory = RecordingFactory()
    settings = load_language_model_settings(environ=BASE_ENV)

    models = build_role_language_models(settings, lm_factory=factory)

    assert isinstance(models, RoleLanguageModels)
    assert [call["model"] for call in factory.calls] == [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.0-flash",
    ]
    assert [call["api_key"] for call in factory.calls] == [SENTINEL, SENTINEL]
    assert models.teacher is factory.models[0]
    assert models.student is factory.models[1]
    assert models.for_role("teacher") is models.teacher
    assert models.for_role("student") is models.student
    assert LM_ROLES == ("teacher", "student")


def test_masked_roles_never_include_the_key() -> None:
    factory = RecordingFactory()
    settings = load_language_model_settings(environ=BASE_ENV)

    models = build_role_language_models(settings, lm_factory=factory)

    assert models.masked() == {
        "teacher_model": "gemini/gemini-2.5-pro",
        "student_model": "gemini/gemini-2.0-flash",
    }
    assert SENTINEL not in json.dumps(models.masked())


def test_unknown_role_is_rejected() -> None:
    factory = RecordingFactory()
    models = build_role_language_models(
        load_language_model_settings(environ=BASE_ENV), lm_factory=factory
    )

    with pytest.raises(ValueError, match="unknown language-model role"):
        models.for_role("judge")  # type: ignore[arg-type]


def test_default_factory_builds_offline_dspy_lms() -> None:
    models = build_role_language_models(load_language_model_settings(environ=BASE_ENV))

    assert models.masked() == {
        "teacher_model": "gemini/gemini-2.5-pro",
        "student_model": "gemini/gemini-2.0-flash",
    }
