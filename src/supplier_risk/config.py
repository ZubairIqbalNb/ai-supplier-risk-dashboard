"""Environment-backed configuration for the Supplier Risk Dashboard."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, cast
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, field_validator

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_NEO4J_URI = "bolt://localhost:7687"
_DEFAULT_NEO4J_USER = "neo4j"
_DEFAULT_LOG_LEVEL: LogLevel = "INFO"
_NEO4J_URI_SCHEMES: Final = frozenset({"bolt", "neo4j", "bolt+s", "bolt+ssc", "neo4j+s", "neo4j+ssc"})


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


class Neo4jSettings(BaseModel):
    """Validated Neo4j connection settings for the Module 4 graph store.

    The password is required here even though it is optional in
    :class:`Settings`, because the graph engine cannot run without one. Values
    are never included in representations or error messages.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    uri: str = _DEFAULT_NEO4J_URI
    user: str = _DEFAULT_NEO4J_USER
    password: SecretStr

    @field_validator("uri")
    @classmethod
    def _uri_is_a_bolt_or_neo4j_endpoint(cls, value: str) -> str:
        text = value.strip()
        parts = urlsplit(text)
        if parts.scheme not in _NEO4J_URI_SCHEMES or not parts.hostname:
            raise ValueError("NEO4J_URI must be a bolt:// or neo4j:// URI with a host")
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                "NEO4J_URI must not include credentials; use NEO4J_USER and NEO4J_PASSWORD"
            )
        return text

    @field_validator("user")
    @classmethod
    def _user_must_not_be_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("NEO4J_USER must not be blank")
        return text

    @field_validator("password")
    @classmethod
    def _password_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("NEO4J_PASSWORD must not be blank")
        return value

    def masked(self) -> dict[str, str]:
        """Return a display-safe representation without the password."""
        return {
            "uri": self.uri,
            "user": self.user,
            "password": "***",
        }

    def __repr__(self) -> str:
        return f"Neo4jSettings({self.masked()!r})"

    __str__ = __repr__


class Settings(BaseModel):
    """Validated application configuration.

    Secret fields use :class:`pydantic.SecretStr`, so their values are not
    included in normal representations. Use :meth:`masked` for display.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    gemini_api_key: SecretStr
    gemini_teacher_model: str | None = None
    gemini_student_model: str | None = None
    neo4j_uri: str = _DEFAULT_NEO4J_URI
    neo4j_user: str = _DEFAULT_NEO4J_USER
    neo4j_password: SecretStr | None = None
    log_level: LogLevel = _DEFAULT_LOG_LEVEL

    @field_validator("gemini_api_key")
    @classmethod
    def _gemini_api_key_must_not_be_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("GEMINI_API_KEY must not be blank")
        return value

    @field_validator("neo4j_password")
    @classmethod
    def _blank_neo4j_password_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        if not value.get_secret_value().strip():
            return None
        return value

    def masked(self) -> dict[str, str]:
        """Return a display-safe representation without secret values."""
        return {
            "gemini_api_key": "***",
            "gemini_teacher_model": self.gemini_teacher_model or "",
            "gemini_student_model": self.gemini_student_model or "",
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": "***" if self.neo4j_password is not None else "",
            "log_level": self.log_level,
        }

    def __repr__(self) -> str:
        return f"Settings({self.masked()!r})"

    __str__ = __repr__


def _read_env_file(env_file: str | Path | None) -> dict[str, str]:
    """Read a dotenv file without mutating the process environment."""
    if env_file is None:
        return {}
    path = Path(env_file)
    if not path.is_file():
        return {}
    values = dotenv_values(path)
    return {key: value for key, value in values.items() if value is not None}


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _optional_secret(value: str | None) -> SecretStr | None:
    text = _optional_text(value)
    return SecretStr(text) if text is not None else None


def _sanitized_validation_message(error: ValidationError) -> str:
    """Return field locations only; never include rejected input values."""
    locations: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail.get("loc", ())) or "configuration"
        locations.append(location)
    unique_locations = ", ".join(dict.fromkeys(locations))
    return f"Invalid configuration for: {unique_locations}"


def _load_environment(
    env_file: str | Path | None,
    environ: Mapping[str, str] | None,
) -> Mapping[str, str]:
    """Resolve configuration values without mutating the process environment."""
    if environ is not None:
        return environ
    return {**_read_env_file(env_file), **os.environ}


def load_settings(
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate configuration.

    When ``environ`` is supplied, the dotenv file is not read at all.
    Otherwise, values from ``env_file`` are merged with the process environment,
    with process environment values taking precedence.
    """
    source = _load_environment(env_file, environ)

    api_key = (source.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError("Missing or blank required environment variable: GEMINI_API_KEY")

    raw_log_level = (source.get("LOG_LEVEL") or _DEFAULT_LOG_LEVEL).strip().upper()
    log_level = cast(LogLevel, raw_log_level)

    try:
        return Settings(
            gemini_api_key=SecretStr(api_key),
            gemini_teacher_model=_optional_text(source.get("GEMINI_TEACHER_MODEL")),
            gemini_student_model=_optional_text(source.get("GEMINI_STUDENT_MODEL")),
            neo4j_uri=(source.get("NEO4J_URI") or _DEFAULT_NEO4J_URI).strip(),
            neo4j_user=(source.get("NEO4J_USER") or _DEFAULT_NEO4J_USER).strip(),
            neo4j_password=_optional_secret(source.get("NEO4J_PASSWORD")),
            log_level=log_level,
        )
    except ValidationError as error:
        raise ConfigError(_sanitized_validation_message(error)) from None


def load_neo4j_settings(
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> Neo4jSettings:
    """Load and validate the Neo4j connection settings.

    This loader is independent of ``GEMINI_API_KEY`` so graph-only commands and
    tests never need a language-model credential. When ``environ`` is supplied,
    the dotenv file is not read at all.

    Raises:
        ConfigError: if a required Neo4j variable is missing or invalid. Error
            messages name the variables only; values are never echoed.
    """
    source = _load_environment(env_file, environ)
    uri = (source.get("NEO4J_URI") or _DEFAULT_NEO4J_URI).strip()
    user = (source.get("NEO4J_USER") or _DEFAULT_NEO4J_USER).strip()
    password = _optional_secret(source.get("NEO4J_PASSWORD"))

    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_USER", user),
            ("NEO4J_PASSWORD", password),
        )
        if value is None or (isinstance(value, str) and not value)
    ]
    if missing:
        names = ", ".join(missing)
        raise ConfigError(f"Missing or blank required environment variable(s): {names}")
    assert password is not None

    try:
        return Neo4jSettings(uri=uri, user=user, password=password)
    except ValidationError as error:
        raise ConfigError(_sanitized_validation_message(error)) from None
