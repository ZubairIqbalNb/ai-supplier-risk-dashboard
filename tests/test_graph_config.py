from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from supplier_risk.config import ConfigError, Neo4jSettings, load_neo4j_settings

SENTINEL = "graph-secret-value"


def test_neo4j_settings_default_uri_and_user() -> None:
    settings = load_neo4j_settings(environ={"NEO4J_PASSWORD": SENTINEL})

    assert settings.uri == "bolt://localhost:7687"
    assert settings.user == "neo4j"
    assert settings.password.get_secret_value() == SENTINEL


def test_neo4j_settings_do_not_require_a_gemini_key() -> None:
    settings = load_neo4j_settings(
        environ={
            "NEO4J_URI": "neo4j://graph.example:7687",
            "NEO4J_USER": "graph-user",
            "NEO4J_PASSWORD": SENTINEL,
        }
    )

    assert settings.uri == "neo4j://graph.example:7687"
    assert settings.user == "graph-user"


def test_neo4j_settings_ignore_the_dotenv_file_when_environ_is_given() -> None:
    settings = load_neo4j_settings(
        env_file=Path("/definitely/not/a/real/env-file"),
        environ={"NEO4J_PASSWORD": SENTINEL},
    )

    assert settings.password.get_secret_value() == SENTINEL


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"NEO4J_URI": "bolt://localhost:7687", "NEO4J_USER": "neo4j"}, "NEO4J_PASSWORD"),
        ({"NEO4J_PASSWORD": "   "}, "NEO4J_PASSWORD"),
        ({"NEO4J_PASSWORD": SENTINEL, "NEO4J_URI": "   "}, "NEO4J_URI"),
        ({"NEO4J_PASSWORD": SENTINEL, "NEO4J_USER": "  "}, "NEO4J_USER"),
    ],
)
def test_missing_or_blank_values_are_named_without_values(
    overrides: dict[str, str], missing: str
) -> None:
    with pytest.raises(ConfigError, match=missing) as info:
        load_neo4j_settings(environ=overrides)

    assert SENTINEL not in str(info.value)


@pytest.mark.parametrize(
    "uri",
    ["http://localhost:7474", "localhost:7687", "bolt://", "not-a-uri"],
)
def test_invalid_uri_is_rejected_without_echoing_it(uri: str) -> None:
    with pytest.raises(ConfigError, match="uri") as info:
        load_neo4j_settings(environ={"NEO4J_URI": uri, "NEO4J_PASSWORD": SENTINEL})

    assert uri not in str(info.value)
    assert SENTINEL not in str(info.value)


def test_uri_with_embedded_credentials_is_rejected_without_echoing_them() -> None:
    with pytest.raises(ConfigError, match="uri") as info:
        load_neo4j_settings(
            environ={
                "NEO4J_URI": "bolt://graph-user:uri-secret@localhost:7687",
                "NEO4J_PASSWORD": SENTINEL,
            }
        )

    assert "uri-secret" not in str(info.value)
    assert SENTINEL not in str(info.value)


def test_direct_settings_construction_rejects_a_blank_password() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        Neo4jSettings(password=SecretStr("   "))


def test_masked_representation_never_contains_the_password() -> None:
    settings = Neo4jSettings(
        uri="bolt://localhost:7687",
        user="neo4j",
        password=SecretStr(SENTINEL),
    )

    for rendered in (repr(settings), str(settings), str(settings.masked())):
        assert SENTINEL not in rendered
        assert "***" in rendered
