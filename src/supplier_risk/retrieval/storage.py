"""Artifact paths and safe persistence helpers for the retrieval stack.

Artifacts live under ``artifacts/retrieval/`` by default, which is Git-ignored.
Metadata and corpus documents are stored as JSON, never pickle. Single files are
written to a temporary sibling and moved into place with :func:`os.replace`, so
an interrupted write cannot leave a partially written file at the final path.
Replacing the several artifact files of one build is not a single transaction;
``manifest.json`` is written last and carries the corpus fingerprint so a
mismatched set of files is detected on load instead of being used silently.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import ValidationError

from supplier_risk.retrieval.errors import RetrievalArtifactError

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_DIR: Final = PROJECT_ROOT / "artifacts" / "retrieval"

ARTIFACT_VERSION: Final = "module3.v1"
ArtifactVersion = Literal["module3.v1"]

MANIFEST_FILENAME: Final = "manifest.json"
CORPUS_FILENAME: Final = "corpus.json"
DENSE_INDEX_FILENAME: Final = "dense.index.faiss"
DENSE_METADATA_FILENAME: Final = "dense.metadata.json"
LEXICAL_FILENAME: Final = "lexical.json"

BUILD_COMMAND: Final = "uv run --locked python -m supplier_risk.retrieval.build"


def describe_validation_error(error: ValidationError) -> str:
    """Return the first validation problem without echoing rejected input values."""
    details = error.errors(include_input=False, include_url=False)
    if not details:
        return "invalid payload"
    first = details[0]
    location = ".".join(str(part) for part in first["loc"]) or "<payload>"
    message = str(first["msg"])
    remaining = len(details) - 1
    suffix = f" (+{remaining} more)" if remaining > 0 else ""
    return f"{location}: {message}{suffix}"


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` through a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        reason = error.strerror or str(error)
        raise RetrievalArtifactError(
            f"could not write retrieval artifact {path}: {reason}"
        ) from None


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 ``text`` to ``path`` atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize ``payload`` as stable, human-readable JSON and write it atomically."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_artifact_text(path: Path, *, description: str) -> str:
    """Read a JSON artifact as text, raising a safe error when it is missing or unreadable.

    Callers validate the returned text with a Pydantic ``model_validate_json`` so
    that malformed JSON and schema violations are reported the same way.
    """
    if not path.is_file():
        raise RetrievalArtifactError(
            f"missing {description}: {path}. Build the retrieval artifacts with `{BUILD_COMMAND}`."
        )
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        raise RetrievalArtifactError(
            f"{description} at {path} is unreadable or corrupt; "
            f"rebuild it with `{BUILD_COMMAND}`."
        ) from None
