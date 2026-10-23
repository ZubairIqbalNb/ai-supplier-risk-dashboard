"""Shared, offline fixtures for the Module 3 retrieval tests.

Only ``tests/test_retrieval_real_model.py`` downloads or loads a real embedding
model; every other retrieval test uses the deterministic doubles from
:mod:`tests.support`.
"""

from __future__ import annotations

import pytest

from supplier_risk.contracts import SupplierDataset
from supplier_risk.data.generate import build_dataset
from supplier_risk.retrieval.corpus import CorpusDocument, build_corpus
from tests.support import StubEmbedder


@pytest.fixture(scope="session")
def module2_dataset() -> SupplierDataset:
    """The deterministic Module 2 dataset, rebuilt in memory and never written to disk."""
    return build_dataset()


@pytest.fixture(scope="session")
def corpus_documents(module2_dataset: SupplierDataset) -> tuple[CorpusDocument, ...]:
    return build_corpus(module2_dataset)


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()
