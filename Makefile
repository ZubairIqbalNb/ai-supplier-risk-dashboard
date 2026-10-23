.PHONY: install lock index test test-real-model lint typecheck check

install:
	uv sync --locked --group dev

lock:
	uv lock

index:
	uv run --locked python -m supplier_risk.retrieval.build

test:
	uv run --locked pytest -q

test-real-model:
	SUPPLIER_RISK_REAL_MODEL_TESTS=1 uv run --locked pytest -q -m real_model

lint:
	uv run --locked ruff check .

typecheck:
	uv run --locked mypy src/supplier_risk

check: lint typecheck test
