.PHONY: install lock test lint typecheck check

install:
	uv sync --locked --group dev

lock:
	uv lock

test:
	uv run --locked pytest -q

lint:
	uv run --locked ruff check .

typecheck:
	uv run --locked mypy src/supplier_risk

check: lint typecheck test
