.PHONY: install lock index graph-ingest sql-build neo4j-up neo4j-down test test-graph test-real-model lint typecheck check

install:
	uv sync --locked --group dev

lock:
	uv lock

index:
	uv run --locked python -m supplier_risk.retrieval.build

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose stop neo4j

graph-ingest:
	uv run --locked python -m supplier_risk.graph.ingest

sql-build:
	uv run --locked python -m supplier_risk.sql.build

test:
	uv run --locked pytest -q

test-graph:
	SUPPLIER_RISK_NEO4J_TESTS=1 uv run --locked pytest -q -m neo4j

test-real-model:
	SUPPLIER_RISK_REAL_MODEL_TESTS=1 uv run --locked pytest -q -m real_model

lint:
	uv run --locked ruff check .

typecheck:
	uv run --locked mypy src/supplier_risk

check: lint typecheck test
