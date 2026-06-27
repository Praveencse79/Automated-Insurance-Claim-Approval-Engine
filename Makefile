# Developer convenience commands for the Claim Approval Engine.
# Usage: `make <target>`

.DEFAULT_GOAL := help
PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help venv install install-dev demo seed test lint typecheck check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)

install: venv ## Install runtime dependencies
	$(BIN)/pip install --upgrade pip && $(BIN)/pip install -r requirements.txt

install-dev: venv ## Install runtime + dev/test dependencies
	$(BIN)/pip install --upgrade pip && $(BIN)/pip install -r requirements-dev.txt

demo: ## Run the end-to-end demo (mock mode)
	$(BIN)/python scripts/run_demo.py

seed: ## Seed the knowledge base into the vector store
	$(BIN)/python scripts/seed_knowledge_base.py

test: ## Run the test suite
	$(BIN)/python -m pytest

lint: ## Lint with ruff
	$(BIN)/ruff check src tests scripts

typecheck: ## Static type-check with mypy
	$(BIN)/mypy src

check: lint typecheck test ## Run lint + types + tests

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ build dist *.egg-info
