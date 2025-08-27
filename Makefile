.PHONY: help install install-dev install-docs lint format type-check test test-cov clean pre-commit-install pre-commit-run docs-init docs-build docs-apidoc docs-serve docs-clean docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	uv pip install -e .

install-dev: ## Install the package with development dependencies
	uv pip install -e ".[dev]"

install-docs: ## Install the package with documentation dependencies
	uv pip install -e ".[docs]"

lint: ## Run linting (ruff)
	uv run ruff check .
	uv run ruff format --check .

format: ## Format code (black, isort, ruff)
	uv run isort .
	uv run ruff format .

type-check: ## Run type checking (mypy)
	uv run mypy src

security-check: ## Run security checks (bandit)
	uv run bandit -r src

doc-check: ## Run documentation style checks (pydocstyle)
	uv run pydocstyle src

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest bertblocks --cov=bertblocks.modeling --cov-report=term-missing --cov-report=html


clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

docs-build: ## Build Sphinx documentation
	uv run sphinx-build -b html docs docs/_build/html --keep-going

docs-serve: ## Serve documentation locally (requires Python http.server)
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build artifacts
	rm -rf docs/_build/

docs: docs-apidoc docs-build ## Build complete documentation (API + Sphinx)

check: lint type-check security-check doc-check ## Run all checks

ci: check test ## Run all CI checks

fix: format ## Fix all auto-fixable issues
	uv run ruff check --fix .

# Development workflow commands
dev-setup: install-dev pre-commit-install ## Set up development environment
	@echo "Development environment set up successfully!"
	@echo "Run 'make help' to see available commands."

dev-check: fix check test ## Run full development check (fix, lint, type-check, test)
