# Makefile for Fugatto Audio Lab
# Development workflow automation

.PHONY: help install install-dev test lint format type-check security clean docs docker

help: ## Show this help message
	@echo "Fugatto Audio Lab - Development Commands"
	@echo "======================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,eval]"
	pre-commit install

test: ## Run test suite
	pytest --cov=fugatto_lab --cov-report=term-missing --cov-report=html

test-quick: ## Run tests without coverage
	pytest -x -v

lint: ## Run linting checks
	ruff check fugatto_lab tests
	flake8 fugatto_lab tests

format: ## Format code
	black fugatto_lab tests
	isort fugatto_lab tests

type-check: ## Run type checking
	mypy fugatto_lab

security: ## Run security checks
	bandit -r fugatto_lab/ -f json -o reports/bandit-report.json
	safety check --json --output reports/safety-report.json

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

quality: format lint type-check security ## Run all quality checks

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	cd docs && sphinx-build -b html . _build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docker-build: ## Build Docker image
	docker build -t fugatto-audio-lab .

docker-run: ## Run Docker container
	docker run --gpus all -p 7860:7860 -p 8501:8501 fugatto-audio-lab

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

release-patch: ## Release patch version
	bumpversion patch
	git push && git push --tags

release-minor: ## Release minor version
	bumpversion minor
	git push && git push --tags

release-major: ## Release major version
	bumpversion major
	git push && git push --tags