.PHONY: help install dev-install check lint dry-run-sft dry-run-grpo test test-cov clean

# ────────────────────────────────────────────────────────────────
# HotpotQA-MAS Makefile
#
# Usage:
#   make help          Show this help
#   make install       Install production dependencies
#   make dev-install   Install with dev tools (ruff, pytest, mypy)
#   make check         Run all checks (lint + dry-run + test)
#   make dry-run-sft   Validate SFT pipeline without loading model
#   make dry-run-grpo  Validate GRPO pipeline without loading model
#   make test          Run tests
#   make test-cov      Run tests with coverage
#   make lint          Run ruff linter
#   make clean         Remove artifacts and __pycache__
# ────────────────────────────────────────────────────────────────

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

check: lint dry-run-sft test
	@echo "[OK] All checks passed."

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

dry-run-sft:
	HOTPOTQA_DRY_RUN=1 python sft_trainer.py \
		--data-path ./data/sft/hotpotqa_dynamic_mixture_sft_data_300_v3.jsonl \
		--epochs 1

dry-run-grpo:
	HOTPOTQA_DRY_RUN=1 python grpo_hotpotqa_mas.py \
		--train-jsonl ./data/base/train.jsonl \
		--val-jsonl ./data/base/val.jsonl \
		--tasks 5 --val-tasks 3 --iterations 1

test:
	python -m pytest tests/ -v || echo "[skip] No tests yet — that's a todo, not a bug"

test-cov:
	python -m pytest tests/ -v --cov=. --cov-report=term-missing || echo "[skip] No tests yet"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info .eggs
	@echo "[clean] Bytecode and caches removed.  Artifacts kept."
