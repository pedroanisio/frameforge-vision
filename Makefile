UV ?= uv

.PHONY: help sync test schema schema-check goldens lint check build clean

help:  ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync:  ## install the dev environment
	$(UV) sync --all-groups

test:  ## run the measurement suite (seam tests skip without `author`)
	$(UV) run pytest

seam:  ## run the cross-package seam tests too (needs the `author` extra)
	$(UV) sync --extra author --group dev && $(UV) run pytest

goldens:  ## rewrite tests/golden/ — ONLY when the contract is meant to move
	$(UV) run python tests/regen_goldens.py

lint:  ## ruff
	$(UV) run ruff check src tests

check: lint test  ## every local gate

build:  ## build the wheel + sdist
	$(UV) build

clean:
	rm -rf dist build .pytest_cache .ruff_cache htmlcov .coverage
