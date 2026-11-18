.PHONY: check fix

check:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run pyright
	uv run lint-imports
	uv run pytest

fix:
	uv run ruff format src tests
	uv run ruff check --fix src tests
