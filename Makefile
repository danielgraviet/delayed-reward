.PHONY: check fix fuzz smoke-daytona

check:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run pyright
	uv run lint-imports
	uv run pytest

fix:
	uv run ruff format src tests
	uv run ruff check --fix src tests

fuzz:
	HYPOTHESIS_PROFILE=fuzz uv run pytest tests/test_fuzz_config.py tests/test_fuzz_envs.py

smoke-daytona:
	set -a; . ./.env; set +a; uv run --group cloud pytest -n 0 -m daytona tests/test_cli_live.py
