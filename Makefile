.PHONY: ci lint typecheck test

ci: lint typecheck test

lint:
	ruff check htf/ tests/ --output-format=github

typecheck:
	mypy htf/ --ignore-missing-imports --no-strict-optional

test:
	python3 -m pytest -q --tb=short
