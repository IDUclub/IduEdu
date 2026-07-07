PACKAGE := iduedu
TESTS := tests
FORMAT_PATHS := $(PACKAGE) $(TESTS)
PYTHON ?= python
UV ?= uv
UV_RUN ?= $(UV) run
COVERAGE_XML ?= coverage.xml
COVERAGE_HTML ?= htmlcov

.PHONY: help install install-dev lock update lint format format-check test test-unit test-numba test-network test-all \
        coverage coverage-xml coverage-html coverage-numba docs clean build publish version version-next changelog

help:
	@echo "Available targets:"
	@echo "  install           - install package into current Python environment"
	@echo "  install-dev       - sync all dependency groups with uv"
	@echo "  lint              - run pylint on package code"
	@echo "  format            - format code with isort + black"
	@echo "  format-check      - check code formatting with isort + black"
	@echo "  test              - run fast default test suite (network tests excluded)"
	@echo "  test-unit         - run unit tests only"
	@echo "  test-numba        - run Numba tests only"
	@echo "  test-network      - run opt-in network tests only"
	@echo "  test-all          - run all tests, including network tests"
	@echo "  coverage          - run fast tests with terminal coverage report"
	@echo "  coverage-xml      - run all tests (incl. network) and write coverage.xml for CI/Codecov"
	@echo "  coverage-html     - run fast tests and write HTML coverage report"
	@echo "  coverage-numba    - run Numba tests and show Numba coverage only"
	@echo "  docs              - build documentation"
	@echo "  build             - build wheel and sdist with uv"
	@echo "  publish           - publish package with uv"
	@echo "  version           - print the current version (from $(PACKAGE)/_version.py)"
	@echo "  version-next      - print the next version semantic-release would compute (no changes)"
	@echo "  changelog         - regenerate CHANGELOG.md from commit history (no release)"

install:
	$(PYTHON) -m pip install .

install-dev:
	$(UV) sync --all-groups

lock:
	$(UV) lock

update:
	$(UV) lock --upgrade

lint:
	$(UV_RUN) python -m pylint $(PACKAGE)

format:
	$(UV_RUN) python -m isort $(FORMAT_PATHS)
	$(UV_RUN) python -m black $(FORMAT_PATHS)

format-check:
	$(UV_RUN) python -m isort --check-only $(FORMAT_PATHS)
	$(UV_RUN) python -m black --check $(FORMAT_PATHS)

test:
	$(UV_RUN) python -m pytest -q

test-unit:
	$(UV_RUN) python -m pytest -q -m "unit and not network"

test-numba:
	$(UV_RUN) python -m pytest -q -m numba

test-network:
	$(UV_RUN) python -m pytest --run-network -q -m network

test-all:
	$(UV_RUN) python -m pytest --run-network -q

coverage:
	$(UV_RUN) python -m coverage erase
	$(UV_RUN) python -m coverage run -m pytest -q
	$(UV_RUN) python -m coverage report -m

coverage-xml:
	$(UV_RUN) python -m coverage erase
	$(UV_RUN) python -m coverage run -m pytest --run-network -q
	$(UV_RUN) python -m coverage xml -o $(COVERAGE_XML)
	$(UV_RUN) python -m coverage report -m

coverage-html:
	$(UV_RUN) python -m coverage erase
	$(UV_RUN) python -m coverage run -m pytest -q
	$(UV_RUN) python -m coverage html -d $(COVERAGE_HTML)
	$(UV_RUN) python -m coverage report -m

coverage-numba:
	$(UV_RUN) python -m coverage erase
	$(UV_RUN) python -m coverage run -m pytest tests/test_numba_unit.py -q
	$(UV_RUN) python -m coverage report --include="$(PACKAGE)/_numba/*" -m

docs:
	$(UV_RUN) sphinx-build -b html docs docs/_build/html

clean:
	rm -rf ./dist ./build ./*.egg-info ./$(COVERAGE_HTML) ./.coverage ./.coverage.* ./$(COVERAGE_XML)

build:
	$(UV) build

publish:
	$(UV) publish

# Releases are automated on push to main (python-semantic-release); see CONTRIBUTING.md.
# These targets are read-only helpers for inspecting versioning locally.
version:
	$(UV_RUN) python -c "from iduedu._version import VERSION; print(VERSION)"

version-next:
	$(UV_RUN) semantic-release --noop version --print

changelog:
	$(UV_RUN) semantic-release changelog
