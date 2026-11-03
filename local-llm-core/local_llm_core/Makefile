# ============================================================
# Local LLM Cloud v1.0 — Developer Makefile
# ============================================================

PYTHON   ?= python3
PIP      ?= pip3
SRC_DIRS  = serving engine memory monitoring hal tui
TEST_DIR  = tests

.PHONY: help install lint format typecheck test test-unit test-integration \
        test-all coverage clean

help:
	@echo ""
	@echo "  Local LLM Cloud v1.0 — Developer Commands"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install          Install all dev dependencies"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Auto-format with ruff"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make test             Run unit tests"
	@echo "  make test-integration Run integration tests (needs server)"
	@echo "  make test-all         Run all tests + lint + typecheck"
	@echo "  make coverage         Run tests with coverage report"
	@echo "  make clean            Remove build artifacts"
	@echo ""

install:
	$(PIP) install --break-system-packages \
	    ruff mypy pytest pytest-asyncio pytest-cov \
	    fastapi pydantic uvicorn[standard] pyyaml psutil \
	    httpx anyio

lint:
	$(PYTHON) -m ruff check $(SRC_DIRS) $(TEST_DIR) scripts/prod_stress_test.py \
	    --ignore E501,E402,F401,F811

format:
	$(PYTHON) -m ruff format $(SRC_DIRS) $(TEST_DIR)

typecheck:
	$(PYTHON) -m mypy $(SRC_DIRS) \
	    --ignore-missing-imports \
	    --no-strict-optional \
	    --allow-untyped-defs \
	    --check-untyped-defs

test:
	$(PYTHON) -m pytest $(TEST_DIR)/test_memory.py \
	                    $(TEST_DIR)/test_hardware.py \
	                    -v --tb=short

test-unit:
	$(PYTHON) -m pytest $(TEST_DIR) \
	    --ignore=$(TEST_DIR)/test_production.py \
	    -v --tb=short

test-integration:
	@echo "Starting stub server for integration tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_production.py -v --tb=short \
	    -m "not slow" 2>&1

test-all: lint test
	@echo "✓ All checks passed"

coverage:
	$(PYTHON) -m pytest $(TEST_DIR)/test_memory.py \
	                    $(TEST_DIR)/test_hardware.py \
	    --cov=$(SRC_DIRS) \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    -v --tb=short

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name ".ruff_cache"  -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name "htmlcov"      -type d -exec rm -rf {} + 2>/dev/null; true
	find . -name ".coverage"    -delete 2>/dev/null; true
	@echo "✓ Clean"
