.PHONY: help install demo clean test lint format check doctor

PYTHON := python
PIP := pip
OUT_DIR := out
DATA_DIR := data

help:  ## Show this help message
	@echo "REINVENT4 Meta-Controller"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
	$(PIP) install -e ".[dev]"
	@echo "✓ Installation complete"

demo: clean-demo  ## Run full demo (generate + report + medchem handoff)
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  REINVENT4 Meta-Controller Demo"
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	@echo "[1/3] Generating demo data..."
	@mkdir -p tests/fixtures/demo_run
	$(PYTHON) tests/generate_demo_data.py
	@echo ""
	@echo "[2/3] Building behavioral report..."
	@mkdir -p $(OUT_DIR)/demo_report
	$(PYTHON) -m r4mc.report_engine.cli build \
		--run_dir tests/fixtures/demo_run \
		--out_dir $(OUT_DIR)/demo_report \
		--verbose
	@echo ""
	@echo "[3/3] Generating medchem handoff..."
	$(PYTHON) -m r4mc.report_engine.cli medchem \
		--run_dir tests/fixtures/demo_run \
		--out_dir $(OUT_DIR)/demo_report/medchem_handoff \
		--molport_dir $(DATA_DIR) \
		--top_n 20 \
		--molport_similarity 0.7 \
		--molport_top_k 3 \
		--verbose
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "✓ Demo complete!"
	@echo ""
	@echo "View outputs:"
	@echo "  Report:   $(OUT_DIR)/demo_report/report.md"
	@echo "  Handoff:  $(OUT_DIR)/demo_report/medchem_handoff/index.html"
	@echo ""
	@echo "Quick view:"
	@echo "  cat $(OUT_DIR)/demo_report/report.md"
	@echo "═══════════════════════════════════════════════════════════"

run:  ## Run a custom campaign (usage: make run CONFIG=configs/my_config.yaml)
ifndef CONFIG
	@echo "Error: CONFIG not specified"
	@echo "Usage: make run CONFIG=configs/my_config.yaml"
	@exit 1
endif
	$(PYTHON) -m r4mc.run --config $(CONFIG)

report:  ## Generate report from run (usage: make report RUN_DIR=out/my_run)
ifndef RUN_DIR
	@echo "Error: RUN_DIR not specified"
	@echo "Usage: make report RUN_DIR=out/my_run"
	@exit 1
endif
	@mkdir -p $(RUN_DIR)_report
	$(PYTHON) -m r4mc.report_engine.cli build \
		--run_dir $(RUN_DIR) \
		--out_dir $(RUN_DIR)_report \
		--verbose

medchem:  ## Generate medchem handoff (usage: make medchem RUN_DIR=out/my_run)
ifndef RUN_DIR
	@echo "Error: RUN_DIR not specified"
	@echo "Usage: make medchem RUN_DIR=out/my_run"
	@exit 1
endif
	$(PYTHON) -m r4mc.report_engine.cli medchem \
		--run_dir $(RUN_DIR) \
		--out_dir $(RUN_DIR)_medchem \
		--molport_dir $(DATA_DIR) \
		--top_n 20 \
		--verbose

doctor:  ## Run diagnostics on a run directory
ifndef RUN_DIR
	@echo "Error: RUN_DIR not specified"
	@echo "Usage: make doctor RUN_DIR=out/my_run"
	@exit 1
endif
	$(PYTHON) -m r4mc.report_engine.cli doctor --run_dir $(RUN_DIR) --verbose

test:  ## Run tests
	pytest tests/ -v --cov=r4mc --cov-report=term-missing

test-fast:  ## Run fast tests only
	pytest tests/ -v -m "not slow"

lint:  ## Lint code
	ruff check r4mc/ || true

format:  ## Format code
	ruff format r4mc/ || true

check: lint test  ## Run all checks (lint + test)

clean-demo:  ## Remove demo outputs
	rm -rf $(OUT_DIR)/demo_report
	rm -rf tests/fixtures/demo_run

clean:  ## Remove generated files
	rm -rf $(OUT_DIR)
	rm -rf demo_output
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned"

clean-all: clean  ## Remove all generated files including caches
	rm -rf .ruff_cache
	rm -rf *.egg-info
	rm -rf dist build
	rm -rf tests/fixtures/demo_run
	@echo "✓ Deep clean complete"

.DEFAULT_GOAL := help
