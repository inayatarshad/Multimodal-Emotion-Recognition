.PHONY: help install data train eval experiments serve web test lint typecheck format \
        check figures report clean clean-all docker

DATASET ?= mosi
MODEL   ?= mult
SEED    ?= 0
PRESET  ?= smoke
UV      ?= uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the environment and install everything
	$(UV) sync --extra serve --extra viz
	$(UV) run pre-commit install

data:  ## Build the aligned feature cache and freeze the split
	$(UV) run wfb-data --dataset $(DATASET)

train:  ## Train one architecture (MODEL=mult DATASET=mosi SEED=0)
	$(UV) run wfb-train model=$(MODEL) data=$(DATASET) seed=$(SEED)

eval:  ## Sweep one trained checkpoint over the corruption grid
	$(UV) run wfb-eval model=$(MODEL) data=$(DATASET) seed=$(SEED)

experiments:  ## Run a full experiment preset (PRESET=smoke|dev|main|cross|mitigation)
	$(UV) run python experiments/run_all.py --preset $(PRESET)

figures:  ## Regenerate the paper figures from the committed results
	$(UV) run python -c "from wfb.reporting.figures import generate_all; \
		[print('wrote', p) for p in generate_all()]"

report:  ## Regenerate the results tables and the README headline
	$(UV) run python -c "from pathlib import Path; \
		from wfb.serving.results_store import ResultsStore; \
		from wfb.reporting.tables import full_report, headline_table, update_readme; \
		s = ResultsStore.load('experiments/results'); \
		Path('experiments/results/REPORT.md').write_text(full_report(s), encoding='utf-8'); \
		update_readme(Path('README.md'), headline_table(s)); \
		print(headline_table(s))"

serve:  ## Run the API at http://localhost:8000 (docs at /docs)
	$(UV) run uvicorn wfb.serving.app:app --reload --port 8000

web:  ## Run the frontend dev server at http://localhost:5173
	cd web && npm install && npm run dev

test:  ## Run the test suite
	$(UV) run pytest -q

lint:  ## Lint with ruff
	$(UV) run ruff check src tests experiments

typecheck:  ## Type-check src with mypy --strict
	$(UV) run mypy --strict src

format:  ## Auto-format and auto-fix
	$(UV) run ruff format src tests experiments
	$(UV) run ruff check --fix src tests experiments

check: lint typecheck test  ## Everything CI runs

docker:  ## Build and start the API container
	docker compose up --build

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-all: clean  ## Also remove checkpoints, outputs and the feature cache
	rm -rf outputs lightning_logs data/processed
