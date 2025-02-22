.PHONY: help setup test lint format typecheck smoke train eval ablate pre-commit clean docs

PY ?= python
COVERAGE_THRESHOLD ?= 70

help:  ## show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup:  ## install dev dependencies
	uv pip install -e ".[dev]"
	pre-commit install --install-hooks
	pre-commit install --hook-type commit-msg

test:  ## run the SED2AM test suite
	pytest tests -q --cov=. --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD)

lint:  ## ruff + mypy on SED2AM modules
	ruff check nets/sed2am_*.py problems/mttdvrp tests train_sed2am.py eval_sed2am.py ablation_sed2am.py
	mypy --strict nets/sed2am_*.py problems/mttdvrp || true

format:  ## auto-format with ruff
	ruff format nets/sed2am_*.py problems/mttdvrp tests train_sed2am.py eval_sed2am.py ablation_sed2am.py
	ruff check --fix nets/sed2am_*.py problems/mttdvrp tests train_sed2am.py eval_sed2am.py ablation_sed2am.py

typecheck:  ## strict mypy
	mypy --strict nets/sed2am_*.py problems/mttdvrp

smoke:  ## quick 2-epoch training smoke test
	$(PY) train_sed2am.py --graph-size 5 --n-vehicles 2 --n-intervals 5 \
		--epochs 2 --iters 5 --batch-size 4 --val-size 8 --out /tmp/runs/smoke

train:  ## full training (CONFIG=configs/n50.yaml by default)
	CONFIG=$${CONFIG:-configs/n50.yaml}; \
	$(PY) train_sed2am.py \
		--graph-size $$($(PY) -c "import yaml; print(yaml.safe_load(open('$$CONFIG'))['graph_size'])") \
		--n-vehicles $$($(PY) -c "import yaml; print(yaml.safe_load(open('$$CONFIG'))['n_vehicles'])") \
		--epochs $$($(PY) -c "import yaml; print(yaml.safe_load(open('$$CONFIG'))['epochs'])") \
		--iters $$($(PY) -c "import yaml; print(yaml.safe_load(open('$$CONFIG'))['iters'])") \
		--batch-size $$($(PY) -c "import yaml; print(yaml.safe_load(open('$$CONFIG'))['batch_size'])") \
		--lr $$($(PY) -c "import yaml; print(yaml.safe_load(open('$$CONFIG'))['lr'])")

eval:  ## evaluate a checkpoint (CKPT=runs/sed2am/best.pt)
	$(PY) eval_sed2am.py --checkpoint $${CKPT:-runs/sed2am/best.pt}

ablate:  ## run the §5.6 ablation harness
	$(PY) ablation_sed2am.py

pre-commit:  ## run all pre-commit hooks against the worktree
	pre-commit run --all-files

clean:  ## remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ \
	       **/__pycache__ build dist *.egg-info coverage.xml .coverage

docs:  ## generate / serve the local docs
	@echo "Inline README.md is the primary doc surface; consider mkdocs if this grows."
