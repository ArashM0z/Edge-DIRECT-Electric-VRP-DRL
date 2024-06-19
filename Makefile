.PHONY: setup test lint format
setup:
	pip install -e ".[dev]"

test:
	pytest tests/efectiw -q

lint:
	ruff check nets/efectiw problems/hf_vrptw tests/efectiw
	mypy --ignore-missing-imports nets/efectiw problems/hf_vrptw || true

format:
	ruff format nets/efectiw problems/hf_vrptw tests/efectiw

train:
	python run.py --problem hf_vrptw --graph_size 50 --baseline rollout --run_name efectiw-n50
