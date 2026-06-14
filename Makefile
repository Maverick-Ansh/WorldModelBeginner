.PHONY: install install-dev test lint format quickstart tb clean

install:        ## install the package (core deps)
	pip install -e .

install-dev:    ## install with dev + mujoco extras
	pip install -e ".[dev,mujoco]"

test:           ## run the test suite
	pytest -q

lint:           ## static checks
	ruff check .

format:         ## auto-format
	ruff format .

quickstart:     ## brief CPU demo of both agents on the toy hand
	bash scripts/quickstart.sh

tb:             ## launch TensorBoard
	tensorboard --logdir logs

clean:          ## remove caches and run artifacts
	rm -rf logs runs outputs .pytest_cache .ruff_cache **/__pycache__ __pycache__
