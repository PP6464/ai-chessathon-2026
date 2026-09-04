SHELL := /bin/bash

.PHONY: setup play arena zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy

play-win:
	uv run python -m harness_win.play --white . --black baselines/greedy

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

arena-win:
	uv run python -m harness_win.arena --opponent baselines/greedy --games 20

train:
	uv run python -m training.train

export-model-files:
	uv run python -m model.export

zip:
	uv run python -m harness.package

zip-win:
	uv run python -m harness_win.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
