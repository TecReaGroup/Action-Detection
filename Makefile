.PHONY: install train run

install:
	uv sync

train:
	uv run action-detection train

run:
	uv run action-detection run
