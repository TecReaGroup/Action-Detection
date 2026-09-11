.PHONY: install train run app

install:
	uv sync

train:
	uv run action-detection train

run:
	uv run action-detection run

app:
	uv run python -m app.main
