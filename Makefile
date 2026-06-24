.PHONY: help install install-dev demo train-reference eval-reference train evaluate api dashboard test lint format docker-up docker-down clean

help:
	@echo "Radar-HAR -- common commands"
	@echo "  make install          Install full production requirements (needs torch)"
	@echo "  make install-dev      Install reference-pipeline-only requirements (no torch needed)"
	@echo "  make demo             Run the full reference pipeline (simulate+train+eval, ~90s, CPU)"
	@echo "  make train            Train the production PyTorch model"
	@echo "  make evaluate         Evaluate the production PyTorch model"
	@echo "  make api              Run the FastAPI backend locally (uvicorn --reload)"
	@echo "  make dashboard        Open the static demo dashboard"
	@echo "  make test             Run the test suite"
	@echo "  make lint             Run ruff + black --check"
	@echo "  make format           Run black + ruff --fix"
	@echo "  make docker-up        docker compose up --build"
	@echo "  make docker-down      docker compose down"
	@echo "  make clean            Remove caches, __pycache__, build artifacts"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r reference_pipeline/requirements.txt -r backend/requirements.txt

demo:
	python reference_pipeline/run_all.py

train-reference:
	python reference_pipeline/train_reference_model.py

eval-reference:
	python reference_pipeline/evaluate_reference_model.py

train:
	python -m radar_har.train

evaluate:
	python -m radar_har.evaluate

api:
	uvicorn backend.app.main:app --reload --port 8000

dashboard:
	python -m http.server 8080 --directory frontend/static-demo

test:
	pytest -v --cov=src/radar_har tests/

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .coverage htmlcov
