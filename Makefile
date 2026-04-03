DOCKER_HUB_USER := sherozshaikh
PROJECT         := predictive-maintenance
API_IMAGE       := $(DOCKER_HUB_USER)/$(PROJECT)-api
WORKER_IMAGE    := $(DOCKER_HUB_USER)/$(PROJECT)-worker
TAG             := latest

# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------

## Start everything (API + infra: MLflow, MinIO, Prometheus, Grafana)
up:
	docker compose --profile infra up --build -d

## Stop all containers
down:
	docker compose --profile infra down

## Stop all containers and remove volumes (full wipe)
down-clean:
	docker compose --profile infra down -v

## Show running container logs (follow mode)
logs:
	docker compose --profile infra logs -f

## Show running containers
ps:
	docker compose --profile infra ps

## Restart all containers
restart: down up

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

## Run tests inside the running API container (65 expected)
test:
	docker compose exec api pytest tests/ -v

## Run tests locally (requires venv with deps installed)
test-local:
	PYTHONPATH=. pytest tests/ -v

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

## Retrain models via Docker worker (requires `make up` running for Prefect server)
train:
	docker compose --profile infra --profile train run --rm worker

## Retrain models locally (skip MLflow)
train-local:
	PYTHONPATH=. python scripts/run_flow.py --models forecast failure --skip-mlflow

## Benchmark LightGBM vs H2O AutoML (requires: uv pip install -e ".[benchmark]")
benchmark:
	PYTHONPATH=. python scripts/benchmark_h2o.py

# ---------------------------------------------------------------------------
# Docker Hub (push pre-built images)
# ---------------------------------------------------------------------------

## Build images for Docker Hub
build:
	docker buildx build --platform linux/amd64 -t $(API_IMAGE):$(TAG) -f docker/Dockerfile.api --load .
	docker buildx build --platform linux/amd64 -t $(WORKER_IMAGE):$(TAG) -f docker/Dockerfile.worker --load .

## Push images to Docker Hub (run `docker login` first)
push: build
	docker push $(API_IMAGE):$(TAG)
	docker push $(WORKER_IMAGE):$(TAG)

## Pull pre-built images from Docker Hub
pull:
	docker pull $(API_IMAGE):$(TAG)
	docker pull $(WORKER_IMAGE):$(TAG)

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

## Format code
format:
	isort . && black . && ruff check --fix . && ruff format .

## Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete
	find . -name ".DS_Store" -type f -delete
	find . -type d -name ".vscode" -exec rm -rf {} +
	find . -type d -name ".idea" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Verify services
# ---------------------------------------------------------------------------

## Check all service health endpoints
verify:
	@echo "--- API ---"
	@curl -sf http://localhost:8000/v1/health | python3 -m json.tool || echo "API not reachable"
	@echo "\n--- MLflow ---"
	@curl -sf http://localhost:5000/health || echo "MLflow not reachable"
	@echo "\n--- Prometheus ---"
	@curl -sf http://localhost:9090/-/healthy || echo "Prometheus not reachable"
	@echo "\n--- MinIO ---"
	@curl -sf http://localhost:9000/minio/health/live || echo "MinIO not reachable"
	@echo "\n--- Grafana ---"
	@curl -sf http://localhost:3000/api/health || echo "Grafana not reachable"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

## Show this help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  make up          Start everything (API + MLflow + MinIO + Prometheus + Grafana)"
	@echo "  make down        Stop all containers"
	@echo "  make down-clean  Stop all containers and remove volumes"
	@echo "  make restart     Restart all containers"
	@echo "  make logs        Follow container logs"
	@echo "  make ps          Show running containers"
	@echo ""
	@echo "  make test        Run tests inside API container (65 expected)"
	@echo "  make test-local  Run tests locally"
	@echo ""
	@echo "  make train       Retrain models via Docker worker"
	@echo "  make train-local Retrain models locally (skip MLflow)"
	@echo ""
	@echo "  make build       Build images for Docker Hub"
	@echo "  make push        Build and push images to Docker Hub"
	@echo "  make pull        Pull pre-built images from Docker Hub"
	@echo ""
	@echo "  make verify      Check all service health endpoints"
	@echo "  make format      Format code (isort + black + ruff)"
	@echo "  make clean       Remove generated files"

.PHONY: up down down-clean logs ps restart test test-local train train-local \
        build push pull format clean verify help
