# Runbook

Operational guide for running, testing, training, and deploying the Predictive Maintenance Platform.

---

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- Docker and Docker Compose (via [Docker Desktop](https://www.docker.com/products/docker-desktop/) or [Colima](https://github.com/abiosoft/colima) on macOS)
- [Docker Hub](https://hub.docker.com) account (for pushing images)

---

## Local Development Setup

```bash
git clone https://github.com/sherozshaikh/predictive-maintenance-platform.git
cd predictive-maintenance-platform

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run Tests

```bash
make test-local
```

### Train Models Locally

```bash
make train-local
```

### Start the API Locally (without Docker)

```bash
PYTHONPATH=. uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

---

## Docker Setup

Start Docker (Docker Desktop, or on macOS with Colima: `colima start --memory 4 --cpu 2`).

### Start Everything (API + Infra)

```bash
make up
```

This starts the API, Prefect Server, MLflow, MinIO, Prometheus, and Grafana.

### Service URLs

| URL | Service | Credentials |
|-----|---------|-------------|
| http://localhost:8000 | Dashboard + API | -- |
| http://localhost:8000/docs | Swagger API Docs | -- |
| http://localhost:4200 | Prefect Pipeline UI | -- |
| http://localhost:5000 | MLflow Experiment Tracking | -- |
| http://localhost:9001 | MinIO Object Storage | minioadmin / minioadmin |
| http://localhost:9090 | Prometheus Metrics | -- |
| http://localhost:3000 | Grafana Dashboards | admin / admin |

### Train Models via Docker

```bash
make train
```

The worker connects to the Prefect server. View the DAG and task states at http://localhost:4200.

### Run Tests in Docker

```bash
make test
```

### Stop Everything

```bash
make down        # stop containers
make down-clean  # stop + remove volumes
```

---

## Docker Hub

### Build and Push Images

```bash
docker login
make build
make push
```

### Pull Pre-built Images

```bash
make pull
docker compose --profile infra up -d
```

### Version Tags

```bash
docker tag sherozshaikh/predictive-maintenance-api:latest sherozshaikh/predictive-maintenance-api:1.1.0
docker push sherozshaikh/predictive-maintenance-api:1.1.0
```

---

## Deploy to GCP Free-Tier VM

See [docs/DEPLOY_GCP.md](docs/DEPLOY_GCP.md) for the full step-by-step guide.

Quick summary:

```bash
# On the VM (Ubuntu 22.04, e2-micro):
sudo apt-get update && sudo apt-get install -y docker.io
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
sudo docker pull sherozshaikh/predictive-maintenance-api:1.1.0
sudo docker run -d --name pm-api --restart unless-stopped -p 8000:8000 -e PYTHONPATH=/app sherozshaikh/predictive-maintenance-api:1.1.0
```

---

## Benchmark (LightGBM vs H2O AutoML)

```bash
uv pip install -e ".[dev,benchmark]"
make benchmark
```

Results are saved to `data/artifacts/benchmark_results.json`.

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make up` | Start all services (API + infra) |
| `make down` | Stop all containers |
| `make down-clean` | Stop + remove volumes |
| `make restart` | Restart all containers |
| `make logs` | Follow container logs |
| `make ps` | Show running containers |
| `make test` | Run tests in Docker |
| `make test-local` | Run tests locally |
| `make train` | Train models via Docker worker |
| `make train-local` | Train models locally |
| `make benchmark` | Run LightGBM vs H2O benchmark |
| `make build` | Build Docker images |
| `make push` | Build + push to Docker Hub |
| `make pull` | Pull images from Docker Hub |
| `make verify` | Health-check all services |
| `make format` | Format code (isort + black + ruff) |
| `make clean` | Remove generated files |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Models not loaded" | Pre-trained models should be in `data/artifacts/`. Retrain with `make train-local`. |
| Port in use | `lsof -ti:8000 \| xargs kill -9` |
| Docker build fails | `docker compose build --no-cache api` |
| Tests fail with import errors | Ensure `PYTHONPATH=.` is set, or use `make test-local`. |
| Prefect timeout in Docker | Ensure `make up` is running (Prefect server must be healthy). |
