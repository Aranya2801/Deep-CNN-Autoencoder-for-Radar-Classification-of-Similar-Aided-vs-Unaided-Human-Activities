# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versioning
follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-21

### Added

- Initial public release.
- Physics-inspired micro-Doppler radar simulator (`src/radar_har/simulation/`)
  covering 14 activity classes (6 unaided, 8 aided/similar).
- Production PyTorch architecture: residual CNN encoder + CBAM attention +
  temporal attention pooling + convolutional autoencoder + multi-task
  classifier/confidence/projection heads (`src/radar_har/models/`).
- Multi-task loss (classification + reconstruction + confidence), NT-Xent
  contrastive loss for self-supervised pretraining (`src/radar_har/losses.py`).
- Grad-CAM and occlusion-based explainability (`src/radar_har/explain/`).
- Latent-space analytics: PCA, t-SNE, UMAP-optional, clustering, similarity
  search, MMD-based drift detection (`src/radar_har/explain/latent_space.py`).
- Dependency-light reference pipeline (`reference_pipeline/`) -- Gabor
  filter-bank features + PCA + MLPClassifier -- that runs end-to-end on CPU
  with zero extra installs, producing the real metrics in `docs/results.md`.
- FastAPI backend (`backend/`) with prediction, batch prediction, training
  trigger, model registry, similarity search, drift monitoring, and health
  endpoints, with graceful fallback between the production and reference
  model backends.
- React + TypeScript dashboard scaffold (`frontend/dashboard-react/`) and a
  zero-build static HTML demo dashboard (`frontend/static-demo/`).
- Daily-use applications: elderly fall-detection / activity-trend monitoring
  service (`deployment/fall_detection/`), ONNX edge export, federated
  learning simulation (`deployment/edge/`), Kubernetes manifests
  (`deployment/k8s/`).
- MLOps scaffolding: Dockerfiles, `docker-compose.yml`, DVC pipeline
  (`dvc.yaml`, `params.yaml`), MLflow tracking integration, GitHub Actions
  CI/CD (`.github/workflows/`).
- Full documentation suite: architecture, dataset guide (with verified real
  dataset citations), results, API reference, deployment guide.
- Test suite (`tests/`): simulator + reference-pipeline tests runnable with
  zero extra dependencies; model + API tests that auto-skip without
  torch/fastapi installed.

### Notes

- The production PyTorch model has not been trained in this repository's
  authoring environment (no network access to install `torch`); the
  reference pipeline's real, reproducible results stand in until you run
  `python -m radar_har.train` yourself. See `reference_pipeline/README.md`.

## [Unreleased]

See `ROADMAP.md` for planned work.
