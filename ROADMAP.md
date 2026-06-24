# Roadmap

## v1.0 (current)

- [x] Physics-inspired 14-class radar simulator
- [x] Production multi-task architecture (residual CNN + CBAM + autoencoder
      + classifier/confidence/contrastive heads)
- [x] Dependency-light reference pipeline with real, reproducible results
- [x] FastAPI backend with prediction/training/registry/monitoring APIs
- [x] React dashboard scaffold + zero-build static demo dashboard
- [x] Fall-detection daily-use application
- [x] ONNX edge export + federated learning simulation
- [x] MLOps scaffolding (Docker, DVC, MLflow, CI/CD)
- [x] Full documentation suite

## v1.1 (next)

- [ ] Train and publish the production PyTorch model's actual checkpoint +
      benchmark numbers (blocked on a torch-enabled training environment --
      tracked in `docs/results.md`)
- [ ] Real-dataset integration: a verified loader for at least one of the
      openly-downloadable datasets in `docs/dataset_guide.md` (RadHAR is the
      best first target -- openly cloneable, no registration wall)
- [ ] Replace the React dashboard's demo random-noise predictions with a
      live WebSocket radar feed connector
- [ ] Add `pip-audit` / `npm audit` / Trivy image scanning to CI
- [ ] Add a proper job queue (Celery/RQ) behind `/train` instead of the
      current in-process `BackgroundTasks` (fine for demo, not for scale)

## v1.5

- [ ] Domain-adaptation experiment: pretrain on simulated data, fine-tune
      on a small real-data sample, report transfer-learning gains
- [ ] SHAP/LIME explainability backends as alternatives to Grad-CAM/occlusion
      (the brief's full XAI list) -- currently Grad-CAM (gradient-based) and
      occlusion (gradient-free) are implemented; SHAP/LIME are natural next
      additions on the same `embedding`/`logits` interface
- [ ] UMAP as a first-class latent-space view (currently falls back to
      t-SNE if `umap-learn` isn't installed -- works, but a bundled
      dependency would make it the default)
- [ ] Multi-person / multi-target radar scenes (current scope is single-
      person activity recognition)

## v2.0 (exploratory)

- [ ] Cross-radar-platform generalization study (10 GHz CW lab radar ->
      77 GHz automotive FMCW), exercising the `replace_classifier` /
      domain-adaptation hooks already built into `DeepCNNAutoencoderHAR`
- [ ] On-device (true edge, not just ONNX-exported-but-still-server-run)
      deployment validation on a Raspberry Pi / Jetson Nano with a real CW
      radar front-end
- [ ] Publish a short technical report comparing the reference pipeline's
      Gabor+PCA+MLP results against the trained production model, as a
      worked example of "classical features vs. learned features" for the
      same task

## Out of scope (for now)

- Multi-modal fusion with camera/LiDAR (radar-only by design, for the
  privacy and lighting/weather-invariance properties that make radar
  attractive for in-home elderly monitoring in the first place)
- Real-time streaming inference benchmarks at scale (the architecture
  supports it; rigorous latency/throughput benchmarking on target hardware
  is a follow-up, not a v1 claim)

Contributions toward any of the above are welcome -- see
`CONTRIBUTING.md`.
