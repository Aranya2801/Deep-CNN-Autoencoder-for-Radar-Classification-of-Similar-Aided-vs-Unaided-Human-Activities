# Contributing to Radar-HAR

Thanks for considering a contribution! This project welcomes issues, PRs,
new datasets/loaders, model improvements, and documentation fixes.

## Getting started

```bash
git clone https://github.com/your-org/deep-cnn-autoencoder-radar-har.git
cd deep-cnn-autoencoder-radar-har

# fast path -- no torch needed, runs in ~90s
pip install -r reference_pipeline/requirements.txt
python reference_pipeline/run_all.py

# full path -- production PyTorch model
pip install -r requirements.txt
python -m radar_har.train
```

## Project layout

See the root `README.md` "Repository structure" section. In short:
`src/radar_har/` is the production package, `reference_pipeline/` is the
dependency-light demo, `backend/` is the FastAPI service, `frontend/` has
both a zero-build static dashboard and a React/TS scaffold.

## Development workflow

1. Fork and branch from `main`: `git checkout -b feat/your-feature`
2. Make your change. Add or update tests in `tests/`.
3. Run the relevant test subset:
   ```bash
   pytest tests/test_simulation.py tests/test_reference_pipeline.py  # always
   pytest tests/test_models.py tests/test_api.py                     # if torch/fastapi installed
   ```
4. Lint and format:
   ```bash
   ruff check . --fix
   black .
   ```
5. Open a PR against `main`. CI (`.github/workflows/ci.yml`) runs lint +
   tests + a full reference-pipeline run automatically.

## Code style

- Python: `black` (110 char line length) + `ruff`. Type hints required on
  new public functions. Docstrings explain *why*, not just *what* -- see
  any existing module for the house style (a short module-level docstring
  explaining the design rationale, not just an API listing).
- TypeScript: keep components small and typed; avoid `any`.

## Adding a new dataset loader

Implement a function returning `(X: np.ndarray[N,H,W], y: np.ndarray[N],
class_names: list[str])` -- the same contract as
`radar_har.simulation.micro_doppler_simulator.generate_dataset` -- and wire
it into `radar_har.data.dataset.RadarSpectrogramDataset`. Add the dataset to
`docs/dataset_guide.md` with a verified source link and license.

## Adding a new model component

New encoder/decoder/head architectures go in `src/radar_har/models/`.
Update `docs/architecture.md` with the rationale (what problem does this
component solve, and why this approach over alternatives) -- "why," not
just "what," is the standard the rest of the codebase holds itself to.

## Reporting bugs / requesting features

Open a GitHub issue with:
- What you expected vs. what happened
- Minimal repro steps (ideally against the reference pipeline, which needs
  no GPU/torch to reproduce)
- Python version, OS, and `pip freeze` output if dependency-related

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you agree to uphold it.
