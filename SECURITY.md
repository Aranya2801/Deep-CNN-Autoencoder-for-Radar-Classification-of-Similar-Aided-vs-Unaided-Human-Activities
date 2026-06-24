# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.x (main) | ✅ |
| < 1.0 (pre-release) | ❌ |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead:

1. Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
   feature on this repository (Security tab -> "Report a vulnerability"), or
2. Email the maintainers directly (see repository contact info) with:
   - A description of the vulnerability and its potential impact
   - Steps to reproduce
   - Any suggested fix, if you have one

We aim to acknowledge reports within 5 business days and to ship a fix or
mitigation within 30 days for confirmed high/critical issues.

## Scope

This is a research/portfolio project, not a hardened production system as
shipped. Specific areas where security hardening is the deploying team's
responsibility, not handled by default in this repo:

- **Authentication/authorization**: the FastAPI backend ships with **no
  auth** by default (`backend/app/main.py`). Add an auth layer (OAuth2/JWT,
  API keys, or a gateway) before exposing `/train` or `/predict/batch` to
  untrusted networks -- `/train` in particular executes a subprocess and
  should never be reachable from the public internet without auth.
- **Input validation**: spectrogram inputs are shape/range validated
  (`backend/app/api/routes_predict.py`), but extremely large batch sizes
  are only soft-capped via `MAX_BATCH_SIZE`; tune for your deployment.
- **Model/data privacy**: if deploying the fall-detection service
  (`deployment/fall_detection/`) with real radar captures of people's
  homes, ensure compliance with applicable health/privacy regulations
  (e.g. HIPAA in the US) -- this repository provides the technical
  capability, not legal compliance.
- **Dependency scanning**: run `pip-audit` / `npm audit` / Docker image
  scanning (e.g. Trivy) in your own CI before deploying to production; this
  repo's `ci.yml` does not include a dependency-vulnerability scan step by
  default.

## Known Limitations (not vulnerabilities, but worth knowing)

- The synthetic data simulator (`src/radar_har/simulation/`) is for
  development/demo only -- a model trained purely on it should not be
  trusted for real safety-critical fall detection without validation on
  real radar captures first.
