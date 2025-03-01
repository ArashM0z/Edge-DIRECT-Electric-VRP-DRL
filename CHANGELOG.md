# Changelog

All notable changes to this fork are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Full CI/CD pipeline (lint, multi-Python pytest, training-loop smoke, security scan)
- Tagged-release workflow with SBOM + GitHub release notes
- Dependabot + Renovate for dependency updates
- Pre-commit hooks (ruff, gitleaks, actionlint, conventional-commits)
- Repository hygiene: PR template, issue templates, CODEOWNERS
- Makefile with `setup`, `test`, `lint`, `smoke`, `train`, `eval`, `ablate`
- SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md
- Renovate config and editorconfig

## [1.0.0] — 2025-05-12 — TKDD camera-ready

### Added

- Simultaneous encoder with temporal-locality inductive bias (§4.2)
- Vehicle Selection Decoder (§4.3.1) and Trip Construction Decoder (§4.3.2)
- SED2AM agent emitting 2-tuple actions `(vehicle, location)`
- MTTDVRP problem with per-interval edge travel-time tensor and τ_max
- Full state with five §3.3 transition rules
- Training entrypoint with REINFORCE + rollout baseline (Kool 2019 Algorithm 1)
- Evaluation script against AM, OR-Tools, NN baselines (§5.3)
- Ablation harness for temporal-locality / dual-decoder / max-hours (§5.6)
- Per-problem-size hyperparameter configs (n20 / n50 / n100)
- Test suite for encoder math, state transitions, dual-decoder masks, rollout, runtime
