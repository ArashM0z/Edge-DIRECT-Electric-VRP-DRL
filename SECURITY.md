# Security Policy

## Reporting a Vulnerability

Open a [private security advisory](../../security/advisories/new) on this repository.

You will receive an acknowledgement within 72 hours. Coordinated disclosure preferred.

## Supported Versions

The `master` branch is the only actively-supported branch. Tagged releases receive security patches for 6 months after publication.

## Threat Model

This is a research codebase, not a production system. The main attack surfaces:

- **Supply chain** — third-party dependencies. Updated weekly via Dependabot. CI runs `trivy` on every PR.
- **Secret leakage** — no secrets should be committed. `gitleaks` runs in pre-commit and CI.
- **Reproducibility** — model checkpoints are not signed. Always verify hashes when using a published checkpoint.

## Hardening

- `pyproject.toml` pins major versions; minor / patch updates flow through CI.
- CI matrix runs on Python 3.10–3.12 to surface deprecation issues early.
- Branch protection on `master` requires green CI, signed commits, and one review.
