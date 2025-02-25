# Contributing

Thanks for the interest. This is a paper-implementation repo; contributions, issues, and questions are welcome.

## Development setup

```bash
git clone https://github.com/ArashM0z/SED2AM-Deep-RL-Time-Dependent-VRP.git
cd SED2AM-Deep-RL-Time-Dependent-VRP
make setup
```

## Working agreement

- **Conventional Commits** — `feat:`, `fix:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, `build:`, `ci:`, `chore:`. Enforced by pre-commit.
- **No upstream Kool changes without justification.** The point of the fork is to be readable as `Kool 2019 + SED2AM additions`. If you must change an upstream file, document the why in the PR.
- **Tests required.** Every code change needs a test.
- **Type hints.** All SED2AM modules pass `mypy --strict`.
- **Docs.** README updated when behaviour changes.

## Style

- Python: `ruff` for lint + format (configured in `pyproject.toml`).
- Typing: `mypy --strict` on SED2AM modules only (upstream files exempted).
- Markdown: hard wrap at 100 chars except for tables and code blocks.

## Branching

Branches off `master` named `<type>/<short-kebab>`. PRs target `master`.

## Releasing

Releases are automated by the `Release` workflow on git tags `v*`. Each release publishes a wheel + SBOM.

## License

By contributing you agree your contributions are licensed under MIT, matching the repo.
