# Contributing to raghilda

## Setup

Create and activate the development environment:

```bash
uv sync --all-groups --extra test
source .venv/bin/activate
```

## Running checks

Before submitting changes, run the full check suite:

```bash
uv run task check
```

This runs formatting, linting, type checking, and tests. You can also run
individual checks:

```bash
uv run task format_check   # check formatting
uv run task lint_check     # check linting
uv run task types_check    # check types
uv run task tests          # run tests
```

## Releasing

Releases are published to PyPI automatically via GitHub Actions when a GitHub
Release is created.

### Steps

1. Update the `version` in `pyproject.toml`.
2. Commit the version bump and merge to `main`.
3. Go to the repository's
   [Releases](https://github.com/posit-dev/raghilda/releases) page.
4. Click **Draft a new release**.
5. Create a new tag matching the version (e.g., `v0.2.0`).
6. Add release notes describing the changes.
7. Click **Publish release**.

The `release.yml` workflow will build the package and publish it to PyPI using
[trusted publishing](https://docs.pypi.org/trusted-publishers/).
