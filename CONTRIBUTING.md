# Contributing to nano-agent

Thanks for your interest in nano-agent. This is an experimental R&D project that
builds an autonomous agent for playing a game from screenshots. The notes below
describe how to set up the project, the conventions we follow, and how to submit
changes.

## Development setup

This project uses [uv](https://github.com/astral-sh/uv) with `pyproject.toml` as
the single source of truth for dependencies. Python 3.10 to 3.13 is supported.

```bash
# Install uv if needed
pip install uv

# CPU-only environment
uv sync --extra cpu

# CUDA 11.8 environment
uv sync --extra cu118
```

Run the code through uv so it uses the synced environment:

```bash
uv run python main.py
uv run python state_sim/demo.py
```

See `README.md` for training and evaluation commands.

## Running tests

Tests live under `tests/`. Run them with the synced environment:

```bash
uv run python -m pytest tests/
```

Some tests require the machine-learning dependencies (for example PyTorch), so run
them inside a synced environment rather than a bare interpreter.

## Branch and pull request conventions

- Create a feature branch off `main`; do not commit directly to `main`.
- Keep each pull request focused on one logical change.
- Write commit messages in the imperative mood with a short type prefix, for example
  `docs: ...`, `feat: ...`, `fix: ...`, `refactor: ...`, `chore: ...`.
- Update `CHANGELOG.md` under the `Unreleased` section when your change is user visible.
- Make sure the CodeQL analysis checks pass before requesting a merge.

## Code style

- Follow the style of the surrounding code; match its naming and structure.
- Keep functions small and give variables descriptive names.
- Add comments only where the intent is not obvious from the code.

## Documentation language

The internal design documents under `docs/` are written in Vietnamese. Community
files such as this one, `README.md`, and `CHANGELOG.md` are kept in English so that
outside contributors can read them.

## Reporting issues

Use GitHub issues for bugs and feature requests. For security matters, follow the
process in `SECURITY.md` instead of opening a public issue.
