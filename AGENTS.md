# Beyond Vibes — Agent Guidelines

Model evaluation benchmark for local agentic coding models.

## Commands
uv sync                         # Install deps
uv run pytest                   # Run tests
uv run pytest tests/test_x.py   # Single file
uv run ruff check --fix . && uv run ruff format .  # Lint+format
uv run mypy src/                # Type check

## Project Layout
src/evaluators/   # LLM judge logic (DSPy-based)
src/models/       # Model configs and wrappers
configs/          # Task and benchmark configs
data/             # Golden datasets (gitignored)
tests/

## Key Concepts
- **Archetypes**: Four task categories — Architectural Planning, Repo Maintenance, Feature Implementation, Comparative Research
- **Golden dataset**: Human-validated input/output pairs in `data/golden/`
- **LLM judge**: DSPy-optimized multi-metric scorer in `src/evaluators/`
- **Metrics**: Universal (applies to all) + Category-specific (per-archetype)
- **Tasks**: Defined in `configs/tasks/`, run via OpenCode

## Git Rules
- Feature branches: `feature/description`
- Never commit: `.env`, `data/`, `__pycache__/`
- Run tests + ruff before pushing
