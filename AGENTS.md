# Beyond Vibes — Agent Guidelines

Model evaluation benchmark for local agentic coding models.

## Commands
uv sync                         # Install deps
uv run pytest                   # Run tests
uv run pytest tests/test_x.py   # Single file

# Lint + format (use nix develop on NixOS):
nix develop -c ruff check --fix . && nix develop -c ruff format .
nix develop -c mypy src/        # Type check

## Project Layout
src/beyond_vibes/evaluations/   # LLM judge logic (DSPy-based)
src/beyond_vibes/model_config.py # Model configs
src/beyond_vibes/model_downloader/ # HF / S3 download helpers
src/beyond_vibes/simulations/   # Simulation engine (pi.dev client, orchestration, MLflow tracing)
src/beyond_vibes/simulations/prompts/tasks/ # Task prompt YAMLs
data/             # Golden datasets (gitignored)
tests/

## Key Concepts
- **Archetypes**: Four task categories — Architectural Planning, Repo Maintenance, Feature Implementation, Comparative Research
- **Golden dataset**: Human-validated input/output pairs in `data/golden/`
- **LLM judge**: DSPy-optimized multi-metric scorer in `src/beyond_vibes/evaluations/`
- **Metrics**: Universal (applies to all) + Category-specific (per-archetype)
- **Tasks**: Defined in `src/beyond_vibes/simulations/prompts/tasks/`, run via pi.dev

## Git Rules
- Feature branches: `feature/description`
- Never commit: `.env`, `data/`, `__pycache__/`
- Run tests + ruff before pushing

## Engineering Philosophy
*Follow the zen of python as much as possible:*
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

### Docstrings

Keep docstrings brief and descriptive. Well-typed signatures eliminate the need for verbose Args/Returns sections. Avoid copying types into docstrings — this creates drift and redundant state. One-line docstrings are preferred unless explaining non-obvious behavior.

### TDD Workflow (Optional)

When making code changes in `src/`, prefer Test Driven Development:

1. **Red** — Write a failing test first, run it to confirm failure
2. **Green** — Get human approval, then write minimal code to pass
3. **Blue** — Refactor as needed

**When to skip TDD:** Prototypes, exploratory refactors, research tasks, docs, quick fixes, or any non-`src/` changes.

**Approval checkpoint:** Before any implementation code in `src/`, get human approval.
