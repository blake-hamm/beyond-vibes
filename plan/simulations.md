# Simulation Implementation Plan

## Overview

Implement CLI command(s) for running simulations that clone repos into a temporary sandbox, execute prompts via OpenCode Python SDK, and log all session data to MLflow for evaluation.

## Requirements Summary

- **OpenCode integration**: Python SDK (`opencode-ai`)
- **Prompts storage**: `src/beyond_vibes/prompts/` (MLflow 3.10 compatible)
- **Sandbox**: Temporary directory using Python's `tempfile`
- **MLflow**: From environment (`MLFLOW_TRACKING_URI`), capture all metrics for evals
- **Scope**: Single simulation focus now, extensible for multiple later

---

## Architecture

### Directory Structure

```
src/beyond_vibes/
├── cli.py                      # Add `simulate` command
├── prompts/
│   ├── __init__.py
│   ├── loader.py               # Load + render prompts with {{var}} syntax
│   └── tasks/
│       └── <archetype>/
│           └── <task_name>.yaml
├── simulation/
│   ├── __init__.py
│   ├── sandbox.py              # Temp dir + git clone
│   ├── runner.py              # Execute via OpenCode SDK
│   ├── config.py              # Pydantic models
│   └── logging.py             # MLflow tracing integration
└── opencode_client.py          # Wrapper around opencode-ai SDK
```

### Prompt Format (MLflow 3.10 Compatible)

YAML files with double-brace templating:

```yaml
name: "repo_maintenance_poetry_to_uv"
description: "Switch from poetry to uv for lighthearted"
archetype: "repo_maintenance"
repository:
  url: "https://github.com/blake-hamm/lighthearted"
  branch: "main"
prompt: |
  You are an expert DevOps engineer. Migrate this project from poetry to uv.
  
  Requirements:
  {{requirements}}
  
  Steps:
  1. Remove pyproject.toml if it exists
  2. Add pyproject.toml for uv
  3. Update CI/CD if applicable
```

---

## Implementation Steps

### Phase 1: Core Infrastructure

1. **Create prompt module** (`src/beyond_vibes/prompts/`)
   - `loader.py` - Load YAML prompts, render `{{variable}}` substitutions

2. **Create simulation module** (`src/beyond_vibes/simulation/`)
   - `sandbox.py` - `SandboxManager`:
     - `tempfile.mkdtemp()` for workspace
     - `clone_repo(url, branch)` - Git clone
     - Context manager for automatic cleanup
   - `config.py` - Pydantic models: `SimulationConfig`, `RepositoryConfig`

3. **Create OpenCode wrapper** (`src/beyond_vibes/opencode_client.py`)
   - Wrap `opencode-ai` SDK
   - `create_session(working_dir)` - Initialize with sandbox path
   - `run_prompt(session_id, prompt)` - Execute prompt
   - `get_messages(session_id)` - Retrieve conversation
   - `get_tool_calls(messages)` - Extract tool invocations

4. **Create MLflow logger** (`src/beyond_vibes/simulation/logging.py`)
   - `SimulationLogger` class with context manager
   - Log all metrics from README.md evals:
     - **Universal**: Turns/steps, tool calls, schema errors, loop detection, TPS, TTFT
     - **Architectural**: Specificity, security, constraints
     - **Repo Maintenance**: Pass/fail, diff stats
     - **Feature Implementation**: Completeness
     - **Comparative Research**: Citations, decisiveness
   - Log git diff as artifact

5. **Add CLI command** (`src/beyond_vibes/cli.py`)
   - Add `simulate` subcommand
   - Options: `--task`, `--prompt-vars` (JSON), `--keep-sandbox`
   - Flow: Load prompt → Clone repo → Run simulation → Log to MLflow

### Phase 2: First Task

6. **Create first simulation task**
   - Choose "poetry to uv" (simplest from README.md)
   - Create `prompts/tasks/repo_maintenance/poetry_to_uv.yaml`

### Phase 3: Extensibility (Future)

7. **Multi-simulation support**
   - Add `batch` subcommand for stratified runs
   - Add stratification config (model, quantization, container)

---

## Key Design Decisions

1. **Prompt loader abstraction**: Allows swapping between local YAML and MLflow registry without changing calling code
2. **Sandbox as context manager**: Ensures cleanup even on errors
3. **YAML over Python**: Prompts stored as YAML for version control, easier editing
4. **Separation of concerns**: Prompts, simulation, and CLI are separate modules

---

## CLI Usage (After Implementation)

```bash
# Run simulation
uv run beyond-vibes simulate --task poetry_to_uv

# With custom prompt variables
uv run beyond-vibes simulate --task auth_plan --prompt-vars '{"requirements": "OAuth2"}'

# Keep sandbox for debugging
uv run beyond-vibes simulate --task auth_plan --keep-sandbox

# Environment setup
export MLFLOW_TRACKING_URI=databricks
export OPENAI_API_KEY=...
```

## Dependencies to Add

```toml
[project.optional-dependencies]
simulate = [
    "opencode-ai>=0.1.0",
    "mlflow>=3.0.0",
]
```

## Testing Strategy

1. Unit tests for `SandboxManager` (mock git)
2. Unit tests for prompt loader (verify variable substitution)
3. Integration test with public repo
4. Test MLflow logging (mock MLflow client)
