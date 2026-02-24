# Simulation Implementation Plan

## Overview

Implement CLI command(s) for running simulations that clone repos into a temporary sandbox, execute prompts via OpenCode Python SDK, and log all session data to MLflow for evaluation.

## Requirements Summary

- **OpenCode integration**: Python SDK (`opencode-ai`) - requires OpenCode server running on configurable URL (default: http://localhost:54321, set via `OPENCODE_URL` env var), configurable provider (default: llamacpp, set via `OPENCODE_PROVIDER` env var)
- **Prompts storage**: `src/beyond_vibes/simulations/prompts/` (MLflow 3.10 compatible)
- **Sandbox**: Temporary directory using Python's `tempfile`
- **MLflow**: From environment (`MLFLOW_TRACKING_URI`), capture all metrics for evals
- **Scope**: Single simulation focus now, extensible for multiple later

### MLflow 3.10 Features

This implementation leverages MLflow 3.10 features:
- **Multi-turn evaluation & conversation simulation** - for evaluating full agent sessions
- **Trace cost tracking** - automatically captures LLM costs from spans
- **Session-level scorers** - for archetype-specific evaluations
- **In-UI trace evaluation** - run judges from MLflow UI

---

## Architecture

### Directory Structure

```
src/beyond_vibes/
├── cli.py                      # Add `simulate` command
└── simulations/
    ├── __init__.py
    ├── prompts/
    │   ├── __init__.py
    │   ├── loader.py           # Load + render prompts with {{var}} syntax
    │   └── tasks/
    │       └── <archetype>/
    │           └── <task_name>.yaml
    ├── sandbox.py              # Temp dir + git clone
    ├── runner.py               # Execute via OpenCode SDK
    ├── config.py               # Pydantic models
    ├── logging.py              # MLflow tracing integration
    └── opencode_client.py      # Wrapper around opencode-ai SDK
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

1. **Create prompt module** (`src/beyond_vibes/simulations/prompts/`)
   - `loader.py` - Load YAML prompts, render `{{variable}}` substitutions

2. **Create simulation module** (`src/beyond_vibes/simulations/`)
   - `sandbox.py` - `SandboxManager`:
     - `tempfile.mkdtemp()` for workspace
     - `clone_repo(url, branch)` - Git clone using GitPython
     - Context manager for automatic cleanup
     - **Error handling**: Try/except around git clone for network/auth failures
   - `config.py` - Pydantic models: `SimulationConfig`, `RepositoryConfig`

3. **Create OpenCode wrapper** (`src/beyond_vibes/opencode_client.py`)
   - Wrap `opencode-ai` SDK (REST client - requires OpenCode server running, URL configurable via `OPENCODE_URL` env var, default: http://localhost:54321; provider configurable via `OPENCODE_PROVIDER` env var, default: llamacpp)
   - `create_session(working_dir)` - Initialize with sandbox path
   - `run_prompt(session_id, prompt)` - Execute prompt, returns response with content/parts
   - **Error handling**: Wrap API calls in try/except for connection errors (server not running)

4. **Create MLflow logger** (`src/beyond_vibes/simulations/logging.py`)
   - `SimulationLogger` class with context manager
   - **Real-time approach**: Capture raw session data for later DSPy eval processing:
     - **Params**: task_name, archetype, repo_url, branch
     - **Metrics**: total_turns, total_time, tps, ttft, token_count
     - **Artifacts**: Full conversation as JSON, git diff (if applicable), prompt, final response
   - Use MLflow traces/spans for multi-turn conversation (MLflow 3.10 feature)
   - **Error handling**: Wrap MLflow logging calls in try/except to prevent simulation failures from crashing on logging errors

   **Note**: Eval metrics (sycophancy, specificity, pass/fail, etc.) are computed post-process by DSPy judges on historical MLflow data. Real-time logging captures raw data needed for this.

5. **Add CLI command** (`src/beyond_vibes/cli.py`)
   - Add `simulate` subcommand
   - Options: `--task`, `--prompt-vars` (JSON), `--no-cleanup`
   - Flow: Load prompt → Clone repo → Run simulation → Log to MLflow
   - **Error handling**: Wrap each phase in try/except - log failures but continue to cleanup

### Phase 2: First Task

6. **Create first simulation task**
   - Choose "poetry to uv" (simplest from README.md)
   - Create `simulations/prompts/tasks/repo_maintenance/poetry_to_uv.yaml`

### Phase 3: Extensibility (Future)

7. **Multi-simulation support**
   - Add `batch` subcommand for stratified runs
   - Add stratification config (model, quantization, container)

### Other ideas:
#### Advanced Git Operations (Future)
 **Branch checkout and push for evals**
   - After simulation, commit changes to a new branch
   - Push to remote for evaluation/verification
   - Could be leveraged for human-in-the-loop evals or automated PR workflows
   - Note: Overkill for now, but valuable for later

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

# Keep sandbox for debugging (skip cleanup)
uv run beyond-vibes simulate --task auth_plan --no-cleanup

# Environment setup
export MLFLOW_TRACKING_URI=databricks
export OPENAI_API_KEY=...
export OPENCODE_URL=http://localhost:54321  # Optional, this is the default
export OPENCODE_PROVIDER=llamacpp  # Optional, this is the default

# Ensure OpenCode server is running (required)
opencode
```

## Dependencies to Add

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "gitpython>=3.1.0",
    "opencode-ai>=0.1.0",
    "mlflow>=3.0.0",
]
```

## Prerequisites

- OpenCode CLI installed and server running (`opencode` in background)
- MLflow tracking server configured via `MLFLOW_TRACKING_URI`
- Optional: Set `OPENCODE_URL` to configure server URL (default: http://localhost:54321)
- Optional: Set `OPENCODE_PROVIDER` to configure provider (default: llamacpp)

## Testing Strategy

1. Unit tests for `SandboxManager` (mock GitPython)
2. Unit tests for prompt loader (verify variable substitution)
3. Integration test with public repo
4. Test MLflow logging (mock MLflow client)
