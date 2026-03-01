# Simulation Implementation Plan

## Overview

Implement CLI command(s) for running simulations that clone repos into a temporary sandbox, execute prompts via OpenCode, and log all session data to MLflow for evaluation.

## Requirements Summary

- **OpenCode integration**: Direct HTTP calls via httpx (SDK was unmaintained) - requires OpenCode server running on configurable URL (default: http://127.0.0.1:4096, set via `OPENCODE_URL` env var), configurable provider (default: opencode, set via `OPENCODE_PROVIDER` env var)
- **Model selection**: Models defined in `models.yaml`, selected via `--model` CLI argument
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
    │       └── <task_name>.yaml
    ├── sandbox.py              # Temp dir + git clone
    ├── config.py               # Pydantic models
    └── logging.py              # MLflow tracing integration
```

Note: `opencode_client.py` is in `src/beyond_vibes/` (not in simulations/) as a standalone module.

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

3. **Create OpenCode client** (`src/beyond_vibes/opencode_client.py`)
   - Direct HTTP calls via httpx (replaced unmaintained opencode-ai SDK)
   - `create_session(working_dir)` - Initialize session (no init call - repos already have AGENTS.md)
   - `run_prompt(session_id, prompt, model_id)` - Execute prompt, returns response with content/parts
   - **Error handling**: httpx exceptions propagate naturally

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
   - Options: `--task`, `--model` (required), `--config-path`, `--prompt-vars` (JSON)
   - Flow: Load model config → Load prompt → Clone repo → Run simulation → Log to MLflow
   - **Error handling**: Wrap each phase in try/except - log failures but continue to cleanup

### Phase 2: First Task

6. **Create first simulation task**
   - Choose "poetry to uv" (simplest from README.md)
   - Create `simulations/prompts/tasks/poetry_to_uv.yaml`

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
5. **httpx over SDK**: Replaced unmaintained opencode-ai SDK with direct HTTP calls for better control and reliability
6. **models.yaml as source of truth**: Model configuration lives in models.yaml, selected via CLI

---

## CLI Usage (After Implementation)

```bash
# Run simulation with model from models.yaml
uv run beyond-vibes simulate --task poetry_to_uv --model minimax-m2.5-free

# With custom config (different models.yaml)
uv run beyond-vibes simulate --task poetry_to_uv --model qwen3-0.6B --config-path mymodels.yaml

# With custom prompt variables
uv run beyond-vibes simulate --task auth_plan --model minimax-m2.5-free --prompt-vars '{"requirements": "OAuth2"}'

# Environment setup
export MLFLOW_TRACKING_URI=https://mlflow.bhamm-lab.com
export OPENCODE_URL=http://127.0.0.1:4096  # Optional, this is the default
export OPENCODE_PROVIDER=opencode  # Optional, this is the default

# Ensure OpenCode server is running (required)
opencode
```

## models.yaml Format

```yaml
bucket: beyond-vibes
models:
  - name: minimax-m2.5-free
    repo_id: opencode/minimax-m2.5-free
    quant_tags: []
  - name: qwen3-0.6B
    repo_id: unsloth/Qwen3-0.6B-GGUF
    quant_tags:
      - Q6_K_XL
      - Q8_K_XL
      - BF16
```

## Dependencies

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "gitpython>=3.1.0",
    "httpx>=0.27.0",
    "mlflow>=3.0.0",
]
```

Note: Removed `opencode-ai` SDK dependency - replaced with httpx.

## Prerequisites

- OpenCode CLI installed and server running (`opencode serve` in background)
- MLflow tracking server configured via `MLFLOW_TRACKING_URI`
- Optional: Set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` if MLflow requires auth
- Optional: Set `OPENCODE_URL` to configure server URL (default: http://127.0.0.1:4096)
- Optional: Set `OPENCODE_PROVIDER` to configure provider (default: opencode)

## Testing Strategy

1. Unit tests for `SandboxManager` (mock GitPython)
2. Unit tests for prompt loader (verify variable substitution)
3. Integration test with public repo
4. Test MLflow logging (mock MLflow client)
