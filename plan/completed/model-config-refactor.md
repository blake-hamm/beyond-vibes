# Model Config Refactor Plan

## Overview
Refactor model configuration to support both local (downloadable from HuggingFace) and remote (API-only) models. This enables testing with various providers while maintaining a single source of truth in models.yaml.

## Goals
- Single models.yaml defines all supported models regardless of provider
- Automatic download for local provider, skip for remote
- CLI supports filtering by provider and model name
- Model_id field allows testing same model from different providers
- Shared model loading logic to avoid code duplication

## Design Decisions

### Provider Logic
- **local**: Requires `repo_id`, triggers HF download to S3
- **non-local** (e.g., openai, anthropic): No `repo_id`, API-only
- Validation: Option A - local provider requires repo_id (fail fast)

### Default Values
- `provider`: "local"
- `model_id`: falls back to `name` if not specified
- `repo_id`: None (optional, required only for local)

### Directory Restructure
Move ModelConfig schema and loading logic to `src/beyond_vibes/model_config.py` for shared access across:
- `cli.py` (both download and simulate commands)
- `model_downloader/` modules
- `simulations/` modules

## Implementation Tasks

### Phase 1: Create Shared Models Module
**New File:** `src/beyond_vibes/model_config.py`

Move from `src/beyond_vibes/model_downloader/models.py`:
- `ESSENTIAL_MODEL_CONFIGS` constant
- `ModelConfig` class with updates:
  - `repo_id: str | None = None` (make optional)
  - `provider: str = "local"`
  - `model_id: str | None = None`
  - Add Pydantic validator: if provider == "local", require repo_id
- `Config` class (root config model)

Add new loading function:
```python
def load_models_config(path: Path | None = None) -> Config:
    """Load and validate models.yaml."""
```

### Phase 2: Update model_downloader Module
**File:** `src/beyond_vibes/model_downloader/__init__.py`

Changes:
- Re-export ModelConfig, Config from `beyond_vibes.models`
- Remove local models.py file entirely

**File:** `src/beyond_vibes/model_downloader/hf.py` and `s3.py`

Changes:
- Update imports to use `beyond_vibes.models`

### Phase 3: Download Command
**File:** `src/beyond_vibes/cli.py`

Changes:
- Import from `beyond_vibes.models` instead of model_downloader
- Skip models where `repo_id is None` during download
- Add logging for skipped models
- Add `--provider` CLI flag for filtering

### Phase 4: MLflow Logging
**File:** `src/beyond_vibes/simulations/mlflow.py`

Changes:
- Import from `beyond_vibes.models`
- Log `model.provider`
- Log `model.model_id` (use name as fallback)
- Handle `repo_id` being None

### Phase 5: OpenCode Integration
**File:** `src/beyond_vibes/simulations/orchestration.py`

Changes:
- Import from `beyond_vibes.models`
- Use `model_config.model_id or model_config.name` for OpenCode API

### Phase 6: CLI Filtering
**File:** `src/beyond_vibes/cli.py`

Changes:
- Add `--provider` option to simulate command
- Support filtering: if both --model and --provider specified, use both
- If only --provider, run all models matching that provider
- Use shared `load_models_config()` function

### Phase 7: Update Tests
**File:** `tests/test_models.py`

Changes:
- Update imports to use `beyond_vibes.models`
- Update existing tests for optional repo_id
- Fix validation: local requires repo_id (existing tests may need adjustment)

**File:** `tests/test_hf.py`, `tests/test_cli.py`

Changes:
- Update imports to use `beyond_vibes.models`

## Example Configuration

```yaml
models:
  # Local models - will be downloaded
  - name: qwen3-0.6B
    repo_id: unsloth/Qwen3-0.6B-GGUF
    provider: local
    model_id: qwen3-0.6b
    quant_tags:
      - Q6_K_XL
      - Q8_K_XL

  # API-only models - no download
  - name: gpt-4o
    provider: openai
    model_id: gpt-4o

  - name: claude-sonnet-4
    provider: anthropic
    model_id: claude-sonnet-4-20250514
```

## Usage Examples

```bash
# Download only local models
uv run beyond-vibes download

# Run specific local model
uv run beyond-vibes simulate --task test-repo --model qwen3-0.6B --quant Q6_K_XL

# Run API model
uv run beyond-vibes simulate --task test-repo --model gpt-4o

# Run all models from a provider
uv run beyond-vibes simulate --task test-repo --provider openai

# Run specific model with explicit provider check
uv run beyond-vibes simulate --task test-repo --model gpt-4o --provider openai
```

## Testing Strategy

- Update existing tests to import from new location
- Ensure existing tests pass with updated ModelConfig schema
- No new test expansion (per requirements)

## Files to Modify

### New Files
- `src/beyond_vibes/model_config.py` (new shared module)

### Modified Files
- `src/beyond_vibes/model_downloader/models.py` (delete - moved to shared)
- `src/beyond_vibes/model_downloader/__init__.py` (re-export from shared)
- `src/beyond_vibes/model_downloader/hf.py` (update imports)
- `src/beyond_vibes/model_downloader/s3.py` (update imports if needed)
- `src/beyond_vibes/cli.py` (use shared loader, add --provider flag)
- `src/beyond_vibes/simulations/mlflow.py` (update imports, log new fields)
- `src/beyond_vibes/simulations/orchestration.py` (update imports, use model_id)
- `tests/test_models.py` (update imports, fix for optional repo_id)
- `tests/test_hf.py` (update imports)
- `tests/test_cli.py` (update imports)
- `models.yaml` (update example configs)

### Import Changes
```python
# Old imports (to be replaced)
from beyond_vibes.model_downloader import Config, ModelConfig
from beyond_vibes.model_downloader.models import ESSENTIAL_MODEL_CONFIGS

# New imports
from beyond_vibes.models import Config, ModelConfig, ESSENTIAL_MODEL_CONFIGS
from beyond_vibes.models import load_models_config
```

## Migration Notes

- Existing configs with `repo_id` and no `provider` will continue working (defaults to local)
- No breaking changes to existing behavior
- New fields are optional with sensible defaults
- Import paths change but functionality remains the same
