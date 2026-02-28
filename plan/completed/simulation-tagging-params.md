# Migration: Tags to Parameters

## Overview

Convert categorical configuration values from MLflow tags to parameters for better performance analysis and comparison capabilities.

## Changes

### What to Log as Parameters

Move these from `mlflow.set_tag()` to `mlflow.log_param()`:

```python
# Model configuration - affects performance
mlflow.log_param("model.name", model_config.name)          # e.g., "qwen3-8b"
mlflow.log_param("model.quant", self.quant_tag)            # e.g., "Q4_K_M"
mlflow.log_param("model.repo_id", model_config.repo_id)    # e.g., "Qwen/Qwen3-8B"

# Runtime configuration - affects performance
mlflow.log_param("runtime.container", container_tag)       # e.g., "rocm-6.2"
```

### What Remains as Tags

Keep these as `mlflow.set_tag()` for filtering:

```python
# Grouping identifiers
mlflow.set_tag("eval.batch_id", batch_id)                  # e.g., "eval-2024-01-15-143052"
mlflow.set_tag("model.config_id", config_id)               # e.g., "qwen3-8b-Q4_K_M-rocm"
mlflow.set_tag("session.id", session_id)                   # e.g., "qwen3-8b_Q4_K_M_20240115_143052"

# Task metadata - for filtering only, doesn't affect performance
mlflow.set_tag("task.name", sim_config.name)               # e.g., "poetry_to_uv"
mlflow.set_tag("task.archetype", sim_config.archetype)     # e.g., "repo_maintenance"
mlflow.set_tag("repository.url", sim_config.repository.url)
mlflow.set_tag("repository.branch", sim_config.repository.branch)

# Status flags
mlflow.set_tag("status.has_error", "true")                 # Only set on failure
mlflow.set_tag("status.judge_complete", "true")            # Set after LLM-as-judge
```

## Implementation

### File: `src/beyond_vibes/simulations/mlflow.py`

Replace in `log_simulation()` context manager:

```python
# OLD - These move to params (performance-related)
mlflow.set_tag("model.name", model_config.name)            # → mlflow.log_param()
mlflow.set_tag("model.repo_id", model_config.repo_id)      # → mlflow.log_param()
mlflow.set_tag("model.quant_tag", self.quant_tag)          # → mlflow.log_param()

# OLD - These stay as tags (filtering/grouping)
mlflow.set_tag("simulation.task", sim_config.name)         # → task.name tag
mlflow.set_tag("simulation.archetype", sim_config.archetype)  # → task.archetype tag
mlflow.set_tag("simulation.repo_url", sim_config.repository.url)  # → repository.url tag

# NEW - Parameters (performance analysis)
mlflow.log_param("model.name", model_config.name)
mlflow.log_param("model.repo_id", model_config.repo_id)
mlflow.log_param("model.quant", self.quant_tag)
mlflow.log_param("runtime.container", container_tag)  # Pass via constructor

# NEW - Tags (filtering)
mlflow.set_tag("task.name", sim_config.name)
mlflow.set_tag("task.archetype", sim_config.archetype)
mlflow.set_tag("repository.url", sim_config.repository.url)
mlflow.set_tag("repository.branch", sim_config.repository.branch)
```

## Benefits

1. **MLflow Compare View**: Use built-in parameter comparison to see how model/quant/container affect metrics
2. **Correlation Analysis**: Query parameters to correlate configuration with performance
3. **Filtering + Analysis**: Tags for UI filtering, parameters for quantitative analysis
4. **Reproducibility**: Parameters capture the exact configuration for each run

## Validation

After implementation, verify in MLflow UI:
- [ ] Parameters section shows all config values
- [ ] Tags section shows only grouping/identifier tags  
- [ ] Compare view can filter by parameters
- [ ] Existing dashboards/queries still work (update if needed)
