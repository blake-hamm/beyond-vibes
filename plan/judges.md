# Evaluation Implementation Plan

## Overview

Add evaluation capability that reads MLflow traces, generates judges from task configs, and logs scores back to runs. No separate curation phase.

## Architecture Decisions

- **Guidelines format**: Simple list of strings in task YAML
- **Judge composition**: Declared in task YAML with explicit input mapping
- **Judge model**: Configurable via env var (default: `gpt-4o-mini` via OpenAI-compatible API)
- **Results storage**: Log metrics back to existing simulation runs
- **Trace handling**: Full trace metadata available in `trace_session.json` artifact
- **Artifact routing**: Judges explicitly declare which artifact they evaluate (git_diff, final_message, trace)

## Implementation Phases

This plan is split into 7 phases for incremental implementation:

1. **Phase 1** - Trace artifact (`trace_session.json`) ✅ *Already complete*
2. **Phase 2** - Core data models and settings
3. **Phase 3** - Evaluators module structure
4. **Phase 4** - Extractor module
5. **Phase 5** - Judge factory module
6. **Phase 6** - Runner module
7. **Phase 7** - CLI command and task YAML updates

See individual phase files for detailed implementation:
- `judge-phase-2.md` - Core Data Models and Settings
- `judge-phase-3.md` - Evaluators Module Structure
- `judge-phase-4.md` - Extractor Module
- `judge-phase-5.md` - Judge Factory Module
- `judge-phase-6.md` - Runner Module
- `judge-phase-7.md` - CLI Command and Task YAML Updates

## File Changes Summary

### 1. Trace Artifact

**File**: `src/beyond_vibes/simulations/mlflow.py` (already implemented)

The simulation logs full session metadata as `trace_session.json` artifact.

### 2-10. Implementation Details

See individual phase files for complete implementation details:

- **Phase 2**: `judge-phase-2.md` - Core Data Models and Settings
  - `SimulationConfig` with `JudgeMapping`
  - `judges.yaml` registry
  - Settings for judge configuration

- **Phase 3**: `judge-phase-3.md` - Evaluators Module Structure
  - Directory structure
  - `__init__.py` with exports

- **Phase 4**: `judge-phase-4.md` - Extractor Module
  - `JudgeInput` dataclass
  - `extract_run_data()` function
  - `query_simulation_runs()` function

- **Phase 5**: `judge-phase-5.md` - Judge Factory Module
  - `create_judge()` function
  - `build_judges_for_task()` function
  - Registry loading

- **Phase 6**: `judge-phase-6.md` - Runner Module
  - `EvaluationRunner` class
  - Result logging

- **Phase 7**: `judge-phase-7.md` - CLI Command and Task YAML Updates
  - `evaluate` CLI command
  - Task YAML examples

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Task YAML (poetry_to_uv.yaml)                                │
│  ├── guidelines: [criterion_1, criterion_2, ...]               │
│  └── judges:                                                │
│      - {name: guidelines, input: git_diff}                   │
│      - {name: tool_efficiency, input: trace}                 │
│      - {name: faithfulness, input: git_diff}                 │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Judges Registry (judges.yaml)                                │
│  ├── guidelines → mlflow.builtin.Guidelines                  │
│  ├── tool_efficiency → mlflow.builtin.ToolCallEfficiency   │
│  └── faithfulness → mlflow.third_party.deepeval.Faithfulness│
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  MLflow Run (from simulation)                                 │
│  ├── Tags: task.name=poetry_to_uv                            │
│  ├── Artifacts:                                              │
│  │   ├── system_prompt.txt                                   │
│  │   ├── git_diff.patch                                     │
│  │   └── trace_session.json (full session metadata)        │
│  └── Metrics: total_tool_calls, tool_loop_detected, etc.    │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Extractor (extractor.py)                                     │
│  └── JudgeInput:                                             │
│      ├── task_prompt, system_prompt                        │
│      ├── final_message                                     │
│      ├── git_diff                                          │
│      └── trace (full metadata from trace_session.json)    │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Judge Factory (judge_factory.py)                             │
│  ├── Resolve judge names from registry                     │
│  ├── Create Guidelines from task config                    │
│  ├── Instantiate DeepEval/third-party judges               │
│  └── Return (judge, input_artifact) tuples                 │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Runner (runner.py)                                           │
│  ├── For each (judge, input):                              │
│  │   ├── Prepare eval_data with appropriate artifact       │
│  │   ├── Run mlflow.evaluate()                             │
│  │   └── Log results to run                                │
│  └── DeepEval judges: git_diff as context/output           │
│      task_prompt as input                                  │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Results Logged to Original Run                               │
│  ├── eval_guidelines: 1.0                                    │
│  ├── eval_tool_efficiency: 0.85                            │
│  ├── eval_faithfulness: 0.92                               │
│  └── eval_guidelines_rationale: "All criteria met..."       │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# Override default judge model
export JUDGE_MODEL="openai:/gpt-4o"

# For OpenRouter or litellm proxy
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export JUDGE_API_KEY="sk-or-..."

# For local OpenAI-compatible endpoint (litellm, vLLM, etc.)
export JUDGE_BASE_URL="http://localhost:8000/v1"
export JUDGE_API_KEY="dummy-key"  # If required by endpoint
```

### Settings.py

```python
class Settings(BaseSettings):
    # ... existing settings ...
    judge_model: str = "openai:/gpt-4o-mini"
    judge_api_key: str | None = None
    judge_base_url: str | None = None  # Enables local models via litellm
```

## Dependencies to Add

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "mlflow[genai]>=3.10.0",  # MLflow with genai extras
    "openai>=1.0.0",  # Required for judges
    "deepeval>=0.21.0",  # DeepEval metrics for MLflow (POC)
]
```

## Testing Strategy

1. **Unit tests** for extractor (mock MLflow runs)
2. **Unit tests** for judge factory (verify registry loading and input mapping)
3. **Integration test** with real run ID from existing traces
4. **Validation**: Verify guidelines from poetry_to_uv.yaml create correct Guidelines judge
5. **Validation**: Verify DeepEval judges load and execute correctly with git_diff as context
6. **Test local model support**: Verify JUDGE_BASE_URL works with litellm

## Adding New Judges

### Adding a Built-in MLflow Judge

1. Add entry to `judges.yaml`:
```yaml
new_builtin:
  type: mlflow.builtin
  class: NewBuiltinClass
  description: "Description of what it evaluates"
```

2. Update `_create_mlflow_builtin_judge()` in `judge_factory.py` to handle the new class

### Adding a DeepEval Judge

1. Add entry to `judges.yaml`:
```yaml
new_deepeval_metric:
  type: mlflow.third_party
  module: mlflow.metrics.deepeval
  class: NewDeepEvalClass
  description: "Description of metric"
```

No code changes needed - factory dynamically imports from MLflow.

### Adding a Custom Judge (Future)

1. Create judge implementation in `src/beyond_vibes/evaluations/custom/`
2. Add entry to `judges.yaml`:
```yaml
custom_judge:
  type: custom
  factory: beyond_vibes.evaluations.custom:create_my_judge
  description: "Custom evaluation logic"
```

## Success Criteria

- [ ] Can run `beyond-vibes evaluate --run-id <id>` and get scores logged
- [ ] Guidelines from task YAML are automatically converted to Guidelines judge
- [ ] ToolCallEfficiency evaluates using trace data (pre-computed + LLM assessment)
- [ ] DeepEval judges execute correctly using git_diff as context
- [ ] Results visible in MLflow UI with scores + rationale
- [ ] New judge can be added by updating `judges.yaml` only (no code changes for built-in/third-party)
- [ ] Custom judges can be added with minimal code changes
- [ ] Local models work via JUDGE_BASE_URL (OpenAI-compatible endpoints)
- [ ] Input mapping works: judges evaluate correct artifacts per task config

## Key Changes Summary

1. **trace_session.json** - Contains full session metadata
2. **Explicit Input Mapping** - Task YAML specifies which artifact each judge evaluates
3. **OpenAI-Compatible Endpoints** - Supports litellm, vLLM, and other local models via JUDGE_BASE_URL
4. **DeepEval POC** - Using git_diff as context/output and task_prompt as input
5. **Artifact Routing** - Runner prepares eval_data based on judge's input mapping
