# Evaluation Implementation Plan

## Overview

Add evaluation capability that reads MLflow traces, generates judges from task configs, and logs scores back to runs. No separate curation phase.

## Architecture Decisions

- **Guidelines format**: Simple list of strings in task YAML
- **Judge composition**: Declared in task YAML, resolved via judges registry
- **Judge model**: Configurable via env var (default: `gpt-4o-mini` via OpenAI/OpenRouter compatible API)
- **Results storage**: Log metrics back to existing simulation runs
- **Trace handling**: Extract summary statistics instead of passing full traces to judges

## File Changes Required

### 1. Update Pydantic Models

**File**: `src/beyond_vibes/simulations/models.py`

Add `guidelines` and `judges` fields to `SimulationConfig`:

```python
class SimulationConfig(BaseModel):
    name: str
    description: str
    archetype: str
    repository: RepositoryConfig
    prompt: str
    agent: str = "build"
    system_prompt: str | None = None
    max_turns: int = 50
    capture_git_diff: bool = False
    guidelines: list[str] = []  # NEW - success criteria for task-level judges
    judges: list[str] = []      # NEW - list of judge names from registry
```

### 2. Update Task YAML Files

**Files**:

- `src/beyond_vibes/simulations/prompts/tasks/poetry_to_uv.yaml`
- `src/beyond_vibes/simulations/prompts/tasks/unit_tests.yaml`

Add guidelines and judges sections to `poetry_to_uv.yaml`:

```yaml
name: "poetry_to_uv"
description: "Migrate a project from poetry to uv"
archetype: "repo_maintenance"
guidelines:
  - "The pyproject.toml must not contain [tool.poetry] sections"
  - "The poetry.lock file must be removed or replaced with uv.lock"
  - "The project must be installable with `uv sync`"
judges:
  - guidelines
  - tool_efficiency
repository:
  url: "https://github.com/blake-hamm/lighthearted"
  branch: "main"
agent: "build"
max_turns: 25
capture_git_diff: true
system_prompt: |
  You are running on NixOS. Never ask for human feedback or confirmation.
  Just proceed with the best approach and explain what you did afterward.
prompt: |
  You are an expert DevOps engineer. Migrate this project from poetry to uv.
  ...
```

### 3. Create Judges Registry

**File**: `src/beyond_vibes/evaluators/judges.yaml`

**Purpose**: Central registry of available judges with their configurations

```yaml
# Judges Registry
# Defines all available judges that can be referenced in task YAML files

judges:
  # Built-in MLflow judges
  guidelines:
    type: mlflow.builtin
    class: Guidelines
    description: "Evaluates if response meets task-specific criteria"
    
  tool_efficiency:
    type: mlflow.builtin
    class: ToolCallEfficiency
    description: "Detects redundant tool calls and loops"
    
  # Third-party DeepEval judges (curated subset)
  faithfulness:
    type: mlflow.third_party
    module: mlflow.metrics.deepeval
    class: Faithfulness
    description: "Measures if response is faithful to context"
    
  answer_relevancy:
    type: mlflow.third_party
    module: mlflow.metrics.deepeval
    class: AnswerRelevancy
    description: "Measures if response is relevant to the question"
    
  contextual_precision:
    type: mlflow.third_party
    module: mlflow.metrics.deepeval
    class: ContextualPrecision
    description: "Measures precision of retrieved context"
    
  contextual_recall:
    type: mlflow.third_party
    module: mlflow.metrics.deepeval
    class: ContextualRecall
    description: "Measures recall of relevant context"
    
  contextual_relevancy:
    type: mlflow.third_party
    module: mlflow.metrics.deepeval
    class: ContextualRelevancy
    description: "Measures relevancy of retrieved context"

# Future: Custom judges can be added here
# custom_judge_name:
#   type: custom
#   factory: beyond_vibes.evaluators.custom:create_custom_judge
```

### 4. Create Evaluators Module Structure

**New directory structure**:

```
src/beyond_vibes/evaluators/
├── __init__.py              # Public API exports
├── models.py                # Pydantic models (JudgeInput, EvalResult)
├── extractor.py             # MLflow run data extraction
├── judge_factory.py         # Judge creation from registry
├── runner.py                # Main evaluation orchestration
└── judges.yaml              # Judge registry configuration
```

### 5. Extractor Module

**File**: `src/beyond_vibes/evaluators/extractor.py`

**Purpose**: Query MLflow runs and extract data for judges

```python
@dataclass
class JudgeInput:
    """Standardized input for all judges."""
    run_id: str
    task_name: str
    archetype: str
    system_prompt: str
    task_prompt: str
    final_message: str
    git_diff: str | None
    # Trace summaries instead of full raw trace
    trace_summary: dict  # Pre-computed metrics for judges
    
@dataclass
class TraceSummary:
    """Summary statistics extracted from MLflow trace."""
    total_messages: int
    total_tool_calls: int
    tool_error_count: int
    total_tokens: int
    tool_loop_detected: bool  # Same tool called >3 times consecutively
    error_rate: float  # tool_error_count / total_tool_calls
    token_efficiency: float  # tokens per tool call
    # Sample of message indices with errors (for analysis)
    error_message_indices: list[int]

def extract_run_data(run_id: str) -> JudgeInput:
    """Extract standardized input from MLflow run."""
    # Load run and artifacts
    # Extract final message from last span
    # Compute trace summary from spans (not full trace)
    pass

def compute_trace_summary(trace: mlflow.entities.Trace) -> TraceSummary:
    """Compute summary statistics from trace without storing full trace."""
    # Analyze spans for:
    # - Tool call patterns (detect loops)
    # - Error counts
    # - Token usage
    # Return summary only
    pass

def query_simulation_runs(
    experiment: str = "beyond-vibes",
    run_ids: list[str] | None = None,
    task_name: str | None = None,
    archetype: str | None = None,
) -> list[mlflow.entities.Run]:
    """Query runs matching filters."""
    pass
```

### 6. Judge Factory Module

**File**: `src/beyond_vibes/evaluators/judge_factory.py`

**Purpose**: Create judges from registry based on task configuration

```python
import yaml
from pathlib import Path
from typing import Any

# Load judges registry
JUDGES_REGISTRY_PATH = Path(__file__).parent / "judges.yaml"

with open(JUDGES_REGISTRY_PATH) as f:
    _registry = yaml.safe_load(f)

JUDGE_MODEL = "openai:/gpt-4o-mini"  # Default, overridable via env

def create_judge(judge_name: str, task_config: SimulationConfig | None = None) -> Any:
    """Create a judge instance from registry by name."""
    judge_config = _registry["judges"].get(judge_name)
    if not judge_config:
        raise ValueError(f"Unknown judge: {judge_name}")
    
    judge_type = judge_config["type"]
    
    if judge_type == "mlflow.builtin":
        return _create_mlflow_builtin_judge(judge_config, task_config)
    elif judge_type == "mlflow.third_party":
        return _create_mlflow_third_party_judge(judge_config, task_config)
    elif judge_type == "custom":
        return _create_custom_judge(judge_config, task_config)
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")

def _create_mlflow_builtin_judge(config: dict, task_config: SimulationConfig | None) -> Any:
    """Create MLflow built-in judge."""
    from mlflow.metrics import Guidelines, ToolCallEfficiency
    
    judge_class = config["class"]
    
    if judge_class == "Guidelines":
        if not task_config or not task_config.guidelines:
            raise ValueError("Guidelines judge requires task guidelines")
        return Guidelines(
            name=f"{task_config.name}_guidelines",
            guidelines=task_config.guidelines,
            model=JUDGE_MODEL,
        )
    elif judge_class == "ToolCallEfficiency":
        return ToolCallEfficiency(model=JUDGE_MODEL)
    else:
        raise ValueError(f"Unknown built-in judge: {judge_class}")

def _create_mlflow_third_party_judge(config: dict, task_config: SimulationConfig | None) -> Any:
    """Create MLflow third-party judge (e.g., DeepEval)."""
    import importlib
    
    module_path = config["module"]
    class_name = config["class"]
    
    module = importlib.import_module(module_path)
    judge_class = getattr(module, class_name)
    
    return judge_class(model=JUDGE_MODEL)

def _create_custom_judge(config: dict, task_config: SimulationConfig | None) -> Any:
    """Create custom judge from factory function."""
    # Reserved for future custom judge implementations
    raise NotImplementedError("Custom judges not yet implemented")

def build_judges_for_task(task_config: SimulationConfig) -> list:
    """Build all judges specified in task config."""
    judges = []
    
    for judge_name in task_config.judges:
        try:
            judge = create_judge(judge_name, task_config)
            if judge:
                judges.append(judge)
        except ValueError as e:
            logger.warning(f"Skipping judge '{judge_name}': {e}")
    
    return judges
```

### 7. Runner Module

**File**: `src/beyond_vibes/evaluators/runner.py`

**Purpose**: Orchestrate evaluation pipeline

```python
class EvaluationRunner:
    """Runs judges on simulation runs and logs results."""
    
    def __init__(self, judge_model: str | None = None):
        self.judge_model = judge_model or settings.judge_model
    
    def evaluate_run(
        self, 
        run_id: str,
        judges: list | None = None,
    ) -> dict:
        """Evaluate a single run with specified judges."""
        # Extract data
        judge_input = extract_run_data(run_id)
        
        # Get task config (from run tags/artifacts)
        task_config = self._load_task_config(judge_input.task_name)
        
        # Build judges if not provided
        if judges is None:
            judges = build_judges_for_task(task_config)
        
        if not judges:
            logger.warning(f"No judges configured for run {run_id}")
            return {}
        
        # Run evaluation
        eval_data = self._prepare_eval_data(judge_input)
        results = mlflow.evaluate(
            data=eval_data,
            model=self.judge_model,
            judges=judges,
        )
        
        # Log results back to run
        self._log_results_to_run(run_id, results)
        
        return results
    
    def _prepare_eval_data(self, judge_input: JudgeInput) -> dict:
        """Convert JudgeInput to format expected by mlflow.evaluate."""
        return {
            "inputs": {
                "request": judge_input.task_prompt,
                "system": judge_input.system_prompt,
            },
            "outputs": {
                "response": judge_input.final_message,
                "git_diff": judge_input.git_diff,
            },
            "trace_summary": judge_input.trace_summary,  # For efficiency judges
        }
    
    def _log_results_to_run(self, run_id: str, results) -> None:
        """Log evaluation scores back to the simulation run."""
        with mlflow.start_run(run_id=run_id):
            for result in results:
                metric_name = f"eval_{result.name}"
                mlflow.log_metric(metric_name, result.score)
                # Also log rationale as tag if available
                if result.rationale:
                    mlflow.set_tag(f"{metric_name}_rationale", result.rationale[:500])
    
    def _load_task_config(self, task_name: str) -> SimulationConfig:
        """Load task config by name."""
        from beyond_vibes.simulations.prompts.loader import load_task_config
        return load_task_config(task_name, "{}")
```

### 8. CLI Command

**File**: `src/beyond_vibes/cli.py` (addition)

```python
@app.command()
def evaluate(
    run_id: str | None = typer.Option(
        None, 
        "--run-id", 
        help="Specific run to evaluate (if omitted, evaluates all matching filters)"
    ),
    task: str | None = typer.Option(
        None, 
        "--task", 
        help="Filter by task name"
    ),
    archetype: str | None = typer.Option(
        None, 
        "--archetype", 
        help="Filter by archetype"
    ),
    experiment: str = typer.Option(
        "beyond-vibes", 
        "--experiment", 
        help="MLflow experiment name"
    ),
) -> None:
    """Evaluate simulation runs using configured judges."""
    from beyond_vibes.evaluators.runner import EvaluationRunner
    from beyond_vibes.evaluators.extractor import query_simulation_runs
    
    runner = EvaluationRunner()
    
    if run_id:
        # Single run evaluation
        results = runner.evaluate_run(run_id)
        logger.info(f"Evaluated run {run_id}: {results}")
    else:
        # Batch evaluation
        runs = query_simulation_runs(
            experiment=experiment,
            task_name=task,
            archetype=archetype,
        )
        success_count = 0
        for run in runs:
            try:
                results = runner.evaluate_run(run.info.run_id)
                logger.info(f"Evaluated run {run.info.run_id}")
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to evaluate run {run.info.run_id}: {e}")
        
        logger.info(f"Successfully evaluated {success_count}/{len(runs)} runs")
```

### 9. Settings Update

**File**: `src/beyond_vibes/settings.py`

Add judge model configuration:

```python
class Settings(BaseSettings):
    # ... existing settings ...
    judge_model: str = "openai:/gpt-4o-mini"
    judge_api_key: str | None = None  # For OpenRouter or custom endpoints
    judge_base_url: str | None = None  # For OpenAI-compatible APIs
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Task YAML (poetry_to_uv.yaml)                                │
│  ├── guidelines: [criterion_1, criterion_2, ...]               │
│  └── judges: [guidelines, tool_efficiency, ...]             │
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
│  │   └── git_diff.patch                                     │
│  └── Spans: message_0, message_1, ..., message_N            │
│       └── TOOL spans with inputs/outputs                   │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Extractor (extractor.py)                                     │
│  └── JudgeInput:                                             │
│      ├── task_prompt, system_prompt                        │
│      ├── final_message (last span output)                  │
│      ├── git_diff                                          │
│      └── trace_summary (computed metrics)                  │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Judge Factory (judge_factory.py)                             │
│  ├── Resolve judge names from registry                     │
│  ├── Create Guidelines from task config                    │
│  └── Instantiate DeepEval/third-party judges               │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  mlflow.evaluate()                                            │
│  ├── Judges score: pass/fail + rationale                   │
│  └── Returns: EvaluationResult objects                     │
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
export BV_JUDGE_MODEL="openai:/gpt-4o"

# For OpenRouter
export BV_JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export BV_JUDGE_API_KEY="sk-or-..."

# For custom OpenAI-compatible endpoint
export BV_JUDGE_BASE_URL="http://localhost:8000/v1"
```

### Settings.py

```python
class Settings(BaseSettings):
    # ... existing settings ...
    judge_model: str = "openai:/gpt-4o-mini"
    judge_api_key: str | None = None
    judge_base_url: str | None = None
```

## Dependencies to Add

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "mlflow[genai]>=3.10.0",  # MLflow with genai extras
    "openai>=1.0.0",  # Required for judges
    "deepeval>=0.21.0",  # DeepEval metrics for MLflow
]
```

## Testing Strategy

1. **Unit tests** for extractor (mock MLflow runs)
2. **Unit tests** for judge factory (verify registry loading)
3. **Integration test** with real run ID from existing traces
4. **Validation**: Verify guidelines from poetry_to_uv.yaml create correct Guidelines judge
5. **Validation**: Verify DeepEval judges load and execute correctly

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

1. Create judge implementation in `src/beyond_vibes/evaluators/custom/`
2. Add entry to `judges.yaml`:
```yaml
custom_judge:
  type: custom
  factory: beyond_vibes.evaluators.custom:create_my_judge
  description: "Custom evaluation logic"
```

## Success Criteria

- [ ] Can run `beyond-vibes evaluate --run-id <id>` and get scores logged
- [ ] Guidelines from task YAML are automatically converted to Guidelines judge
- [ ] ToolCallEfficiency works using trace summary (detects loops without full trace)
- [ ] DeepEval judges (faithfulness, answer_relevancy) execute correctly
- [ ] Results visible in MLflow UI with scores + rationale
- [ ] New judge can be added by updating `judges.yaml` only (no code changes for built-in/third-party)
- [ ] Custom judges can be added with minimal code changes
