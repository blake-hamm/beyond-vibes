# Evaluation Implementation Plan

## Overview

Add evaluation capability that reads MLflow traces, generates judges from task configs, and logs scores back to runs. No separate curation phase.

## Architecture Decisions

- **Guidelines format**: Simple list of strings in task YAML
- **Judge composition**: Hardcoded in Python (functional approach)
- **Judge model**: Fixed to `gpt-4o-mini` (or configurable via env var)
- **Results storage**: Log metrics back to existing simulation runs

## File Changes Required

### 1. Update Pydantic Models

**File**: `src/beyond_vibes/simulations/models.py`

Add `guidelines` field to `SimulationConfig`:

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
    guidelines: list[str] = []  # NEW - success criteria for judges
```

### 2. Update Task YAML Files

**Files**:

- `src/beyond_vibes/simulations/prompts/tasks/poetry_to_uv.yaml`
- `src/beyond_vibes/simulations/prompts/tasks/unit_tests.yaml`

Add guidelines section to `poetry_to_uv.yaml`:

```yaml
name: "poetry_to_uv"
archetype: "repo_maintenance"
guidelines:
  - "The pyproject.toml must not contain [tool.poetry] sections"
  - "The poetry.lock file must be removed or replaced with uv.lock"
  - "The project must be installable with `uv sync`"
```

### 3. Create Evaluators Module Structure

**New directory structure**:

```
src/beyond_vibes/evaluators/
├── __init__.py              # Public API exports
├── models.py                # Pydantic models (JudgeInput, EvalResult)
├── extractor.py             # MLflow run data extraction
├── judges.py                # Judge creation functions
├── archetypes/              # Per-archetype judge logic
│   ├── __init__.py
│   └── repo_maintenance.py  # Repo maintenance specific judges
└── runner.py                # Main evaluation orchestration
```

### 4. Extractor Module

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
    trace_metrics: dict  # total_tokens, tool_error_count, etc.
    raw_trace: mlflow.entities.Trace  # Full trace for complex judges

def extract_run_data(run_id: str) -> JudgeInput:
    """Extract standardized input from MLflow run."""
    # Load run
    # Load artifacts (system_prompt.txt, git_diff.patch)
    # Extract final message from last span
    # Parse trace metrics
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

### 5. Judges Module

**File**: `src/beyond_vibes/evaluators/judges.py`

**Purpose**: Create and configure MLflow judges

```python
JUDGE_MODEL = "openai:/gpt-4o-mini"  # Fixed model

def create_guidelines_judge(task_config: SimulationConfig) -> Guidelines | None:
    """Create Guidelines judge from task config."""
    if not task_config.guidelines:
        return None
    return Guidelines(
        name=f"{task_config.name}_guidelines",
        guidelines=task_config.guidelines,
        model=JUDGE_MODEL,
    )

def create_tool_efficiency_judge() -> ToolCallEfficiency:
    """Universal judge for all tasks."""
    return ToolCallEfficiency(model=JUDGE_MODEL)

def create_universal_judges(task_config: SimulationConfig) -> list:
    """Create judges applied to all archetypes."""
    judges = [create_tool_efficiency_judge()]
    
    guidelines = create_guidelines_judge(task_config)
    if guidelines:
        judges.append(guidelines)
    
    return judges
```

### 6. Archetype Module

**File**: `src/beyond_vibes/evaluators/archetypes/repo_maintenance.py`

**Purpose**: Archetype-specific judges (initially empty, extensible)

```python
def create_repo_maintenance_judges(task_config: SimulationConfig) -> list:
    """Additional judges for repo maintenance tasks."""
    judges = []
    # Future: add code quality judges, diff analysis, etc.
    return judges
```

### 7. Judge Registry

**File**: `src/beyond_vibes/evaluators/judges.py` (extension)

```python
from beyond_vibes.evaluators.archetypes import repo_maintenance

# Registry mapping archetypes to their judge creators
ARCHETYPE_JUDGE_CREATORS: dict[str, list[Callable]] = {
    "repo_maintenance": [
        create_universal_judges,
        repo_maintenance.create_repo_maintenance_judges,
    ],
    # Future archetypes added here
}

def build_judges_for_task(task_config: SimulationConfig) -> list:
    """Factory function - composes judges based on archetype."""
    creators = ARCHETYPE_JUDGE_CREATORS.get(
        task_config.archetype, 
        [create_universal_judges]
    )
    
    judges = []
    for creator in creators:
        judges.extend(creator(task_config))
    return judges
```

### 8. Runner Module

**File**: `src/beyond_vibes/evaluators/runner.py`

**Purpose**: Orchestrate evaluation pipeline

```python
class EvaluationRunner:
    """Runs judges on simulation runs and logs results."""
    
    def __init__(self, judge_model: str = JUDGE_MODEL):
        self.judge_model = judge_model
    
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
        
        # Run evaluation
        eval_data = self._prepare_eval_data(judge_input)
        results = mlflow.genai.evaluate(
            data=[eval_data],
            scorers=judges,
        )
        
        # Log results back to run
        self._log_results_to_run(run_id, results)
        
        return results
    
    def _prepare_eval_data(self, judge_input: JudgeInput) -> dict:
        """Convert JudgeInput to format expected by mlflow.genai.evaluate."""
        return {
            "inputs": {
                "request": judge_input.task_prompt,
                "system": judge_input.system_prompt,
            },
            "outputs": {
                "response": judge_input.final_message,
                "git_diff": judge_input.git_diff,
            },
            "trace": judge_input.raw_trace,  # For ToolCallEfficiency
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
```

### 9. CLI Command

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
        for run in runs:
            try:
                results = runner.evaluate_run(run.info.run_id)
                logger.info(f"Evaluated run {run.info.run_id}")
            except Exception as e:
                logger.error(f"Failed to evaluate run {run.info.run_id}: {e}")
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Task YAML (poetry_to_uv.yaml)                              │
│  ├── archetype: repo_maintenance                           │
│  └── guidelines: [criterion_1, criterion_2, ...]           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  MLflow Run (from simulation)                               │
│  ├── Tags: task.name=poetry_to_uv                          │
│  ├── Artifacts:                                            │
│  │   ├── system_prompt.txt                                 │
│  │   └── git_diff.patch                                    │
│  └── Spans: message_0, message_1, ..., message_N           │
│       └── TOOL spans with inputs/outputs                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Extractor (extractor.py)                                   │
│  └── JudgeInput:                                           │
│      ├── task_prompt, system_prompt                        │
│      ├── final_message (last span output)                  │
│      ├── git_diff                                          │
│      └── trace_metrics (token counts, errors)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Judge Builder (judges.py)                                  │
│  ├── Universal judges:                                     │
│  │   ├── ToolCallEfficiency()                              │
│  │   └── Guidelines(criteria from YAML)                    │
│  └── Archetype judges (repo_maintenance.py)                │
│      └── [empty for now]                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  mlflow.genai.evaluate()                                    │
│  ├── Judges score: pass/fail + rationale                   │
│  └── Returns: Feedback objects                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Results Logged to Original Run                             │
│  ├── eval_tool_call_efficiency: 0.85                       │
│  ├── eval_poetry_to_uv_guidelines: 1.0                     │
│  └── eval_poetry_to_uv_guidelines_rationale: "All criteria │
│      met: pyproject.toml updated, poetry.lock removed..."   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variable (optional)

```bash
# Override default judge model
export BV_JUDGE_MODEL="openai:/gpt-4o"
```

### Settings.py addition

```python
class Settings(BaseSettings):
    # ... existing settings ...
    judge_model: str = "openai:/gpt-4o-mini"
```

## Dependencies to Add

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing ...
    "mlflow[genai]>=3.10.0",  # Ensure genai extras
    "openai>=1.0.0",  # Required for judges
]
```

## Testing Strategy

1. **Unit tests** for extractor (mock MLflow runs)
2. **Integration test** with real run ID from your existing traces
3. **Validation**: Verify guidelines from poetry_to_uv.yaml create correct Guidelines judge

## Future Archetype Addition

When adding "architectural_planning":

1. Create `src/beyond_vibes/evaluators/archetypes/architectural_planning.py`
2. Add to registry:

```python
ARCHETYPE_JUDGE_CREATORS["architectural_planning"] = [
    create_universal_judges,
    architectural_planning.create_architectural_planning_judges,
]
```

3. Guidelines in task YAML still work automatically

## Success Criteria

- [ ] Can run `beyond-vibes evaluate --run-id <id>` and get scores logged
- [ ] Guidelines from task YAML are automatically converted to Guidelines judge
- [ ] ToolCallEfficiency works on existing traces (detects loops)
- [ ] Results visible in MLflow UI with scores + rationale
- [ ] New archetype can be added by creating one file + registry entry
