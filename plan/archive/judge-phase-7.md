# Judge Phase 7: CLI Command and Task YAML Updates

## Overview

Add the CLI command for evaluation and update example task YAML files with judges configuration. This is the final phase that exposes evaluation functionality to users.

## Prerequisites

- ✅ Phase 6: EvaluationRunner with `evaluate_run()` and `evaluate_batch()`
- ✅ Phase 2: Settings with judge configuration
- CLI framework (Typer) already in use

## Changes Required

### 1. Add CLI Evaluate Command

**File**: `src/beyond_vibes/cli.py` (addition)

**Upstream Dependencies**:
- Phase 6: `EvaluationRunner` and `query_simulation_runs`
- Existing CLI structure with Typer
- Settings for defaults

**Downstream Impact**:
- User-facing command: `beyond-vibes evaluate --run-id <id>`
- No other dependencies

**Implementation**:

```python
# Add to existing imports in cli.py
from beyond_vibes.evaluations.runner import EvaluationRunner
from beyond_vibes.evaluations.extractor import query_simulation_runs


# Add this command to the Typer app
@app.command()
def evaluate(
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        "-r",
        help="Specific run to evaluate (if omitted, evaluates all matching filters)",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Filter by task name",
    ),
    archetype: str | None = typer.Option(
        None,
        "--archetype",
        "-a",
        help="Filter by archetype",
    ),
    experiment: str = typer.Option(
        "beyond-vibes",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    judge_model: str | None = typer.Option(
        None,
        "--judge-model",
        "-m",
        help="Override judge model (e.g., 'gpt-4o', 'local-model')",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Show what would be evaluated without running",
    ),
) -> None:
    """Evaluate simulation runs using configured judges.
    
    Evaluates runs from MLflow using judges defined in task configuration.
    Results are logged back to the original runs.
    
    Examples:
        # Evaluate specific run
        beyond-vibes evaluate --run-id abc123
        
        # Evaluate all runs for a task
        beyond-vibes evaluate --task poetry_to_uv
        
        # Evaluate with different model
        beyond-vibes evaluate --run-id abc123 --judge-model gpt-4o
        
        # Dry run to see what would be evaluated
        beyond-vibes evaluate --task poetry_to_uv --dry-run
    """
    # Initialize runner
    runner = EvaluationRunner(judge_model=judge_model)
    
    if run_id:
        # Single run evaluation
        logger.info(f"Evaluating single run: {run_id}")
        
        if dry_run:
            typer.echo(f"Would evaluate run: {run_id}")
            return
        
        try:
            results = runner.evaluate_run(run_id)
            
            # Display results
            typer.echo(f"✓ Evaluated run {run_id}")
            for judge_name, result in results.items():
                if "error" in result:
                    typer.echo(f"  ✗ {judge_name}: ERROR - {result['error']}")
                else:
                    score = result.get("score", 0.0)
                    typer.echo(f"  ✓ {judge_name}: {score:.2f}")
        
        except Exception as e:
            logger.error(f"Failed to evaluate run {run_id}: {e}")
            raise typer.Exit(1)
    
    else:
        # Batch evaluation
        logger.info(f"Querying runs from experiment: {experiment}")
        
        try:
            runs = query_simulation_runs(
                experiment=experiment,
                task_name=task,
                archetype=archetype,
            )
        except Exception as e:
            logger.error(f"Failed to query runs: {e}")
            raise typer.Exit(1)
        
        if not runs:
            typer.echo("No runs found matching criteria")
            raise typer.Exit(0)
        
        typer.echo(f"Found {len(runs)} runs to evaluate")
        
        if dry_run:
            for run in runs:
                typer.echo(f"  Would evaluate: {run.info.run_id}")
            return
        
        # Evaluate all runs
        success_count = 0
        error_count = 0
        
        with typer.progressbar(runs, label="Evaluating") as progress:
            for run in progress:
                try:
                    results = runner.evaluate_run(run.info.run_id)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to evaluate run {run.info.run_id}: {e}")
                    error_count += 1
        
        # Summary
        typer.echo(f"\nEvaluation complete:")
        typer.echo(f"  ✓ Successful: {success_count}")
        if error_count > 0:
            typer.echo(f"  ✗ Failed: {error_count}")
            raise typer.Exit(1)
```

---

### 2. Update poetry_to_uv.yaml Task

**File**: `src/beyond_vibes/simulations/prompts/tasks/poetry_to_uv.yaml`

**Changes**: Add guidelines and judges sections

```yaml
name: "poetry_to_uv"
description: "Migrate a project from poetry to uv"
archetype: "repo_maintenance"

# Evaluation criteria
guidelines:
  - "The pyproject.toml must not contain [tool.poetry] sections"
  - "The poetry.lock file must be removed or replaced with uv.lock"
  - "The project must be installable with `uv sync`"

# Judges to run on this task
judges:
  - name: guidelines
    input: git_diff  # Evaluates git diff against guidelines
  - name: tool_efficiency
    input: trace     # Uses trace metrics
  - name: faithfulness
    input: git_diff  # POC: DeepEval on git diff

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
  
  Steps:
  1. Check the current pyproject.toml for poetry configuration
  2. Create a new pyproject.toml with uv configuration
  3. Remove poetry.lock if present
  4. Run `uv sync` to generate uv.lock
  5. Verify the project installs correctly
  
  Do not ask for confirmation. Just proceed with the migration.
```

---

### 3. Update unit_tests.yaml Task

**File**: `src/beyond_vibes/simulations/prompts/tasks/unit_tests.yaml`

**Changes**: Add guidelines and judges sections

```yaml
name: "unit_tests"
description: "Add comprehensive unit tests to a project"
archetype: "feature_implementation"

# Evaluation criteria
guidelines:
  - "Tests must cover the main functionality of the module"
  - "Tests should use pytest framework"
  - "Test files should be in tests/ directory"
  - "Tests should have meaningful assertions"

# Judges to run on this task
judges:
  - name: guidelines
    input: git_diff
  - name: tool_efficiency
    input: trace

repository:
  url: "https://github.com/example/project"
  branch: "main"

agent: "build"
max_turns: 30
capture_git_diff: true

system_prompt: |
  You are a senior software engineer specializing in test-driven development.
  Write comprehensive, well-structured unit tests.

prompt: |
  Add comprehensive unit tests for the main module.
  
  Requirements:
  - Use pytest as the testing framework
  - Cover edge cases and error conditions
  - Follow AAA pattern (Arrange, Act, Assert)
  - Include docstrings explaining what each test verifies
```

---

### 4. Update __init__.py (Optional)

**File**: `src/beyond_vibes/evaluations/__init__.py`

Already complete from Phase 6, but verify:

```python
"""Evaluations module for judging simulation runs."""

from beyond_vibes.evaluations.extractor import extract_run_data, query_simulation_runs
from beyond_vibes.evaluations.judge_factory import (
    create_judge,
    build_judges_for_task,
    list_available_judges,
)
from beyond_vibes.evaluations.models import JudgeInput
from beyond_vibes.evaluations.runner import EvaluationRunner, evaluate_run

__all__ = [
    "extract_run_data",
    "query_simulation_runs",
    "JudgeInput",
    "create_judge",
    "build_judges_for_task",
    "list_available_judges",
    "EvaluationRunner",
    "evaluate_run",
]
```

---

## Usage Examples

### Command Line Usage

```bash
# Use local model via litellm
export JUDGE_BASE_URL="http://localhost:8000/v1"
beyond-vibes evaluate --run-id abc123

# Evaluate all runs for a task
beyond-vibes evaluate --task poetry_to_uv

# Evaluate by archetype
beyond-vibes evaluate --archetype repo_maintenance

# Use different judge model
beyond-vibes evaluate --run-id abc123 --judge-model gpt-4o

# Use local model via litellm
export JUDGE_BASE_URL="http://localhost:8000/v1"
beyond-vibes evaluate --run-id abc123

# Dry run to preview
beyond-vibes evaluate --task poetry_to_uv --dry-run

# Batch evaluation with progress bar
beyond-vibes evaluate --task poetry_to_uv --experiment beyond-vibes
```

### Programmatic Usage

```python
from beyond_vibes.evaluations import EvaluationRunner

# Single run
runner = EvaluationRunner()
results = runner.evaluate_run("run-id-123")

# With custom model
runner = EvaluationRunner(judge_model="openai:/gpt-4o")
results = runner.evaluate_run("run-id-123")

# Batch evaluation
run_ids = ["run-1", "run-2", "run-3"]
all_results = runner.evaluate_batch(run_ids)
```

---

## Dependencies

### Phase Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| Phase 7 | Phase 6 | CLI uses `EvaluationRunner` |
| Phase 7 | Phase 2 | Task YAMLs use `JudgeMapping` schema |

### File Dependencies

```
Phase 6 outputs:
  └── src/beyond_vibes/evaluations/runner.py
        ↓
Phase 7 updates:
  ├── src/beyond_vibes/cli.py (adds evaluate command)
  └── src/beyond_vibes/simulations/prompts/tasks/*.yaml
```

---

## Testing Strategy

### CLI Tests

```python
# tests/unit/test_cli_evaluate.py

import pytest
from typer.testing import CliRunner

from beyond_vibes.cli import app

runner = CliRunner()


def test_evaluate_help():
    """Test evaluate command help."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--run-id" in result.output
    assert "--task" in result.output


@patch("beyond_vibes.cli.EvaluationRunner")
def test_evaluate_single_run(mock_runner_class):
    """Test evaluating single run."""
    mock_runner = Mock()
    mock_runner.evaluate_run.return_value = {
        "guidelines": {"score": 0.95},
    }
    mock_runner_class.return_value = mock_runner
    
    result = runner.invoke(app, ["evaluate", "--run-id", "abc123"])
    
    assert result.exit_code == 0
    assert "0.95" in result.output
    mock_runner.evaluate_run.assert_called_once_with("abc123")


@patch("beyond_vibes.cli.query_simulation_runs")
@patch("beyond_vibes.cli.EvaluationRunner")
def test_evaluate_batch(mock_runner_class, mock_query):
    """Test batch evaluation."""
    # Setup mocks
    mock_run = Mock()
    mock_run.info.run_id = "run-1"
    mock_query.return_value = [mock_run]
    
    mock_runner = Mock()
    mock_runner.evaluate_run.return_value = {"guidelines": {"score": 0.9}}
    mock_runner_class.return_value = mock_runner
    
    result = runner.invoke(app, ["evaluate", "--task", "test_task"])
    
    assert result.exit_code == 0
    mock_query.assert_called_once_with(
        experiment="beyond-vibes",
        task_name="test_task",
        archetype=None,
    )


def test_evaluate_dry_run():
    """Test dry-run option."""
    result = runner.invoke(app, ["evaluate", "--run-id", "abc123", "--dry-run"])
    
    assert result.exit_code == 0
    assert "Would evaluate" in result.output
```

### Task YAML Validation

```python
# tests/unit/test_task_yaml.py

import yaml
from pathlib import Path

import pytest


def test_poetry_to_uv_has_judges():
    """Test that poetry_to_uv task has judges configured."""
    task_path = Path("src/beyond_vibes/simulations/prompts/tasks/poetry_to_uv.yaml")
    
    with open(task_path) as f:
        task = yaml.safe_load(f)
    
    assert "judges" in task
    assert len(task["judges"]) > 0
    
    # Verify judge mappings
    for judge in task["judges"]:
        assert "name" in judge
        assert "input" in judge
        assert judge["input"] in ["git_diff", "final_message"]


def test_all_tasks_have_valid_judges():
    """Test that all task YAMLs have valid judge configurations."""
    tasks_dir = Path("src/beyond_vibes/simulations/prompts/tasks")
    
    for task_file in tasks_dir.glob("*.yaml"):
        with open(task_file) as f:
            task = yaml.safe_load(f)
        
        if "judges" in task:
            for judge in task["judges"]:
                assert "name" in judge, f"{task_file}: judge missing name"
                assert "input" in judge, f"{task_file}: judge missing input"
```

---

## Error Handling

### CLI Error Scenarios

| Scenario | Behavior | Exit Code |
|----------|----------|-----------|
| Run not found | Error message, exit | 1 |
| No runs match filters | Warning message, exit 0 | 0 |
| Evaluation fails | Error per run, summary | 1 if any fail |
| Judge config invalid | Error details | 1 |
| MLflow not available | Connection error | 1 |

---

## Success Criteria

- [ ] `beyond-vibes evaluate --help` shows command documentation
- [ ] Can evaluate single run: `beyond-vibes evaluate --run-id <id>`
- [ ] Can evaluate by task: `beyond-vibes evaluate --task <name>`
- [ ] Can evaluate by archetype: `beyond-vibes evaluate --archetype <type>`
- [ ] `--dry-run` option works
- [ ] `--judge-model` override works
- [ ] Progress bar shown for batch evaluation
- [ ] Results displayed in CLI output
- [ ] Exit code 0 on success, 1 on failure
- [ ] `poetry_to_uv.yaml` has guidelines and judges
- [ ] `unit_tests.yaml` has guidelines and judges
- [ ] All task YAMLs have valid judge configurations
- [ ] All CLI tests pass
- [ ] Manual end-to-end test works

---

## Completion Checklist

This completes the judges implementation. Verify all phases:

- [ ] Phase 1: trace_session.json logging (already done)
- [ ] Phase 2: Core Data Models (JudgeMapping, judges.yaml, Settings)
- [ ] Phase 3: Evaluations Module Structure (__init__.py)
- [ ] Phase 4: Extractor Module (extract_run_data, JudgeInput)
- [ ] Phase 5: Judge Factory Module (create_judge, build_judges_for_task)
- [ ] Phase 6: Runner Module (EvaluationRunner)
- [ ] Phase 7: CLI Command and Task YAMLs

### Final Integration Test

```bash
# 1. Run simulation
beyond-vibes simulate --task poetry_to_uv --model gpt-4o

# 2. Get run ID from output
RUN_ID="<run-id-from-step-1>"

# 3. Evaluate the run
beyond-vibes evaluate --run-id $RUN_ID

# 4. Check MLflow UI for eval_* metrics
# Should see: eval_guidelines, eval_tool_efficiency, etc.

# 5. Batch evaluation
beyond-vibes evaluate --task poetry_to_uv

# 6. With different model
beyond-vibes evaluate --run-id $RUN_ID --judge-model gpt-4o-mini
```

---

## Next Steps

After completing all phases:

1. **Update README** with evaluation documentation
2. **Add integration tests** with real MLflow runs
3. **Monitor first evaluations** for any edge cases
4. **Consider adding more judges** to registry as needed
5. **Evaluate custom judge support** if needed

The evaluation system is now complete and ready for use!
