# Judge Phase 6: Runner Module

## Overview

Implement the EvaluationRunner that orchestrates the entire evaluation pipeline. This is the main entry point that ties together extraction, judge creation, and result logging.

## Prerequisites

- ✅ Phase 5: Judge factory with `create_judge()` and `build_judges_for_task()`
- ✅ Phase 4: Extractor with `extract_run_data()` and `JudgeInput`
- ✅ Phase 2: Settings with judge configuration

## Changes Required

### 1. Create Runner Module

**File**: `src/beyond_vibes/evaluations/runner.py`

**Upstream Dependencies**:
- Phase 4: `extract_run_data()`, `JudgeInput`
- Phase 5: `build_judges_for_task()`, `create_judge()`
- Phase 2: `Settings.judge_model`
- Task loader from simulations module
- MLflow evaluate API

**Downstream Impact**:
- Phase 7: CLI imports and uses `EvaluationRunner`
- Results are logged back to MLflow runs

**Implementation**:

```python
"""Evaluation runner for orchestrating judge execution.

This module provides the main entry point for running evaluations
on simulation runs.

Example:
    from beyond_vibes.evaluators import EvaluationRunner
    
    runner = EvaluationRunner()
    results = runner.evaluate_run("run-id-123")
    
    # Batch evaluation
    runner.evaluate_batch(["run-1", "run-2"])
"""

import json
import logging
from typing import Any

import mlflow

from beyond_vibes.evaluations.extractor import extract_run_data, query_simulation_runs
from beyond_vibes.evaluations.judge_factory import build_judges_for_task
from beyond_vibes.evaluations.models import JudgeInput
from beyond_vibes.settings import settings
from beyond_vibes.simulations.models import SimulationConfig

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs judges on simulation runs and logs results.
    
    This class orchestrates the evaluation pipeline:
    1. Extract data from MLflow run
    2. Load task configuration
    3. Build judges from task config
    4. Run each judge with appropriate input artifact
    5. Log results back to the run
    
    Attributes:
        judge_model: Model to use for judges (overrides settings)
    """
    
    def __init__(self, judge_model: str | None = None):
        """Initialize evaluation runner.
        
        Args:
            judge_model: Override default judge model from settings
        """
        self.judge_model = judge_model or settings.judge_model
        logger.info(f"EvaluationRunner initialized with model: {self.judge_model}")
    
    def evaluate_run(
        self,
        run_id: str,
        judges: list[tuple[Any, str]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single run with specified judges.
        
        If judges not provided, loads them from task configuration.
        
        Args:
            run_id: MLflow run ID to evaluate
            judges: List of (judge_instance, input_artifact) tuples (optional)
            
        Returns:
            Dictionary mapping judge names to their results
            
        Raises:
            ValueError: If run cannot be evaluated
            mlflow.exceptions.MlflowException: If MLflow operation fails
        """
        logger.info(f"Evaluating run: {run_id}")
        
        # Step 1: Extract data from run
        try:
            judge_input = extract_run_data(run_id)
            logger.debug(f"Extracted data for task: {judge_input.task_name}")
        except Exception as e:
            logger.error(f"Failed to extract data for run {run_id}: {e}")
            raise ValueError(f"Cannot evaluate run {run_id}: {e}")
        
        # Step 2: Load task config
        try:
            task_config = self._load_task_config(judge_input.task_name)
        except Exception as e:
            logger.error(f"Failed to load task config for '{judge_input.task_name}': {e}")
            raise ValueError(f"Cannot load task config: {e}")
        
        # Step 3: Build judges if not provided
        if judges is None:
            judges = build_judges_for_task(task_config, self.judge_model)
        
        if not judges:
            logger.warning(f"No judges configured for run {run_id}")
            return {}
        
        logger.info(f"Running {len(judges)} judges on run {run_id}")
        
        # Step 4: Evaluate with each judge
        all_results = {}
        
        for judge, input_artifact in judges:
            try:
                result = self._evaluate_with_judge(
                    judge, input_artifact, judge_input
                )
                all_results[judge.name] = result
                
                # Step 5: Log results
                self._log_results_to_run(run_id, judge.name, result)
                
            except Exception as e:
                logger.error(f"Judge '{judge.name}' failed for run {run_id}: {e}")
                all_results[judge.name] = {"error": str(e)}
        
        logger.info(f"Completed evaluation of run {run_id}")
        return all_results
    
    def evaluate_batch(
        self,
        run_ids: list[str],
        continue_on_error: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate multiple runs in batch.
        
        Args:
            run_ids: List of MLflow run IDs to evaluate
            continue_on_error: If True, continue evaluating other runs on error
            
        Returns:
            Dictionary mapping run IDs to their results
        """
        results = {}
        success_count = 0
        
        for run_id in run_ids:
            try:
                run_results = self.evaluate_run(run_id)
                results[run_id] = run_results
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to evaluate run {run_id}: {e}")
                results[run_id] = {"error": str(e)}
                if not continue_on_error:
                    break
        
        logger.info(f"Batch evaluation complete: {success_count}/{len(run_ids)} runs successful")
        return results
    
    def _evaluate_with_judge(
        self,
        judge: Any,
        input_artifact: str,
        judge_input: JudgeInput,
    ) -> dict[str, Any]:
        """Evaluate a single judge with appropriate input.
        
        Args:
            judge: Judge instance
            input_artifact: Which artifact to use (git_diff, final_message, trace)
            judge_input: Standardized input data
            
        Returns:
            Evaluation result dictionary
        """
        # Prepare evaluation data based on input artifact
        eval_data = self._prepare_eval_data(judge_input, input_artifact)
        
        logger.debug(f"Running judge '{judge.name}' on {input_artifact}")
        
        # Run evaluation via MLflow
        # Note: mlflow.evaluate expects a specific data format
        results = mlflow.evaluate(
            data=eval_data,
            model=self.judge_model,
            judges=[judge],
        )
        
        # Extract result (mlflow.evaluate returns an EvaluationResult or list)
        if hasattr(results, 'metrics'):
            # Single result object
            return {
                "score": results.metrics.get(judge.name, 0.0),
                "rationale": getattr(results, 'rationale', None),
            }
        elif isinstance(results, list) and len(results) > 0:
            # List of results
            result = results[0]
            return {
                "score": getattr(result, 'score', 0.0),
                "rationale": getattr(result, 'rationale', None),
            }
        else:
            logger.warning(f"Unexpected result format from judge '{judge.name}': {type(results)}")
            return {"score": 0.0, "rationale": None}
    
    def _prepare_eval_data(
        self,
        judge_input: JudgeInput,
        input_artifact: str,
    ) -> dict[str, Any]:
        """Convert JudgeInput to format expected by mlflow.evaluate.
        
        Args:
            judge_input: Standardized input data
            input_artifact: Which artifact to use
            
        Returns:
            Dictionary in mlflow.evaluate format
            
        Format:
            {
                "inputs": {"request": ..., "system": ...},
                "outputs": {"response": ...},
                "context": ...
            }
        """
        # Map input artifact to the content that will be evaluated
        if input_artifact == "git_diff":
            output_content = judge_input.git_diff or ""
            context = None  # Context not needed when diff is the output
        elif input_artifact == "final_message":
            output_content = judge_input.final_message
            context = judge_input.git_diff  # Use diff as context if available
        elif input_artifact == "trace":
            # For trace-based evaluation, serialize trace as output
            output_content = json.dumps(judge_input.trace, indent=2)
            context = judge_input.git_diff
        else:
            raise ValueError(f"Unknown input artifact: {input_artifact}")
        
        return {
            "inputs": {
                "request": judge_input.task_prompt,
                "system": judge_input.system_prompt,
            },
            "outputs": {
                "response": output_content,
            },
            "context": context,
        }
    
    def _log_results_to_run(
        self,
        run_id: str,
        judge_name: str,
        result: dict[str, Any],
    ) -> None:
        """Log evaluation results back to the simulation run.
        
        Args:
            run_id: MLflow run ID
            judge_name: Name of judge
            result: Evaluation result dictionary
        """
        try:
            with mlflow.start_run(run_id=run_id):
                # Log score as metric
                score = result.get("score", 0.0)
                metric_name = f"eval_{judge_name}"
                mlflow.log_metric(metric_name, score)
                
                # Log rationale as tag (truncated to 500 chars for MLflow limits)
                rationale = result.get("rationale")
                if rationale:
                    tag_name = f"{metric_name}_rationale"
                    mlflow.set_tag(tag_name, rationale[:500])
                
                logger.debug(f"Logged results for '{judge_name}': score={score}")
        
        except Exception as e:
            logger.error(f"Failed to log results for run {run_id}: {e}")
            # Don't raise - logging failures shouldn't stop evaluation
    
    def _load_task_config(self, task_name: str) -> SimulationConfig:
        """Load task configuration by name.
        
        Args:
            task_name: Name of task
            
        Returns:
            Task configuration
            
        Raises:
            ValueError: If task cannot be loaded
        """
        try:
            from beyond_vibes.simulations.prompts.loader import load_task_config
            # Empty context dict for evaluation
            return load_task_config(task_name, "{}")
        except Exception as e:
            raise ValueError(f"Failed to load task '{task_name}': {e}")


# Convenience function for simple use cases
def evaluate_run(run_id: str, judge_model: str | None = None) -> dict[str, Any]:
    """Evaluate a single run with default configuration.
    
    Convenience function that creates a runner and evaluates.
    
    Args:
        run_id: MLflow run ID
        judge_model: Optional model override
        
    Returns:
        Evaluation results
    """
    runner = EvaluationRunner(judge_model=judge_model)
    return runner.evaluate_run(run_id)
```

---

### 2. Update __init__.py

**File**: `src/beyond_vibes/evaluations/__init__.py`

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

## Dependencies

### Phase Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| Phase 6 | Phase 5 | Uses `build_judges_for_task()` |
| Phase 6 | Phase 4 | Uses `extract_run_data()` and `JudgeInput` |
| Phase 7 | Phase 6 | CLI creates `EvaluationRunner` instance |

### File Dependencies

```
Phase 5 outputs:
  └── src/beyond_vibes/evaluations/judge_factory.py
        ↓
Phase 4 outputs:
  └── src/beyond_vibes/evaluations/extractor.py
        ↓
Phase 6 creates:
  └── src/beyond_vibes/evaluations/runner.py
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/evaluations/test_runner.py

import pytest
from unittest.mock import Mock, patch, MagicMock

from beyond_vibes.evaluations.runner import EvaluationRunner
from beyond_vibes.evaluations.models import JudgeInput


class TestEvaluationRunner:
    """Test EvaluationRunner orchestration."""
    
    def test_init_uses_settings_default(self):
        """Test that runner uses settings.judge_model by default."""
        with patch("beyond_vibes.evaluations.runner.settings") as mock_settings:
            mock_settings.judge_model = "openai:/gpt-4o"
            runner = EvaluationRunner()
            assert runner.judge_model == "openai:/gpt-4o"
    
    def test_init_allows_override(self):
        """Test that runner accepts model override."""
        runner = EvaluationRunner(judge_model="openai:/gpt-4o-mini")
        assert runner.judge_model == "openai:/gpt-4o-mini"
    
    @patch("beyond_vibes.evaluations.runner.extract_run_data")
    @patch("beyond_vibes.evaluations.runner.build_judges_for_task")
    @patch("beyond_vibes.evaluations.runner.mlflow.evaluate")
    def test_evaluate_run_success(self, mock_evaluate, mock_build, mock_extract):
        """Test successful run evaluation."""
        # Setup mocks
        mock_judge_input = Mock(spec=JudgeInput)
        mock_judge_input.task_name = "test_task"
        mock_extract.return_value = mock_judge_input
        
        mock_judge = Mock()
        mock_judge.name = "test_guidelines"
        mock_build.return_value = [(mock_judge, "git_diff")]
        
        mock_result = Mock()
        mock_result.score = 0.95
        mock_result.rationale = "Good job"
        mock_evaluate.return_value = [mock_result]
        
        # Run
        runner = EvaluationRunner()
        results = runner.evaluate_run("run-123")
        
        # Verify
        assert "test_guidelines" in results
        assert results["test_guidelines"]["score"] == 0.95
    
    @patch("beyond_vibes.evaluations.runner.extract_run_data")
    def test_evaluate_run_handles_extraction_error(self, mock_extract):
        """Test that extraction errors are handled gracefully."""
        mock_extract.side_effect = Exception("Run not found")
        
        runner = EvaluationRunner()
        with pytest.raises(ValueError, match="Cannot evaluate run"):
            runner.evaluate_run("run-123")


class TestPrepareEvalData:
    """Test evaluation data preparation."""
    
    def test_git_diff_input(self):
        """Test preparing data for git_diff input."""
        runner = EvaluationRunner()
        judge_input = Mock(spec=JudgeInput)
        judge_input.task_prompt = "Do something"
        judge_input.system_prompt = "You are an agent"
        judge_input.git_diff = "diff --git a/file.py"
        
        data = runner._prepare_eval_data(judge_input, "git_diff")
        
        assert data["inputs"]["request"] == "Do something"
        assert data["outputs"]["response"] == "diff --git a/file.py"
        assert data["context"] is None
    
    def test_final_message_input(self):
        """Test preparing data for final_message input."""
        runner = EvaluationRunner()
        judge_input = Mock(spec=JudgeInput)
        judge_input.task_prompt = "Do something"
        judge_input.final_message = "I did it"
        judge_input.git_diff = "diff content"
        
        data = runner._prepare_eval_data(judge_input, "final_message")
        
        assert data["outputs"]["response"] == "I did it"
        assert data["context"] == "diff content"
    
    def test_trace_input(self):
        """Test preparing data for trace input."""
        runner = EvaluationRunner()
        judge_input = Mock(spec=JudgeInput)
        judge_input.task_prompt = "Do something"
        judge_input.trace = {"total_messages": 5}
        judge_input.git_diff = "diff content"
        
        data = runner._prepare_eval_data(judge_input, "trace")
        
        assert "total_messages" in data["outputs"]["response"]
        assert data["context"] == "diff content"


class TestBatchEvaluation:
    """Test batch evaluation."""
    
    @patch.object(EvaluationRunner, "evaluate_run")
    def test_evaluate_batch_success(self, mock_evaluate):
        """Test batch evaluation with successes."""
        mock_evaluate.side_effect = [
            {"guidelines": {"score": 0.9}},
            {"guidelines": {"score": 0.8}},
        ]
        
        runner = EvaluationRunner()
        results = runner.evaluate_batch(["run-1", "run-2"])
        
        assert len(results) == 2
        assert results["run-1"]["guidelines"]["score"] == 0.9
```

### Integration Tests

```python
# tests/integration/test_runner_integration.py

import pytest

pytestmark = pytest.mark.integration


def test_end_to_end_evaluation():
    """Test full evaluation pipeline with real run."""
    from beyond_vibes.evaluations import EvaluationRunner
    
    # Requires existing run with judges configured
    run_id = "test-run-id"  # Replace with actual ID
    
    runner = EvaluationRunner()
    results = runner.evaluate_run(run_id)
    
    assert isinstance(results, dict)
    # Results should contain scores for configured judges


def test_batch_evaluation():
    """Test batch evaluation with multiple runs."""
    from beyond_vibes.evaluations import EvaluationRunner
    
    run_ids = ["run-1", "run-2"]  # Replace with actual IDs
    
    runner = EvaluationRunner()
    results = runner.evaluate_batch(run_ids)
    
    assert len(results) == len(run_ids)
```

---

## Error Handling

### Error Scenarios

| Scenario | Behavior | Log Level |
|----------|----------|-----------|
| Run not found | Raise ValueError | ERROR |
| Task config missing | Raise ValueError | ERROR |
| No judges configured | Return empty dict, log warning | WARNING |
| Individual judge fails | Continue with other judges | ERROR |
| Result logging fails | Continue, don't raise | ERROR |
| mlflow.evaluate fails | Continue with other judges | ERROR |

---

## Success Criteria

- [ ] `runner.py` exists with `EvaluationRunner` class
- [ ] Can evaluate a single run end-to-end
- [ ] Can evaluate multiple runs in batch
- [ ] Results logged to MLflow as metrics and tags
- [ ] Judge failures don't stop other judges
- [ ] Extraction errors raise ValueError
- [ ] `__init__.py` exports `EvaluationRunner` and `evaluate_run`
- [ ] All unit tests pass
- [ ] Integration test with real run works

---

## Next Phase

**Phase 7**: CLI Command and Task YAML Updates
- Depends on: Phase 6 (runner provides evaluation logic)
- Creates: CLI `evaluate` command and updates example task YAMLs
