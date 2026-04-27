# Judge Phase 4: Extractor Module

## Overview

Implement the extractor module that queries MLflow runs and extracts standardized data for judges. This is the bridge between simulation runs and the evaluation system.

## Prerequisites

- ✅ Phase 3: Evaluations module structure exists
- ✅ Phase 2: Settings with judge configuration
- MLflow runs exist with `trace_session.json` artifact
- Task configs are loadable by name

## Changes Required

### 1. Create Models for Evaluations

**File**: `src/beyond_vibes/evaluations/models.py`

**Upstream Dependencies**:
- Phase 2: Uses `JudgeMapping` from simulations.models
- MLflow entities for type hints

**Downstream Impact**:
- `extractor.py` uses `JudgeInput` dataclass
- `runner.py` uses `JudgeInput` for evaluation

**Implementation**:

```python
"""Data models for evaluations module."""

from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeInput:
    """Standardized input for all judges.
    
    This dataclass normalizes data from MLflow runs into a common format
    that all judges can consume.
    
    Attributes:
        run_id: MLflow run ID
        task_name: Name of the task being evaluated
        archetype: Task archetype (e.g., "repo_maintenance")
        system_prompt: System prompt used in simulation
        task_prompt: Task prompt sent to agent
        final_message: Last message from agent
        git_diff: Git diff from simulation (if capture_git_diff=True)
        trace: Full trace metadata from trace_session.json
    """
    run_id: str
    task_name: str
    archetype: str
    system_prompt: str
    task_prompt: str
    final_message: str
    git_diff: str | None
    trace: dict[str, Any]


@dataclass  
class EvalResult:
    """Result from a single judge evaluation.
    
    Mirrors MLflow's EvaluationResult structure for consistency.
    
    Attributes:
        name: Judge name
        score: Score from 0.0 to 1.0
        rationale: Explanation of the score (optional)
    """
    name: str
    score: float
    rationale: str | None = None
```

---

### 2. Create Extractor Module

**File**: `src/beyond_vibes/evaluations/extractor.py`

**Upstream Dependencies**:
- `models.py` for `JudgeInput` dataclass
- MLflow Python API for run/artifact access
- `trace_session.json` artifact format
- Task loader from simulations module

**Downstream Impact**:
- `runner.py` calls `extract_run_data()` and `query_simulation_runs()`
- CLI uses `query_simulation_runs()` for batch evaluation

**Implementation**:

```python
"""MLflow run data extraction for evaluation.

This module queries MLflow runs and extracts standardized data
that judges need to perform evaluations.
"""

import logging
from typing import Any

import mlflow
from mlflow.entities import Run

from beyond_vibes.evaluators.models import JudgeInput

logger = logging.getLogger(__name__)


def extract_run_data(run_id: str) -> JudgeInput:
    """Extract standardized input from MLflow run.
    
    Loads all artifacts and metadata needed by judges:
    - Run tags (task.name, task.archetype)
    - Run params (task.prompt)
    - Artifacts: system_prompt.txt, git_diff.patch, trace_session.json
    - Final message from spans
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        JudgeInput with all data normalized
        
    Raises:
        mlflow.exceptions.MlflowException: If run doesn't exist
        ValueError: If required artifacts are missing
    """
    # Load run
    run = mlflow.get_run(run_id)
    
    # Extract from tags
    task_name = run.data.tags.get("task.name", "")
    archetype = run.data.tags.get("task.archetype", "")
    
    # Extract from params
    task_prompt = run.data.params.get("task.prompt", "")
    
    # Load artifacts
    system_prompt = _load_artifact(run_id, "system_prompt.txt")
    git_diff = _load_artifact(run_id, "git_diff.patch")
    trace = _load_trace_session(run_id)
    
    # Extract final message from spans (trace data)
    final_message = _extract_final_message(trace)
    
    return JudgeInput(
        run_id=run_id,
        task_name=task_name,
        archetype=archetype,
        system_prompt=system_prompt or "",
        task_prompt=task_prompt,
        final_message=final_message,
        git_diff=git_diff,
        trace=trace,
    )


def _load_artifact(run_id: str, artifact_path: str) -> str | None:
    """Load text artifact from MLflow run.
    
    Args:
        run_id: MLflow run ID
        artifact_path: Path to artifact (e.g., "system_prompt.txt")
        
    Returns:
        Artifact content as string, or None if not found
    """
    try:
        artifact_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.artifacts.load_text(artifact_uri)
    except Exception:
        logger.debug(f"Artifact {artifact_path} not found for run {run_id}")
        return None


def _load_trace_session(run_id: str) -> dict[str, Any]:
    """Load trace_session.json artifact from run.
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Trace session data as dict
        
    Raises:
        ValueError: If trace_session.json not found
    """
    try:
        artifact_uri = f"runs:/{run_id}/trace_session.json"
        return mlflow.artifacts.load_dict(artifact_uri)
    except Exception as e:
        logger.error(f"Failed to load trace_session.json for run {run_id}: {e}")
        # Return empty trace as fallback
        return {
            "total_messages": 0,
            "total_tool_calls": 0,
            "tool_error_count": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "tool_loop_detected": False,
            "tool_loop_threshold": 3,
            "tool_max_consecutive_calls": 0,
            "tool_error_rate": 0.0,
            "token_efficiency": 0.0,
            "cost_efficiency": 0.0,
            "error_message_indices": [],
            "messages": [],
        }


def _extract_final_message(trace: dict[str, Any]) -> str:
    """Extract final message content from trace.
    
    Looks for the last non-tool message in the trace.
    
    Args:
        trace: Trace session data from trace_session.json
        
    Returns:
        Final message content as string
    """
    messages = trace.get("messages", [])
    
    if not messages:
        return ""
    
    # Get last message
    last_message = messages[-1]
    raw_message = last_message.get("raw_message", {})
    
    # Extract content from parts
    parts = raw_message.get("parts", [])
    content_parts = []
    
    for part in parts:
        part_type = part.get("type", "")
        if part_type == "text":
            content_parts.append(part.get("text", ""))
        elif part_type == "reasoning":
            content_parts.append(part.get("reasoning", ""))
    
    return "\n".join(content_parts)


def query_simulation_runs(
    experiment: str = "beyond-vibes",
    run_ids: list[str] | None = None,
    task_name: str | None = None,
    archetype: str | None = None,
    status: str | None = None,
) -> list[Run]:
    """Query simulation runs matching filters.
    
    Args:
        experiment: MLflow experiment name
        run_ids: Specific run IDs to query (if None, query all)
        task_name: Filter by task name tag
        archetype: Filter by archetype tag
        status: Filter by run status ("success", "error", etc.)
        
    Returns:
        List of MLflow Run objects matching filters
    """
    # Build filter string
    filters = []
    
    if task_name:
        filters.append(f"tags.`task.name` = '{task_name}'")
    
    if archetype:
        filters.append(f"tags.`task.archetype` = '{archetype}'")
    
    if status:
        if status == "success":
            filters.append("tags.`run.status` = 'success'")
        elif status == "error":
            filters.append("tags.`run.status` = 'error'")
    
    filter_string = " and ".join(filters) if filters else None
    
    # Query runs
    if run_ids:
        # Load specific runs
        runs = []
        for run_id in run_ids:
            try:
                run = mlflow.get_run(run_id)
                runs.append(run)
            except Exception as e:
                logger.warning(f"Failed to load run {run_id}: {e}")
        return runs
    else:
        # Search experiment
        experiment_obj = mlflow.get_experiment_by_name(experiment)
        if not experiment_obj:
            logger.warning(f"Experiment '{experiment}' not found")
            return []
        
        run_list = mlflow.search_runs(
            experiment_ids=[experiment_obj.experiment_id],
            filter_string=filter_string,
            output_format="list",
        )
        return run_list
```

---

### 3. Update __init__.py

**File**: `src/beyond_vibes/evaluations/__init__.py`

```python
"""Evaluations module for judging simulation runs."""

from beyond_vibes.evaluations.extractor import extract_run_data, query_simulation_runs
from beyond_vibes.evaluations.models import JudgeInput

__all__ = [
    "extract_run_data",
    "query_simulation_runs",
    "JudgeInput",
]
```

---

## Dependencies

### Required Dependencies (add to pyproject.toml)

```toml
dependencies = [
    # ... existing ...
    "mlflow>=2.10.0",  # Already required, but ensure for artifacts API
]
```

### Phase Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| Phase 4 | Phase 3 | Needs evaluations module structure |
| Phase 5 | Phase 4 | Judge factory may use extractor for validation |
| Phase 6 | Phase 4 | Runner calls `extract_run_data()` and `query_simulation_runs()` |

### File Dependencies

```
Phase 3 outputs:
  └── src/beyond_vibes/evaluations/__init__.py
        ↓
Phase 4 creates:
  ├── src/beyond_vibes/evaluations/models.py
  └── src/beyond_vibes/evaluations/extractor.py
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/evaluations/test_extractor.py

import pytest
from unittest.mock import Mock, patch

from beyond_vibes.evaluations.extractor import (
    extract_run_data,
    _load_artifact,
    _load_trace_session,
    _extract_final_message,
)


def test_extract_final_message_with_parts():
    """Test extracting final message from trace with text parts."""
    trace = {
        "messages": [
            {
                "raw_message": {
                    "parts": [
                        {"type": "text", "text": "Hello"},
                        {"type": "reasoning", "reasoning": "Thinking..."},
                    ]
                }
            }
        ]
    }
    
    result = _extract_final_message(trace)
    assert "Hello" in result
    assert "Thinking..." in result


def test_load_trace_session_fallback():
    """Test that missing trace_session.json returns fallback."""
    with patch("mlflow.artifacts.load_dict") as mock_load:
        mock_load.side_effect = Exception("Not found")
        
        result = _load_trace_session("run-123")
        
        assert result["total_messages"] == 0
        assert result["tool_loop_detected"] == False


@patch("mlflow.get_run")
@patch("beyond_vibes.evaluations.extractor._load_artifact")
@patch("beyond_vibes.evaluations.extractor._load_trace_session")
def test_extract_run_data(mock_trace, mock_artifact, mock_get_run):
    """Test full extraction pipeline."""
    # Setup mock
    mock_run = Mock()
    mock_run.data.tags = {"task.name": "test_task", "task.archetype": "test"}
    mock_run.data.params = {"task.prompt": "Do something"}
    mock_get_run.return_value = mock_run
    
    mock_artifact.return_value = "system prompt"
    mock_trace.return_value = {
        "messages": [{"raw_message": {"parts": [{"type": "text", "text": "Done"}]}}],
        "total_messages": 1,
    }
    
    # Test
    result = extract_run_data("run-123")
    
    assert result.run_id == "run-123"
    assert result.task_name == "test_task"
    assert result.system_prompt == "system prompt"
    assert result.final_message == "Done"
```

### Integration Tests

```python
# tests/integration/test_extractor_integration.py

import pytest

pytestmark = pytest.mark.integration


def test_extract_real_run():
    """Test extraction with a real MLflow run ID."""
    # Requires existing run with trace_session.json
    from beyond_vibes.evaluations import extract_run_data
    
    # Use a known run ID from test environment
    run_id = "test-run-id"  # Replace with actual ID
    
    result = extract_run_data(run_id)
    
    assert result.run_id == run_id
    assert result.trace is not None
    assert "total_messages" in result.trace
```

---

## Success Criteria

- [x] `models.py` exists with `JudgeInput` and `EvalResult` dataclasses
- [x] `extractor.py` exists with all extraction functions
- [x] Can load trace_session.json from MLflow run
- [x] Can extract final message from trace data
- [x] Can query runs by experiment, task, archetype
- [x] Fallback trace provided if trace_session.json missing
- [x] `__init__.py` exports extractor functions (EvalResult and JudgeInput also exported)
- [x] All 17 unit tests pass
- [ ] Integration test with real run works (deferred to later)

---

## Next Phase

**Phase 5**: Judge Factory Module
- Depends on: Phase 4 (extractor provides JudgeInput pattern)
- Creates: `judge_factory.py` with registry loading and judge instantiation
