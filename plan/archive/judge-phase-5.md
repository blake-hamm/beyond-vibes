# Judge Phase 5: Judge Factory Module

## Overview

Implement the judge factory that creates judge instances from the registry. This bridges the declarative YAML configuration to actual MLflow judge objects.

## Prerequisites

- ✅ Phase 4: Extractor module with JudgeInput
- ✅ Phase 2: judges.yaml registry and JudgeMapping model
- MLflow judges available (built-in and DeepEval)

## Changes Required

### 1. Create Judge Factory Module

**File**: `src/beyond_vibes/evaluations/judge_factory.py`

**Upstream Dependencies**:
- Phase 2: judges.yaml registry format
- Phase 2: SimulationConfig with guidelines and judges fields
- Phase 2: Settings.judge_model for default model
- MLflow metrics API

**Downstream Impact**:
- Phase 6: Runner calls `build_judges_for_task()` and `create_judge()`
- Judges registry changes affect this module

**Implementation**:

```python
"""Judge factory for creating judge instances from registry.

This module loads the judges registry and creates MLflow judge instances
based on task configuration.

Example:
    from beyond_vibes.evaluations.judge_factory import create_judge, build_judges_for_task
    
    # Create single judge
    judge = create_judge("guidelines", task_config, "openai:/gpt-4o")
    
    # Build all judges for a task
    judges = build_judges_for_task(task_config)
"""

import importlib
import logging
from pathlib import Path
from typing import Any

import yaml

from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.settings import settings

logger = logging.getLogger(__name__)

# Load judges registry at module import time
_JUDGES_REGISTRY_PATH = Path(__file__).parent / "judges.yaml"


def _load_registry() -> dict:
    """Load judges registry from YAML file.
    
    Returns:
        Dictionary with judges configuration
    """
    try:
        with open(_JUDGES_REGISTRY_PATH) as f:
            data = yaml.safe_load(f)
            return data.get("judges", {})
    except Exception as e:
        logger.error(f"Failed to load judges registry: {e}")
        return {}


# Load registry once at import
_REGISTRY = _load_registry()


def create_judge(
    judge_name: str,
    task_config: SimulationConfig | None = None,
    judge_model: str | None = None,
) -> Any:
    """Create a judge instance from registry by name.
    
    Args:
        judge_name: Name of judge in registry (e.g., "guidelines")
        task_config: Task configuration (required for Guidelines judge)
        judge_model: Model to use (defaults to settings.judge_model)
        
    Returns:
        MLflow judge instance
        
    Raises:
        ValueError: If judge not found or configuration invalid
    """
    # Get judge configuration from registry
    judge_config = _REGISTRY.get(judge_name)
    if not judge_config:
        raise ValueError(f"Unknown judge: {judge_name}")
    
    # Use default model if not specified
    if judge_model is None:
        judge_model = settings.judge_model
    
    # Route to appropriate factory function
    judge_type = judge_config.get("type")
    
    if judge_type == "mlflow.builtin":
        return _create_mlflow_builtin_judge(judge_config, judge_name, task_config, judge_model)
    elif judge_type == "mlflow.third_party":
        return _create_mlflow_third_party_judge(judge_config, judge_name, judge_model)
    elif judge_type == "custom":
        return _create_custom_judge(judge_config, judge_name, task_config, judge_model)
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


def _create_mlflow_builtin_judge(
    config: dict,
    judge_name: str,
    task_config: SimulationConfig | None,
    judge_model: str,
) -> Any:
    """Create MLflow built-in judge.
    
    Args:
        config: Judge configuration from registry
        judge_name: Name of judge
        task_config: Task configuration
        judge_model: Model to use
        
    Returns:
        MLflow judge instance
    """
    from mlflow.metrics import Guidelines, ToolCallEfficiency
    
    judge_class = config.get("class")
    
    if judge_class == "Guidelines":
        # Guidelines judge requires task-specific configuration
        if not task_config:
            raise ValueError(f"Judge '{judge_name}' requires task_config")
        if not task_config.guidelines:
            raise ValueError(f"Judge '{judge_name}' requires task guidelines")
        
        return Guidelines(
            name=f"{task_config.name}_guidelines",
            guidelines=task_config.guidelines,
            model=judge_model,
        )
    
    elif judge_class == "ToolCallEfficiency":
        # ToolCallEfficiency uses default configuration
        return ToolCallEfficiency(model=judge_model)
    
    else:
        raise ValueError(f"Unknown built-in judge class: {judge_class}")


def _create_mlflow_third_party_judge(
    config: dict,
    judge_name: str,
    judge_model: str,
) -> Any:
    """Create MLflow third-party judge (e.g., DeepEval).
    
    Dynamically imports the judge class from specified module.
    
    Args:
        config: Judge configuration from registry
        judge_name: Name of judge
        judge_model: Model to use
        
    Returns:
        MLflow judge instance
    """
    module_path = config.get("module")
    class_name = config.get("class")
    
    if not module_path or not class_name:
        raise ValueError(
            f"Judge '{judge_name}' missing module or class in config"
        )
    
    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)
        judge_class = getattr(module, class_name)
        
        # Instantiate the judge
        return judge_class(model=judge_model)
    
    except ImportError as e:
        raise ValueError(
            f"Failed to import module '{module_path}' for judge '{judge_name}': {e}"
        )
    except AttributeError as e:
        raise ValueError(
            f"Class '{class_name}' not found in module '{module_path}': {e}"
        )


def _create_custom_judge(
    config: dict,
    judge_name: str,
    task_config: SimulationConfig | None,
    judge_model: str,
) -> Any:
    """Create custom judge from factory function.
    
    Future extension point for custom evaluation logic.
    
    Args:
        config: Judge configuration from registry
        judge_name: Name of judge
        task_config: Task configuration
        judge_model: Model to use
        
    Returns:
        Custom judge instance
        
    Raises:
        NotImplementedError: Custom judges not yet supported
    """
    raise NotImplementedError(
        f"Custom judges not yet implemented (requested: {judge_name})"
    )


def build_judges_for_task(
    task_config: SimulationConfig,
    judge_model: str | None = None,
) -> list[tuple[Any, str]]:
    """Build all judges specified in task config.
    
    Creates judge instances with their input artifact mappings.
    
    Args:
        task_config: Task configuration with judges list
        judge_model: Model to use (defaults to settings.judge_model)
        
    Returns:
        List of (judge_instance, input_artifact) tuples
        
    Example:
        >>> judges = build_judges_for_task(task_config)
        >>> for judge, input_artifact in judges:
        ...     print(f"{judge.name} evaluates {input_artifact}")
    """
    if not task_config.judges:
        logger.warning(f"No judges configured for task '{task_config.name}'")
        return []
    
    judges = []
    
    for mapping in task_config.judges:
        try:
            # Create the judge instance
            judge = create_judge(mapping.name, task_config, judge_model)
            
            if judge:
                # Pair with input artifact from mapping
                judges.append((judge, mapping.input))
                logger.debug(
                    f"Created judge '{mapping.name}' for task '{task_config.name}'"
                )
        
        except ValueError as e:
            logger.warning(f"Skipping judge '{mapping.name}': {e}")
        except Exception as e:
            logger.error(f"Failed to create judge '{mapping.name}': {e}")
    
    return judges


def list_available_judges() -> dict[str, str]:
    """List all available judges from registry.
    
    Returns:
        Dictionary mapping judge names to descriptions
    """
    return {
        name: config.get("description", "No description")
        for name, config in _REGISTRY.items()
    }
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

__all__ = [
    "extract_run_data",
    "query_simulation_runs",
    "JudgeInput",
    "create_judge",
    "build_judges_for_task",
    "list_available_judges",
]
```

---

## Dependencies

### Required Dependencies

Already covered in Phase 2 (pyproject.toml updates):
```toml
dependencies = [
    "mlflow[genai]>=3.10.0",  # For built-in and third-party judges
    "deepeval>=0.21.0",  # DeepEval POC
]
```

### Phase Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| Phase 5 | Phase 4 | Uses JudgeInput type hints (optional) |
| Phase 5 | Phase 2 | Depends on judges.yaml, JudgeMapping, Settings |
| Phase 6 | Phase 5 | Runner uses `build_judges_for_task()` and `create_judge()` |

### File Dependencies

```
Phase 4 outputs:
  ├── src/beyond_vibes/evaluations/models.py
  └── src/beyond_vibes/evaluations/extractor.py
        ↓
Phase 2 outputs:
  ├── src/beyond_vibes/evaluations/judges.yaml
  ├── src/beyond_vibes/simulations/models.py (JudgeMapping)
  └── src/beyond_vibes/settings.py
        ↓
Phase 5 creates:
  └── src/beyond_vibes/evaluations/judge_factory.py
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/evaluations/test_judge_factory.py

import pytest
from unittest.mock import Mock, patch, mock_open

from beyond_vibes.evaluations.judge_factory import (
    create_judge,
    build_judges_for_task,
    _create_mlflow_builtin_judge,
    _create_mlflow_third_party_judge,
    _REGISTRY,
)
from beyond_vibes.simulations.models import SimulationConfig, JudgeMapping, RepositoryConfig


class TestCreateJudge:
    """Test judge creation from registry."""
    
    def test_unknown_judge_raises_error(self):
        """Test that unknown judge names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown judge"):
            create_judge("nonexistent_judge")
    
    def test_guidelines_judge_requires_task_config(self):
        """Test Guidelines judge requires task configuration."""
        with patch.dict(_REGISTRY, {"guidelines": {"type": "mlflow.builtin", "class": "Guidelines"}}):
            with pytest.raises(ValueError, match="requires task_config"):
                create_judge("guidelines", task_config=None)
    
    def test_guidelines_judge_requires_guidelines(self):
        """Test Guidelines judge requires non-empty guidelines."""
        task_config = Mock(spec=SimulationConfig)
        task_config.guidelines = []
        
        with patch.dict(_REGISTRY, {"guidelines": {"type": "mlflow.builtin", "class": "Guidelines"}}):
            with pytest.raises(ValueError, match="requires task guidelines"):
                create_judge("guidelines", task_config=task_config)


class TestBuildJudgesForTask:
    """Test building judges for a task."""
    
    def test_empty_judges_list(self):
        """Test that empty judges list returns empty list."""
        task_config = Mock(spec=SimulationConfig)
        task_config.name = "test"
        task_config.judges = []
        
        result = build_judges_for_task(task_config)
        assert result == []
    
    def test_build_multiple_judges(self):
        """Test building multiple judges with input mappings."""
        task_config = Mock(spec=SimulationConfig)
        task_config.name = "test"
        task_config.guidelines = ["Criterion 1"]
        task_config.judges = [
            JudgeMapping(name="guidelines", input="git_diff"),
            JudgeMapping(name="tool_efficiency", input="trace"),
        ]
        
        with patch("beyond_vibes.evaluations.judge_factory.create_judge") as mock_create:
            mock_judge1 = Mock()
            mock_judge1.name = "test_guidelines"
            mock_judge2 = Mock()
            mock_judge2.name = "tool_efficiency"
            
            mock_create.side_effect = [mock_judge1, mock_judge2]
            
            result = build_judges_for_task(task_config)
            
            assert len(result) == 2
            assert result[0] == (mock_judge1, "git_diff")
            assert result[1] == (mock_judge2, "trace")


class TestRegistryLoading:
    """Test judges registry loading."""
    
    def test_registry_loads_yaml(self):
        """Test that registry loads from YAML file."""
        yaml_content = """
judges:
  test_judge:
    type: mlflow.builtin
    class: Guidelines
    description: "Test judge"
"""
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch("yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {
                    "judges": {
                        "test_judge": {
                            "type": "mlflow.builtin",
                            "class": "Guidelines",
                            "description": "Test judge",
                        }
                    }
                }
                # Trigger reload
                from beyond_vibes.evaluations.judge_factory import _load_registry
                registry = _load_registry()
                
                assert "test_judge" in registry
```

### Integration Tests

```python
# tests/integration/test_judge_factory_integration.py

import pytest

pytestmark = pytest.mark.integration


def test_create_real_guidelines_judge():
    """Test creating a real Guidelines judge."""
    from beyond_vibes.evaluations import create_judge
    from beyond_vibes.simulations.models import SimulationConfig, JudgeMapping, RepositoryConfig
    
    task_config = SimulationConfig(
        name="test_task",
        description="Test",
        archetype="test",
        repository=RepositoryConfig(url="http://example.com"),
        prompt="Test prompt",
        guidelines=["Criterion 1", "Criterion 2"],
    )
    
    judge = create_judge("guidelines", task_config, "openai:/gpt-4o-mini")
    
    assert judge is not None
    assert judge.name == "test_task_guidelines"


def test_create_real_deepeval_judge():
    """Test creating a real DeepEval judge (if available)."""
    from beyond_vibes.evaluations import create_judge
    
    try:
        judge = create_judge("faithfulness", judge_model="openai:/gpt-4o-mini")
        assert judge is not None
    except ValueError as e:
        pytest.skip(f"DeepEval not available: {e}")
```

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Unknown judge: X` | Judge not in registry | Add to judges.yaml or check spelling |
| `requires task_config` | Guidelines judge needs config | Pass task_config to create_judge |
| `requires task guidelines` | Guidelines list is empty | Add guidelines to task YAML |
| `Failed to import module` | DeepEval not installed | Install deepeval package |
| `Class 'X' not found` | Wrong class name in registry | Check class name in judges.yaml |

---

## Success Criteria

- [ ] `judge_factory.py` exists with all factory functions
- [ ] Can create Guidelines judge with task config
- [ ] Can create ToolCallEfficiency judge
- [ ] Can create DeepEval judges (faithfulness, answer_relevancy)
- [ ] `build_judges_for_task()` returns list of (judge, input) tuples
- [ ] Unknown judges raise ValueError with clear message
- [ ] Judges without required config raise ValueError
- [ ] `__init__.py` exports factory functions
- [ ] All unit tests pass
- [ ] Integration tests with real judges work

---

## Next Phase

**Phase 6**: Runner Module
- Depends on: Phase 5 (factory creates judges)
- Creates: `runner.py` with EvaluationRunner orchestration class
