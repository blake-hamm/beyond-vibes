"""Judge factory for creating judge instances from registry.

This module loads the judges registry and creates MLflow judge instances
based on task configuration.

Example:
    from beyond_vibes.evaluations.judge_factory import (
        create_judge,
        build_judges_for_task,
    )

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

from beyond_vibes.settings import settings
from beyond_vibes.simulations.models import SimulationConfig

logger = logging.getLogger(__name__)

# Load judges registry at module import time
_JUDGES_REGISTRY_PATH = Path(__file__).parent / "judges.yaml"


def _load_registry() -> dict:
    """Load judges registry from YAML file.

    Returns:
        Dictionary with judges configuration

    """
    try:
        with _JUDGES_REGISTRY_PATH.open() as f:
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
) -> Any:  # noqa: ANN401
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
        return _create_mlflow_builtin_judge(
            judge_config, judge_name, task_config, judge_model
        )
    if judge_type == "mlflow.third_party":
        return _create_mlflow_third_party_judge(judge_config, judge_name, judge_model)
    if judge_type == "custom":
        return _create_custom_judge(judge_config, judge_name, task_config, judge_model)
    raise ValueError(f"Unknown judge type: {judge_type}")


def _create_mlflow_builtin_judge(
    config: dict,
    judge_name: str,
    task_config: SimulationConfig | None,
    judge_model: str,
) -> Any:  # noqa: ANN401
    """Create MLflow built-in judge.

    Args:
        config: Judge configuration from registry
        judge_name: Name of judge
        task_config: Task configuration
        judge_model: Model to use

    Returns:
        MLflow judge instance

    """
    from mlflow.genai.scorers import (  # noqa: PLC0415
        Guidelines,
        ToolCallEfficiency,
    )

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

    if judge_class == "ToolCallEfficiency":
        # ToolCallEfficiency uses default configuration
        return ToolCallEfficiency(model=judge_model)

    raise ValueError(f"Unknown built-in judge class: {judge_class}")


def _create_mlflow_third_party_judge(
    config: dict,
    judge_name: str,
    judge_model: str,
) -> Any:  # noqa: ANN401
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
        raise ValueError(f"Judge '{judge_name}' missing module or class in config")

    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)
        judge_class = getattr(module, class_name)

        # Instantiate the judge
        return judge_class(model=judge_model)

    except ImportError as e:
        raise ValueError(
            f"Failed to import module '{module_path}' for judge '{judge_name}': {e}"
        ) from e
    except AttributeError as e:
        raise ValueError(
            f"Class '{class_name}' not found in module '{module_path}': {e}"
        ) from e


def _create_custom_judge(
    config: dict,
    judge_name: str,
    task_config: SimulationConfig | None,
    judge_model: str,
) -> Any:  # noqa: ANN401
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
