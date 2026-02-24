"""Prompt loader for simulations."""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from beyond_vibes.simulations.models import SimulationConfig

logger = logging.getLogger(__name__)

TEMPLATE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Replace {{variable}} placeholders with values from variables dict."""

    def replace(match: re.Match) -> str:
        key = match.group(1)
        value = variables.get(key, match.group(0))
        return str(value)

    return TEMPLATE_PATTERN.sub(replace, template)


def load_prompt(
    path: Path,
    variables: dict[str, Any] | None = None,
) -> SimulationConfig:
    """Load YAML prompt file, render {{variables}} in prompt field, return config."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e

    if not data:
        raise ValueError(f"Empty prompt file: {path}")

    variables = variables or {}

    if "prompt" in data:
        data["prompt"] = render_template(data["prompt"], variables)

    try:
        config = SimulationConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid prompt config in {path}: {e}") from e

    logger.debug("Loaded prompt '%s' from %s", config.name, path)
    return config


def list_prompts(prompts_dir: Path) -> list[Path]:
    """Return all .yaml files recursively under prompts_dir."""
    if not prompts_dir.exists():
        logger.warning("Prompts directory does not exist: %s", prompts_dir)
        return []

    return list(prompts_dir.rglob("*.yaml"))
