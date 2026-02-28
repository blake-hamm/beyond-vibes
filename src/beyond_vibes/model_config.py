"""Shared model configuration models."""

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

ESSENTIAL_MODEL_CONFIGS = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
}

DEFAULT_CONFIG_PATH = "models.yaml"


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    name: str
    repo_id: str | None = None
    provider: str
    model_id: str | None = None
    quant_tags: list[str] = []
    revision: str = "main"

    @model_validator(mode="after")
    def validate_local_requires_repo_id(self) -> "ModelConfig":
        """Ensure local provider has a repo_id."""
        if self.provider == "local" and self.repo_id is None:
            raise ValueError("Local provider requires repo_id")
        return self

    def get_model_id(self) -> str:
        """Get the model ID, falling back to name if not specified."""
        return self.model_id or self.name


class Config(BaseModel):
    """Root configuration model."""

    bucket: str
    models: list[ModelConfig]


def load_models_config(path: Path | None = None) -> Config:
    """Load and validate models.yaml.

    Args:
        path: Path to the models.yaml file. Defaults to DEFAULT_CONFIG_PATH.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config is invalid.

    """
    if path is None:
        path = Path(DEFAULT_CONFIG_PATH)

    config_data = yaml.safe_load(path.read_text())
    return Config(**config_data)
