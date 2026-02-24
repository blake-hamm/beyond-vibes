"""Configuration models for the download utility."""

from pydantic import BaseModel

ESSENTIAL_MODEL_CONFIGS = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
}


class ModelConfig(BaseModel):
    """Configuration for a single model to download."""

    name: str
    repo_id: str
    quant_tags: list[str]
    revision: str = "main"


class Config(BaseModel):
    """Root configuration model."""

    bucket: str
    models: list[ModelConfig]
