"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from beyond_vibes.config import ESSENTIAL_MODEL_CONFIGS, Config, ModelConfig


def test_model_config_valid() -> None:
    """Test that valid ModelConfig parses correctly."""
    config = ModelConfig(
        name="mistral-7b",
        repo_id="TheBloke/Mistral-7B-GGUF",
        quant_tags=["Q8_0", "Q4_K_M"],
    )
    assert config.name == "mistral-7b"
    assert config.repo_id == "TheBloke/Mistral-7B-GGUF"
    assert config.quant_tags == ["Q8_0", "Q4_K_M"]
    assert config.revision == "main"


def test_model_config_custom_revision() -> None:
    """Test that custom revision is parsed correctly."""
    config = ModelConfig(
        name="test-model",
        repo_id="test/repo",
        quant_tags=["Q4_K_M"],
        revision="v1.0",
    )
    assert config.revision == "v1.0"


def test_model_config_missing_required_fields() -> None:
    """Test that ValidationError is raised for missing required fields."""
    with pytest.raises(ValidationError):
        ModelConfig(name="test-model")


def test_model_config_empty_quant_tags() -> None:
    """Test that empty quant_tags is allowed."""
    config = ModelConfig(
        name="test-model",
        repo_id="test/repo",
        quant_tags=[],
    )
    assert config.quant_tags == []


def test_config_valid() -> None:
    """Test that valid Config parses correctly."""
    config = Config(
        bucket="my-models",
        models=[
            ModelConfig(
                name="mistral-7b",
                repo_id="TheBloke/Mistral-7B-GGUF",
                quant_tags=["Q8_0"],
            ),
        ],
    )
    assert config.bucket == "my-models"
    assert len(config.models) == 1


def test_config_missing_bucket() -> None:
    """Test that ValidationError is raised when bucket is missing."""
    with pytest.raises(ValidationError):
        Config(models=[])


def test_config_missing_models() -> None:
    """Test that ValidationError is raised when models is missing."""
    with pytest.raises(ValidationError):
        Config(bucket="test-bucket")


def test_config_multiple_models() -> None:
    """Test that Config parses multiple models correctly."""
    config = Config(
        bucket="my-models",
        models=[
            ModelConfig(
                name="mistral-7b",
                repo_id="TheBloke/Mistral-7B-GGUF",
                quant_tags=["Q8_0"],
            ),
            ModelConfig(
                name="llama-13b",
                repo_id="TheBloke/Llama-2-13B-GGUF",
                quant_tags=["Q4_K_M"],
            ),
        ],
    )
    assert len(config.models) == len(config.models)
    assert config.models[0].name == "mistral-7b"
    assert config.models[1].name == "llama-13b"


def test_essential_model_configs() -> None:
    """Test that ESSENTIAL_MODEL_CONFIGS contains expected files."""
    expected = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    }
    assert ESSENTIAL_MODEL_CONFIGS == expected
