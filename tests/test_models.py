"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from beyond_vibes.model_config import ESSENTIAL_MODEL_CONFIGS, Config, ModelConfig


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


def test_model_config_only_name_required() -> None:
    """Test that only name is required for API models."""
    # API model with just name should work
    config = ModelConfig(
        name="test-model",
        provider="openai",
    )
    assert config.name == "test-model"
    assert config.provider == "openai"


def test_model_config_local_requires_repo_id() -> None:
    """Test that local provider requires repo_id."""
    with pytest.raises(ValidationError) as exc_info:
        ModelConfig(
            name="test-model",
            provider="local",
            # repo_id is missing, should fail for local provider
        )
    error_msg = str(exc_info.value)
    assert "repo_id" in error_msg or "local" in error_msg.lower()


def test_model_config_api_no_repo_id() -> None:
    """Test that API provider works without repo_id."""
    config = ModelConfig(
        name="gpt-4o",
        provider="openai",
        model_id="gpt-4o",
    )
    assert config.name == "gpt-4o"
    assert config.provider == "openai"
    assert config.model_id == "gpt-4o"
    assert config.repo_id is None


def test_model_config_get_model_id_fallback() -> None:
    """Test that get_model_id falls back to name when model_id is not set."""
    config = ModelConfig(
        name="mistral-7b",
        repo_id="TheBloke/Mistral-7B-GGUF",
        quant_tags=[],
    )
    assert config.get_model_id() == "mistral-7b"


def test_model_config_get_model_id_explicit() -> None:
    """Test that get_model_id returns model_id when set."""
    config = ModelConfig(
        name="my-gpt4",
        provider="openai",
        model_id="gpt-4",
    )
    assert config.get_model_id() == "gpt-4"


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
