"""Tests for Settings."""

import os
from unittest.mock import patch

from beyond_vibes.settings import Settings


def test_settings_from_env() -> None:
    """Test that Settings loads from environment variables."""
    with patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "DEBUG",
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
            "S3_ACCESS_KEY": "test-access-key",
            "S3_SECRET_KEY": "test-secret-key",
        },
        clear=True,
    ):
        settings = Settings()
        assert settings.log_level == "DEBUG"
        assert settings.s3_bucket == "test-bucket"
        assert settings.s3_endpoint == "https://s3.example.com"
        assert settings.s3_access_key == "test-access-key"
        assert settings.s3_secret_key == "test-secret-key"


def test_settings_log_level_normalized() -> None:
    """Test that log level is normalized to uppercase."""
    with patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "debug",
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
            "S3_ACCESS_KEY": "test-access-key",
            "S3_SECRET_KEY": "test-secret-key",
        },
        clear=True,
    ):
        settings = Settings()
        assert settings.log_level == "DEBUG"


def test_settings_default_log_level() -> None:
    """Test that default log level is INFO."""
    with patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "INFO",
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
            "S3_ACCESS_KEY": "test-access-key",
            "S3_SECRET_KEY": "test-secret-key",
        },
        clear=True,
    ):
        settings = Settings()
        assert settings.log_level == "INFO"


def test_judge_settings_defaults() -> None:
    """Test default judge settings."""
    with patch.dict(
        os.environ,
        {
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
            "S3_ACCESS_KEY": "test-access-key",
            "S3_SECRET_KEY": "test-secret-key",
            "JUDGE_API_KEY": "",  # Override any .env file value
            "JUDGE_BASE_URL": "",  # Override any .env file value
        },
        clear=True,
    ):
        settings = Settings()
        assert settings.judge_model == "openai:/openai/gpt-4o-mini"
        assert settings.judge_api_key is None
        assert settings.judge_base_url is None


def test_judge_settings_from_env() -> None:
    """Test loading judge settings from environment."""
    with patch.dict(
        os.environ,
        {
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
            "S3_ACCESS_KEY": "test-access-key",
            "S3_SECRET_KEY": "test-secret-key",
            "JUDGE_MODEL": "openai:/gpt-4o",
            "JUDGE_BASE_URL": "http://localhost:8000/v1",
        },
        clear=True,
    ):
        settings = Settings()
        assert settings.judge_model == "openai:/gpt-4o"
        assert settings.judge_base_url == "http://localhost:8000/v1"
