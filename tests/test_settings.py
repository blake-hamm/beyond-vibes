"""Tests for S3Settings."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from beyond_vibes.settings import S3Settings


def test_s3_settings_from_env() -> None:
    """Test that S3Settings loads from environment variables."""
    with patch.dict(
        os.environ,
        {
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
            "S3_ACCESS_KEY": "test-access-key",
            "S3_SECRET_KEY": "test-secret-key",
        },
        clear=True,
    ):
        settings = S3Settings()
        assert settings.bucket == "test-bucket"
        assert settings.endpoint == "https://s3.example.com"
        assert settings.access_key == "test-access-key"
        assert settings.secret_key == "test-secret-key"


def test_s3_settings_missing_required_fields() -> None:
    """Test that ValidationError is raised when required fields are missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError):
            S3Settings()


def test_s3_settings_partial_env_vars() -> None:
    """Test that ValidationError is raised when some required fields are missing."""
    with patch.dict(
        os.environ,
        {
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
        },
        clear=True,
    ):
        with pytest.raises(ValidationError):
            S3Settings()
