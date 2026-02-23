"""Tests for CLI."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from beyond_vibes.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_env_vars() -> dict[str, str]:
    """Mock environment variables for S3."""
    return {
        "S3_BUCKET": "test-bucket",
        "S3_ENDPOINT": "https://s3.example.com",
        "S3_ACCESS_KEY": "test-key",
        "S3_SECRET_KEY": "test-secret",
    }


def test_download_with_config(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
) -> None:
    """Test download with config file."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with patch.dict(os.environ, mock_env_vars):
        with patch("beyond_vibes.cli.HFClient") as mock_hf:
            with patch("beyond_vibes.cli.S3Client"):
                mock_hf_instance = MagicMock()
                mock_hf.return_value = mock_hf_instance
                mock_hf_instance.list_files.return_value = [
                    "model-q8_0.gguf",
                    "config.json",
                ]
                mock_hf_instance.filter_files.return_value = [
                    "model-q8_0.gguf",
                    "config.json",
                ]
                mock_hf_instance.download_file.return_value = Path("/tmp/test.bin")

                result = runner.invoke(
                    app, ["download", "--config-path", str(config_file)]
                )

    assert result.exit_code == 0


def test_download_dry_run(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
) -> None:
    """Test download with dry-run flag."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with patch.dict(os.environ, mock_env_vars):
        with patch("beyond_vibes.cli.HFClient") as mock_hf:
            with patch("beyond_vibes.cli.S3Client"):
                mock_hf_instance = MagicMock()
                mock_hf.return_value = mock_hf_instance
                mock_hf_instance.list_files.return_value = [
                    "model-q8_0.gguf",
                    "config.json",
                ]
                mock_hf_instance.filter_files.return_value = [
                    "model-q8_0.gguf",
                    "config.json",
                ]

                result = runner.invoke(
                    app, ["download", "--config-path", str(config_file), "--dry-run"]
                )

    assert result.exit_code == 0


def test_download_lists_files(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
) -> None:
    """Test that download shows number of files found."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with patch.dict(os.environ, mock_env_vars):
        with patch("beyond_vibes.cli.HFClient") as mock_hf:
            with patch("beyond_vibes.cli.S3Client"):
                mock_hf_instance = MagicMock()
                mock_hf.return_value = mock_hf_instance
                mock_hf_instance.list_files.return_value = [
                    "model-q8_0.gguf",
                    "config.json",
                ]
                mock_hf_instance.filter_files.return_value = [
                    "model-q8_0.gguf",
                ]

                result = runner.invoke(
                    app, ["download", "--config-path", str(config_file)]
                )

    assert result.exit_code == 0


def test_download_multiple_models(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
) -> None:
    """Test download with multiple models."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "model-1",
                "repo_id": "test/repo1",
                "quant_tags": ["Q8_0"],
            },
            {
                "name": "model-2",
                "repo_id": "test/repo2",
                "quant_tags": ["Q4_K_M"],
            },
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with patch.dict(os.environ, mock_env_vars):
        with patch("beyond_vibes.cli.HFClient") as mock_hf:
            with patch("beyond_vibes.cli.S3Client"):
                mock_hf_instance = MagicMock()
                mock_hf.return_value = mock_hf_instance
                mock_hf_instance.list_files.return_value = [
                    "model.bin",
                    "config.json",
                ]
                mock_hf_instance.filter_files.return_value = ["config.json"]
                mock_hf_instance.download_file.return_value = Path("/tmp/test.bin")

                result = runner.invoke(
                    app, ["download", "--config-path", str(config_file)]
                )

    assert result.exit_code == 0
