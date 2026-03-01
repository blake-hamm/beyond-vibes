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
                "provider": "local",
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
                "provider": "local",
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
                "provider": "local",
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
                "provider": "local",
                "quant_tags": ["Q8_0"],
            },
            {
                "name": "model-2",
                "repo_id": "test/repo2",
                "provider": "local",
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


def test_debug_flag_sets_logging_level(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
) -> None:
    """Test that --debug flag sets logging level to DEBUG."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
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
                mock_hf_instance.list_files.return_value = ["model.bin"]
                mock_hf_instance.filter_files.return_value = ["model.bin"]
                mock_hf_instance.download_file.return_value = Path("/tmp/test.bin")

                with patch("logging.getLogger") as mock_get_logger:
                    mock_logger = MagicMock()
                    mock_get_logger.return_value = mock_logger

                    result = runner.invoke(
                        app,
                        ["--debug", "download", "--config-path", str(config_file)],
                    )

    assert result.exit_code == 0
    mock_get_logger.assert_called_once_with("beyond_vibes")
    mock_logger.setLevel.assert_called_once_with(10)


def test_download_skips_models_without_repo_id(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that download skips models without repo_id (e.g., API providers)."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "local-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            },
            {
                "name": "api-model",
                "provider": "openai",
                "model_id": "gpt-4",
            },
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with caplog.at_level("INFO"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.HFClient") as mock_hf:
                with patch("beyond_vibes.cli.S3Client"):
                    mock_hf_instance = MagicMock()
                    mock_hf.return_value = mock_hf_instance
                    mock_hf_instance.list_files.return_value = ["model.bin"]
                    mock_hf_instance.filter_files.return_value = ["model.bin"]
                    mock_hf_instance.download_file.return_value = Path("/tmp/test.bin")

                    result = runner.invoke(
                        app, ["download", "--config-path", str(config_file)]
                    )

    assert result.exit_code == 0
    assert "Skipping api-model" in caplog.text
    assert "provider 'openai' does not require download" in caplog.text
    mock_hf_instance.list_files.assert_called_once()


def test_download_list_files_error(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that download exits with error when list_files fails."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with caplog.at_level("ERROR"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.HFClient") as mock_hf:
                with patch("beyond_vibes.cli.S3Client"):
                    mock_hf_instance = MagicMock()
                    mock_hf.return_value = mock_hf_instance
                    mock_hf_instance.list_files.side_effect = Exception(
                        "Connection failed"
                    )

                    result = runner.invoke(
                        app, ["download", "--config-path", str(config_file)]
                    )

    assert result.exit_code == 1
    assert "Failed to list files for test-model" in caplog.text
    assert "Connection failed" in caplog.text


def test_download_download_file_error(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that download exits with error when download_file fails."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with caplog.at_level("ERROR"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.HFClient") as mock_hf:
                with patch("beyond_vibes.cli.S3Client"):
                    mock_hf_instance = MagicMock()
                    mock_hf.return_value = mock_hf_instance
                    mock_hf_instance.list_files.return_value = ["model.bin"]
                    mock_hf_instance.filter_files.return_value = ["model.bin"]
                    mock_hf_instance.download_file.side_effect = Exception(
                        "Download failed"
                    )

                    result = runner.invoke(
                        app, ["download", "--config-path", str(config_file)]
                    )

    assert result.exit_code == 1
    assert "Failed to download model.bin" in caplog.text
    assert "Download failed" in caplog.text


def test_download_upload_file_error(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that download exits with error when S3 upload fails."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with caplog.at_level("ERROR"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.HFClient") as mock_hf:
                with patch("beyond_vibes.cli.S3Client") as mock_s3:
                    mock_hf_instance = MagicMock()
                    mock_hf.return_value = mock_hf_instance
                    mock_hf_instance.list_files.return_value = ["model.bin"]
                    mock_hf_instance.filter_files.return_value = ["model.bin"]
                    mock_hf_instance.download_file.return_value = Path("/tmp/test.bin")

                    mock_s3_instance = MagicMock()
                    mock_s3.return_value = mock_s3_instance
                    mock_s3_instance.upload_file.side_effect = Exception(
                        "S3 upload failed"
                    )

                    result = runner.invoke(
                        app, ["download", "--config-path", str(config_file)]
                    )

    assert result.exit_code == 1
    assert "Failed to upload model.bin" in caplog.text
    assert "S3 upload failed" in caplog.text


def test_simulate_task_not_found(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that simulate exits with error when task file is not found."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with caplog.at_level("ERROR"):
        with patch.dict(os.environ, mock_env_vars):
            with patch(
                "beyond_vibes.cli.load_task_config",
                side_effect=FileNotFoundError("Task not found"),
            ):
                result = runner.invoke(
                    app,
                    [
                        "simulate",
                        "--task",
                        "nonexistent_task",
                        "--model",
                        "test-model",
                        "--config-path",
                        str(config_file),
                    ],
                )

    assert result.exit_code == 1
    assert "Task not found: nonexistent_task" in caplog.text


def test_simulate_invalid_prompt_vars(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that simulate exits with error when prompt_vars contains invalid JSON."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with caplog.at_level("ERROR"):
        with patch.dict(os.environ, mock_env_vars):
            result = runner.invoke(
                app,
                [
                    "simulate",
                    "--task",
                    "unit_tests",
                    "--model",
                    "test-model",
                    "--config-path",
                    str(config_file),
                    "--prompt-vars",
                    "invalid json",
                ],
            )

    assert result.exit_code == 1
    assert "Failed to load prompt" in caplog.text


def test_simulate_model_not_found(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that simulate exits with error when model is not found in config."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    with caplog.at_level("ERROR"):
        with patch.dict(os.environ, mock_env_vars):
            result = runner.invoke(
                app,
                [
                    "simulate",
                    "--task",
                    "unit_tests",
                    "--model",
                    "nonexistent-model",
                    "--config-path",
                    str(config_file),
                ],
            )

    assert result.exit_code == 1
    assert "nonexistent-model' not found in config" in caplog.text


def test_simulate_successful_execution(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test successful simulate command execution."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    sim_config = MagicMock()
    sim_config.name = "unit_tests"
    sim_config.repository = MagicMock()
    sim_config.repository.url = "https://github.com/test/repo"
    sim_config.repository.branch = "main"
    sim_config.agent = "build"
    sim_config.max_turns = 75
    sim_config.prompt = "Test prompt"
    sim_config.system_prompt = None

    with caplog.at_level("INFO"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.load_task_config", return_value=sim_config):
                with patch("beyond_vibes.cli.build_prompt", return_value="Test prompt"):
                    with patch("beyond_vibes.cli.SandboxManager") as mock_sandbox_class:
                        mock_sandbox = MagicMock()
                        mock_sandbox_class.return_value = mock_sandbox

                        with patch(
                            "beyond_vibes.cli.OpenCodeClient"
                        ) as mock_opencode_class:
                            mock_opencode = MagicMock()
                            mock_opencode_class.return_value.__enter__ = MagicMock(
                                return_value=mock_opencode
                            )
                            mock_opencode_class.return_value.__exit__ = MagicMock(
                                return_value=False
                            )

                            with patch(
                                "beyond_vibes.cli.SimulationLogger"
                            ) as mock_logger_class:
                                mock_logger = MagicMock()
                                mock_logger_class.return_value = mock_logger

                                with patch(
                                    "beyond_vibes.cli.run_simulation",
                                    return_value=False,
                                ):
                                    result = runner.invoke(
                                        app,
                                        [
                                            "simulate",
                                            "--task",
                                            "unit_tests",
                                            "--model",
                                            "test-model",
                                            "--config-path",
                                            str(config_file),
                                        ],
                                    )

    assert result.exit_code == 0
    assert "Running simulation with model: test-model" in caplog.text
    assert "Sandbox cleaned up" in caplog.text
    mock_sandbox.cleanup.assert_called_once()


def test_simulate_error_occurred(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that simulate exits with error when simulation reports an error."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    sim_config = MagicMock()
    sim_config.name = "unit_tests"
    sim_config.repository = MagicMock()
    sim_config.repository.url = "https://github.com/test/repo"
    sim_config.repository.branch = "main"
    sim_config.agent = "build"
    sim_config.max_turns = 75
    sim_config.prompt = "Test prompt"
    sim_config.system_prompt = None

    with caplog.at_level("INFO"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.load_task_config", return_value=sim_config):
                with patch("beyond_vibes.cli.build_prompt", return_value="Test prompt"):
                    with patch("beyond_vibes.cli.SandboxManager") as mock_sandbox_class:
                        mock_sandbox = MagicMock()
                        mock_sandbox_class.return_value = mock_sandbox

                        with patch(
                            "beyond_vibes.cli.OpenCodeClient"
                        ) as mock_opencode_class:
                            mock_opencode = MagicMock()
                            mock_opencode_class.return_value.__enter__ = MagicMock(
                                return_value=mock_opencode
                            )
                            mock_opencode_class.return_value.__exit__ = MagicMock(
                                return_value=False
                            )

                            with patch(
                                "beyond_vibes.cli.SimulationLogger"
                            ) as mock_logger_class:
                                mock_logger = MagicMock()
                                mock_logger_class.return_value = mock_logger

                                with patch(
                                    "beyond_vibes.cli.run_simulation",
                                    return_value=True,
                                ):
                                    result = runner.invoke(
                                        app,
                                        [
                                            "simulate",
                                            "--task",
                                            "unit_tests",
                                            "--model",
                                            "test-model",
                                            "--config-path",
                                            str(config_file),
                                        ],
                                    )

    assert result.exit_code == 1
    assert "Running simulation with model: test-model" in caplog.text
    assert "Sandbox cleaned up" in caplog.text
    mock_sandbox.cleanup.assert_called_once()


def test_simulate_provider_filter(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test simulate command with provider filter for multiple models with same name."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo1",
                "provider": "local",
                "quant_tags": ["Q8_0"],
            },
            {
                "name": "test-model",
                "provider": "openai",
                "model_id": "gpt-4",
            },
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    sim_config = MagicMock()
    sim_config.name = "unit_tests"
    sim_config.repository = MagicMock()
    sim_config.repository.url = "https://github.com/test/repo"
    sim_config.repository.branch = "main"
    sim_config.agent = "build"
    sim_config.max_turns = 75
    sim_config.prompt = "Test prompt"
    sim_config.system_prompt = None

    with caplog.at_level("INFO"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.load_task_config", return_value=sim_config):
                with patch("beyond_vibes.cli.build_prompt", return_value="Test prompt"):
                    with patch("beyond_vibes.cli.SandboxManager") as mock_sandbox_class:
                        mock_sandbox = MagicMock()
                        mock_sandbox_class.return_value = mock_sandbox

                        with patch(
                            "beyond_vibes.cli.OpenCodeClient"
                        ) as mock_opencode_class:
                            mock_opencode = MagicMock()
                            mock_opencode_class.return_value.__enter__ = MagicMock(
                                return_value=mock_opencode
                            )
                            mock_opencode_class.return_value.__exit__ = MagicMock(
                                return_value=False
                            )

                            with patch(
                                "beyond_vibes.cli.SimulationLogger"
                            ) as mock_logger_class:
                                mock_logger = MagicMock()
                                mock_logger_class.return_value = mock_logger

                                with patch(
                                    "beyond_vibes.cli.run_simulation",
                                    return_value=False,
                                ):
                                    result = runner.invoke(
                                        app,
                                        [
                                            "simulate",
                                            "--task",
                                            "unit_tests",
                                            "--model",
                                            "test-model",
                                            "--provider",
                                            "openai",
                                            "--config-path",
                                            str(config_file),
                                        ],
                                    )

    assert result.exit_code == 0
    assert "Running simulation with model: test-model" in caplog.text
    assert "Sandbox cleaned up" in caplog.text


def test_simulate_custom_quant_tag(
    runner: CliRunner,
    mock_env_vars: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test simulate command with custom quant tag."""
    config = {
        "bucket": "test-bucket",
        "models": [
            {
                "name": "test-model",
                "repo_id": "test/repo",
                "provider": "local",
                "quant_tags": ["Q8_0", "Q4_K_M"],
            }
        ],
    }
    config_file = tmp_path / "models.yaml"
    config_file.write_text(yaml.dump(config))

    sim_config = MagicMock()
    sim_config.name = "unit_tests"
    sim_config.repository = MagicMock()
    sim_config.repository.url = "https://github.com/test/repo"
    sim_config.repository.branch = "main"
    sim_config.agent = "build"
    sim_config.max_turns = 75
    sim_config.prompt = "Test prompt"
    sim_config.system_prompt = None

    with caplog.at_level("INFO"):
        with patch.dict(os.environ, mock_env_vars):
            with patch("beyond_vibes.cli.load_task_config", return_value=sim_config):
                with patch("beyond_vibes.cli.build_prompt", return_value="Test prompt"):
                    with patch("beyond_vibes.cli.SandboxManager") as mock_sandbox_class:
                        mock_sandbox = MagicMock()
                        mock_sandbox_class.return_value = mock_sandbox

                        with patch(
                            "beyond_vibes.cli.OpenCodeClient"
                        ) as mock_opencode_class:
                            mock_opencode = MagicMock()
                            mock_opencode_class.return_value.__enter__ = MagicMock(
                                return_value=mock_opencode
                            )
                            mock_opencode_class.return_value.__exit__ = MagicMock(
                                return_value=False
                            )

                            with patch(
                                "beyond_vibes.cli.SimulationLogger"
                            ) as mock_logger_class:
                                mock_logger = MagicMock()
                                mock_logger_class.return_value = mock_logger

                                with patch(
                                    "beyond_vibes.cli.run_simulation",
                                    return_value=False,
                                ):
                                    result = runner.invoke(
                                        app,
                                        [
                                            "simulate",
                                            "--task",
                                            "unit_tests",
                                            "--model",
                                            "test-model",
                                            "--config-path",
                                            str(config_file),
                                            "--quant",
                                            "Q4_K_M",
                                        ],
                                    )

    assert result.exit_code == 0
    assert "Running simulation with model: test-model" in caplog.text
    assert "Sandbox cleaned up" in caplog.text
    mock_logger_class.assert_called_once_with(quant_tag="Q4_K_M")
