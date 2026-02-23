"""Tests for HuggingFace client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beyond_vibes.hf import HFClient


@pytest.fixture
def hf_client() -> HFClient:
    """Create HFClient with mocked HfApi."""
    with patch("beyond_vibes.hf.HfApi") as mock_api:
        mock_client = MagicMock()
        mock_api.return_value = mock_client
        client = HFClient()
        client._api = mock_client
        return client


def test_list_files(hf_client: HFClient) -> None:
    """Test listing files from a repository."""
    mock_files = ["model.bin", "config.json", "tokenizer.json"]
    hf_client._api.list_repo_files.return_value = mock_files

    result = hf_client.list_files("test/repo", "main")

    hf_client._api.list_repo_files.assert_called_once_with(
        repo_id="test/repo", revision="main"
    )
    assert result == mock_files


def test_list_files_custom_revision(hf_client: HFClient) -> None:
    """Test listing files with custom revision."""
    hf_client._api.list_repo_files.return_value = []

    hf_client.list_files("test/repo", "v1.0")

    hf_client._api.list_repo_files.assert_called_once_with(
        repo_id="test/repo", revision="v1.0"
    )


def test_filter_files_quant_tags(hf_client: HFClient) -> None:
    """Test filtering files by quant tags."""
    files = [
        "model-q8_0.gguf",
        "model-q4_k_m.gguf",
        "other-file.bin",
    ]

    result = hf_client.filter_files(files, ["Q8_0", "Q4_K_M"])

    assert "model-q8_0.gguf" in result
    assert "model-q4_k_m.gguf" in result
    assert "other-file.bin" not in result


def test_filter_files_essential_configs(hf_client: HFClient) -> None:
    """Test that essential config files are always included."""
    files = ["model.bin", "config.json", "tokenizer.json", "other.bin"]

    result = hf_client.filter_files(files, [])

    assert "config.json" in result
    assert "tokenizer.json" in result
    assert "model.bin" not in result
    assert "other.bin" not in result


def test_filter_files_case_insensitive(hf_client: HFClient) -> None:
    """Test that quant tag matching is case insensitive."""
    files = [
        "model-Q8_0.gguf",
        "model-q8_0.gguf",
        "MODEL-Q8_0.GGUF",
    ]

    result = hf_client.filter_files(files, ["q8_0"])

    assert len(result) == len(files)


def test_filter_files_empty_tags(hf_client: HFClient) -> None:
    """Test filtering with empty quant tags returns only essential configs."""
    files = ["model.bin", "config.json"]

    result = hf_client.filter_files(files, [])

    assert result == ["config.json"]


def test_filter_files_subdirectory(hf_client: HFClient) -> None:
    """Test that files in subdirectories are handled correctly."""
    files = [
        "models/model-q8_0.gguf",
        "models/config.json",
        "other/file.bin",
    ]

    result = hf_client.filter_files(files, ["Q8_0"])

    assert "models/model-q8_0.gguf" in result
    assert "models/config.json" in result


def test_download_file(hf_client: HFClient) -> None:
    """Test downloading a single file."""
    with patch("beyond_vibes.hf.hf_hub_download") as mock_download:
        mock_download.return_value = "/tmp/downloaded/model.bin"

        result = hf_client.download_file("test/repo", "main", "model.bin")

        mock_download.assert_called_once_with(
            repo_id="test/repo", revision="main", filename="model.bin", token=None
        )
        assert result == Path("/tmp/downloaded/model.bin")


def test_download_file_custom_revision(hf_client: HFClient) -> None:
    """Test downloading with custom revision."""
    with patch("beyond_vibes.hf.hf_hub_download") as mock_download:
        mock_download.return_value = "/tmp/model.bin"

        hf_client.download_file("test/repo", "v2.0", "model.bin")

        mock_download.assert_called_once_with(
            repo_id="test/repo", revision="v2.0", filename="model.bin", token=None
        )
