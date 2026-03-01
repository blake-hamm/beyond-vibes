"""Tests for S3Client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beyond_vibes.model_downloader import S3Client


@pytest.fixture
def s3_client() -> S3Client:
    """Create S3Client with mocked Minio and settings."""
    with patch.dict(
        "os.environ",
        {
            "S3_BUCKET": "test-bucket",
            "S3_ENDPOINT": "https://s3.example.com",
            "S3_ACCESS_KEY": "test-access-key",
            "S3_SECRET_KEY": "test-secret-key",
        },
    ):
        with patch("beyond_vibes.model_downloader.s3.Minio") as mock_minio:
            mock_client = MagicMock()
            mock_minio.return_value = mock_client
            with patch("beyond_vibes.model_downloader.s3.settings") as mock_settings:
                mock_settings.s3_bucket = "test-bucket"
                mock_settings.s3_endpoint = "https://s3.example.com"
                mock_settings.s3_access_key = "test-access-key"
                mock_settings.s3_secret_key = "test-secret-key"
                client = S3Client()
                client._client = mock_client
                return client


def test_upload_file(s3_client: S3Client) -> None:
    """Test uploading a local file to S3."""
    local_path = Path("/tmp/test-file.txt")
    key = "models/test/model.bin"

    s3_client.upload_file(local_path, key)

    s3_client._client.fput_object.assert_called_once_with(
        "test-bucket", key, str(local_path)
    )


def test_upload_stream(s3_client: S3Client) -> None:
    """Test uploading bytes content to S3."""
    content = b"test content data"
    key = "models/test/data.json"

    s3_client.upload_stream(content, key)

    s3_client._client.put_object.assert_called_once()
    call_args = s3_client._client.put_object.call_args
    assert call_args[0][0] == "test-bucket"
    assert call_args[0][1] == key
    assert call_args[1]["length"] == len(content)


def test_upload_stream_empty_content(s3_client: S3Client) -> None:
    """Test uploading empty bytes content to S3."""
    content = b""
    key = "models/test/empty.json"

    s3_client.upload_stream(content, key)

    s3_client._client.put_object.assert_called_once()
    call_args = s3_client._client.put_object.call_args
    assert call_args[1]["length"] == 0
