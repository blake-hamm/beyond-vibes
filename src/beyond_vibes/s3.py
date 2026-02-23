"""S3 client for uploading files."""

import io
from pathlib import Path

from minio import Minio

from beyond_vibes.settings import S3Settings


class S3Client:
    """Client for uploading files to S3-compatible storage."""

    def __init__(self, settings: S3Settings) -> None:
        """Initialize the S3 client."""
        self._client = Minio(
            settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
        )
        self._bucket = settings.bucket

    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a local file to S3."""
        self._client.fput_object(self._bucket, key, str(local_path))

    def upload_stream(self, content: bytes, key: str) -> None:
        """Upload bytes content to S3."""
        self._client.put_object(
            self._bucket,
            key,
            io.BytesIO(content),
            length=len(content),
        )
