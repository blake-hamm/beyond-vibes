"""S3 client for uploading files."""

import io
from pathlib import Path

from minio import Minio

from beyond_vibes.settings import settings


class S3Client:
    """Client for uploading files to S3-compatible storage."""

    def __init__(self) -> None:
        """Initialize the S3 client."""
        self._client = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
        )
        self._bucket = settings.s3_bucket

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
