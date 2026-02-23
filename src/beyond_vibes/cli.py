"""CLI for downloading models from HuggingFace to S3."""

from pathlib import Path

import typer
import yaml

from beyond_vibes.config import Config
from beyond_vibes.hf import HFClient
from beyond_vibes.logger import logger
from beyond_vibes.s3 import S3Client
from beyond_vibes.settings import S3Settings

app = typer.Typer()
DEFAULT_CONFIG = "models.yaml"


@app.command()
def download(
    config_path: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Download models from HuggingFace to S3."""
    if config_path is None:
        config_path = Path(DEFAULT_CONFIG)

    config_data = yaml.safe_load(config_path.read_text())
    config = Config(**config_data)

    s3_settings = S3Settings()
    s3_client = S3Client(s3_settings)
    hf_client = HFClient()

    for model in config.models:
        logger.info(f"Processing {model.name}...")

        files = hf_client.list_files(model.repo_id, model.revision)
        filtered = hf_client.filter_files(files, model.quant_tags)

        logger.info(f"  Found {len(filtered)} files to download")

        for f in filtered:
            s3_key = f"{model.name}/{model.repo_id}/{f}"

            if dry_run:
                logger.info(
                    f"  [DRY RUN] Would upload to s3://{s3_settings.bucket}/{s3_key}"
                )
            else:
                local_path = hf_client.download_file(model.repo_id, model.revision, f)
                s3_client.upload_file(local_path, s3_key)
                logger.info(f"  Uploaded {f}")

    logger.info("Done!")


if __name__ == "__main__":
    app()
