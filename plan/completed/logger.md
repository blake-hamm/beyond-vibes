# Logger Enhancement Plan

This plan adapts the enhanced logging system to the beyond-vibes project.

---

## Current State

- `settings.py` — Has `S3Settings` with `pydantic-settings` already (env prefix `S3_`)
- `logger.py` — Module-level logger with hardcoded `INFO` level
- `cli.py` — Imports `from beyond_vibes.logger import logger`
- `.env.example` — Has S3 vars, no log level
- `pydantic-settings` already in dependencies

---

## Step 1: Consolidate `settings.py`

Merge into a single `Settings` class:

```python
# src/beyond_vibes/settings.py
"""Application settings loaded from environment variables."""

from typing import Literal

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Main application settings."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    s3_bucket: str
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Accept both uppercase and lowercase log levels."""
        if isinstance(v, str):
            return v.upper()
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
```

---

## Step 2: Update imports

All places using `S3Settings` need to use the consolidated `settings` instead:

- `cli.py`: `from beyond_vibes.settings import settings` (remove `S3Settings` import)
- `s3.py`: Update to use `settings.s3_bucket`, `settings.s3_endpoint`, etc.

---

## Step 3: Rewrite `logger.py`

Shift from module-level logger to a configure function:

```python
# src/beyond_vibes/logger.py
"""Logging configuration."""

import logging
import sys

from beyond_vibes.settings import settings


def configure_logging() -> None:
    """Configure the beyond_vibes package logger."""
    logger = logging.getLogger("beyond_vibes")
    logger.propagate = False

    logger.setLevel(settings.log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
```

---

## Step 4: Update `__init__.py`

Call configure on package import:

```python
# src/beyond_vibes/__init__.py
"""Beyond Vibes - LLM evaluation framework."""

from beyond_vibes.logger import configure_logging

configure_logging()
```

---

## Step 5: Update `cli.py`

Replace the import and add `--debug` flag:

```python
# src/beyond_vibes/cli.py
"""CLI for downloading models from HuggingFace to S3."""

import logging
from pathlib import Path

import typer
import yaml

from beyond_vibes.config import Config
from beyond_vibes.hf import HFClient
from beyond_vibes.s3 import S3Client
from beyond_vibes.settings import S3Settings

logger = logging.getLogger(__name__)

app = typer.Typer()
DEFAULT_CONFIG = "models.yaml"


@app.callback()
def main(debug: bool = typer.Option(False, "--debug", help="Enable debug logging")) -> None:
    if debug:
        logging.getLogger("beyond_vibes").setLevel(logging.DEBUG)


@app.command()
def download(
    config_path: Path | None = Path(DEFAULT_CONFIG),
    dry_run: bool = False,
) -> None:
    ...
```

---

## Step 6: Update `.env.example`

```bash
# .env.example
LOG_LEVEL=INFO   # DEBUG | INFO | WARNING | ERROR | CRITICAL

S3_BUCKET=my-models
S3_ENDPOINT=s3.example.com
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
```

---

## Execution Order

| Step | File | Action |
|------|------|--------|
| 1 | `src/beyond_vibes/settings.py` | Consolidate into single `Settings` class with `log_level` and S3 fields |
| 2 | `src/beyond_vibes/logger.py` | Replace with `configure_logging()` function |
| 3 | `src/beyond_vibes/__init__.py` | Import and call `configure_logging()` |
| 4 | `src/beyond_vibes/cli.py` | Swap to `getLogger(__name__)`, add `--debug` flag, use consolidated `settings` |
| 5 | `src/beyond_vibes/s3.py` | Update to use `settings.s3_*` instead of `S3Settings` |
| 6 | `.env.example` | Document `LOG_LEVEL` (unprefixed) |
