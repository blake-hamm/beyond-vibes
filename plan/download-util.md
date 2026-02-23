# HuggingFace to S3 Model Downloader CLI - Implementation Plan

## Overview

Create a CLI utility that reads a YAML config defining HuggingFace models, validates it with Pydantic, downloads the model files (filtered by quant tags), and uploads them to S3.

## Project Structure (Flat)

```
src/beyond_vibes/
├── __init__.py
├── settings.py  # S3Settings (BaseSettings from env vars)
├── config.py    # Pydantic models for YAML config (ModelConfig, Config)
├── hf.py        # HuggingFace download + filtering
├── s3.py        # S3 upload service
└── cli.py       # Typer app with `download` command
```

## YAML Schema (`models.yaml`)

```yaml
bucket: "my-models"

models:
  - name: "mistral-7b"
    repo_id: "TheBloke/Mistral-7B-GGUF"
    quant_tags: ["Q8_0", "Q4_K_M"]
```

## S3 Path Format

```
s3://{bucket}/{name}/{repo_id}/{filename}
```

Example: `s3://my-models/mistral-7b/TheBloke/Mistral-7B-GGUF/model-q8_0.gguf`

## Essential Config Files (Always Include)

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`

## CLI Usage

```bash
beyond-vibes download                    # Uses default models.yaml
beyond-vibes download config/custom.yaml
beyond-vibes download --dry-run
```

## Environment Variables

```
S3_BUCKET          # Required
S3_ENDPOINT        # Required (S3-compatible endpoint)
S3_ACCESS_KEY      # Required
S3_SECRET_KEY      # Required
```

## Implementation Steps

### Step 1: Add Dependencies
~~Update `pyproject.toml` to add required packages:~~
- ~~`typer`~~
- ~~`pydantic`~~
- ~~`pyyaml`~~
- ~~`minio`~~
- ~~`huggingface-hub`~~
- ~~`python-dotenv`~~

**COMPLETED** - Added typer, pydantic, pyyaml, minio, huggingface-hub, python-dotenv to pyproject.toml

### Step 2: Create `src/beyond_vibes/settings.py` and `src/beyond_vibes/config.py`

**settings.py** - Project-wide S3 settings:
- `S3Settings(BaseSettings)` - Loads from env vars (bucket, endpoint, access_key, secret_key)

**config.py** - Model/container config:
- `ModelConfig` - name, repo_id, quant_tags, revision
- `Config` - Root model with bucket + list of models

Also include constant for essential config files to always include:
```python
ESSENTIAL_MODEL_CONFIGS = {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"}
```

**COMPLETED**

### Step 3: Create `src/beyond_vibes/s3.py`
Implement `S3Client` class:
- Initialize Minio client from S3Settings
- `upload_file(local_path: Path, key: str)` - Upload single file
- `upload_stream(content: bytes, key: str)` - Upload from memory

**COMPLETED**

### Step 4: Create `src/beyond_vibes/hf.py`
Implement `HFClient` class:
- `list_files(repo_id: str, revision: str)` - List all files in repo
- `filter_files(files: list[str], quant_tags: list[str])` - Filter by tags + essential configs
- `download_file(repo_id: str, revision: str, filename: str)` - Download single file to temp

**COMPLETED**

### Step 5: Create `src/beyond_vibes/cli.py`
Implement Typer app:
- `app = Typer()`
- `download` command with:
  - `config_path: Path` argument (default: `models.yaml`)
  - `--dry-run` flag
- Load YAML → validate with Pydantic
- Load S3Settings from env
- Loop models: list files → filter → download → upload
- Fail fast on error, print progress

**COMPLETED**

### Step 6: Update `pyproject.toml`
Add CLI entry point:
```toml
[project.scripts]
beyond-vibes = "beyond_vibes.cli:app"
```

**COMPLETED**

### Step 7: Create `tests/test_settings.py` and `tests/test_config.py`
Test Pydantic validation:
- Valid YAML parsing
- Env var loading for S3Settings
- Invalid config raises ValidationError

### Step 8: Run Lint and Typecheck
```bash
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/
```

## Logic Flow

1. Load config from YAML + S3Settings from env
2. For each model:
   - List HF repo files
   - Filter: matches quant_tag OR is essential config (config.json, tokenizer*.json, etc.)
   - Stream download → upload to S3
3. Fail fast on any error (Argo can retry)
