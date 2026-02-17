# Agent Guidelines for Beyond Vibes

This repository is a model evaluation benchmark for testing agentic coding capabilities. Code should be clean, well-tested, and follow established conventions.

## Build, Lint, and Test Commands

### Python (Primary Language)
```bash
# Install dependencies
uv sync                    # Sync dependencies from pyproject.toml
uv add <package>           # Add production dependency
uv add --dev pytest        # Add dev dependency (pytest, ruff, black, mypy)

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_example.py

# Run a single test
uv run pytest tests/test_example.py::test_function_name -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Linting and formatting
uv run ruff check .           # Check for linting errors
uv run ruff check --fix .     # Auto-fix linting errors
uv run ruff format .          # Format code
uv run black .                # Alternative formatter
uv run isort .                # Sort imports

# Type checking
uv run mypy src/
uv run pyright
```

### General
```bash
# Pre-commit hooks (if configured)
pre-commit run --all-files

# Git hooks
git diff --check  # Check for trailing whitespace
```

## Code Style Guidelines

### General Principles
- **Readability over cleverness**: Write code that is easy to understand
- **Explicit over implicit**: Avoid magic unless well-documented
- **Single responsibility**: Functions and classes should do one thing well
- **Fail fast**: Validate inputs early and raise descriptive errors

### Python Conventions

#### Imports
```python
# Order: stdlib, third-party, local
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.models import ModelConfig
from src.utils import helper_function
```

#### Naming Conventions
- `snake_case` for variables, functions, methods
- `PascalCase` for classes
- `UPPER_CASE` for constants
- `_private_prefix` for internal use
- `__double_underscore` for name mangling (rarely needed)

#### Type Hints
```python
from typing import List, Optional, Union

def process_data(
    data: List[dict],
    threshold: float = 0.5,
    callback: Optional[callable] = None
) -> dict:
    """Process data with optional callback."""
    ...
```

#### Docstrings
Use Google-style docstrings:
```python
def evaluate_model(
    model: str,
    dataset: Dataset,
    metrics: List[str]
) -> EvaluationResult:
    """Evaluate a model on a dataset.

    Args:
        model: Model identifier or path
        dataset: Dataset to evaluate on
        metrics: List of metric names to compute

    Returns:
        EvaluationResult containing scores and metadata

    Raises:
        ValueError: If model or dataset is invalid
        RuntimeError: If evaluation fails
    """
```

#### Error Handling
```python
# Use specific exceptions
raise ValueError(f"Invalid threshold: {threshold}")
raise FileNotFoundError(f"Config not found: {path}")

# Context managers for resources
with open(file_path, 'r') as f:
    data = json.load(f)

# Don't catch bare exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Validation failed: {e}")
    raise
```

### Configuration Management
- Use environment variables for secrets (`python-dotenv`)
- Use Pydantic models for config validation
- Keep configs in `configs/` directory
- Never commit secrets to version control

### Testing
- Use `pytest` as the test runner
- Test file naming: `test_*.py`
- Test function naming: `test_*`
- Use fixtures for setup/teardown
- Mock external API calls
- Aim for >80% coverage on critical paths

### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed diagnostic info")
logger.info("General operational info")
logger.warning("Something unexpected but handled")
logger.error("Error that prevents operation")
```

### Project Structure
```
├── src/                    # Source code
│   ├── __init__.py
│   ├── models/
│   ├── evaluators/
│   └── utils/
├── tests/                  # Test files
│   ├── __init__.py
│   ├── conftest.py
│   └── test_*.py
├── configs/                # Configuration files
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Utility scripts
├── data/                   # Data files (gitignored)
├── pyproject.toml         # Project configuration
├── uv.lock                # uv lockfile
└── README.md
```

## Git Workflow
1. Create feature branches: `git checkout -b feature/description`
2. Make atomic commits with descriptive messages
3. Run tests before pushing: `uv run pytest`
4. Run linting: `uv run ruff check . && uv run ruff format .`
5. No commit of: `.env`, `__pycache__/`, `*.pyc`, `.DS_Store`, `data/`

## LLM/AI Guidelines
- Add `# AI Generated` comment for significant AI-written code sections
- Review all AI-generated code for correctness
- Ensure AI code follows these style guidelines
- Document assumptions and limitations in complex AI-generated logic
