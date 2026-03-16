"""Data models for evaluations module."""

from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeInput:
    """Standardized input for all judges."""

    run_id: str
    task_name: str
    archetype: str
    system_prompt: str
    task_prompt: str
    final_message: str
    git_diff: str | None
    trace: dict[str, Any]


@dataclass
class EvalResult:
    """Result from a single judge evaluation."""

    name: str
    score: float
    rationale: str | None = None
