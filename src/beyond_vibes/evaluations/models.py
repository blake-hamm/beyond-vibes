"""Data models for evaluations module."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


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


class EvalResult(BaseModel):
    """Result from a single judge evaluation."""

    name: str
    score: float
    rationale: str | None = None
    criteria: str | None = None


class EvaluationArtifact(BaseModel):
    """Complete evaluation results artifact saved to JSON."""

    timestamp: str
    run_id: str
    task_name: str
    model: str
    guidelines: dict[str, EvalResult]
    average_score: float
    git_diff_original_length: int
    git_diff_filtered_length: int
    git_diff_filtered: bool
