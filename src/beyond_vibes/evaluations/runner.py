"""Evaluation runner for running Guidelines scorer on simulation runs."""

import fnmatch
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.genai.scorers import Guidelines

from beyond_vibes.evaluations.extractor import extract_run_data
from beyond_vibes.settings import settings
from beyond_vibes.simulations.prompts.loader import load_task_config

logger = logging.getLogger(__name__)

# Patterns to exclude from git diff (lock files and generated content)
EXCLUDE_PATTERNS = [
    "*.lock",
    "poetry.lock",
    "uv.lock",
    "package-lock.json",
    "yarn.lock",
    "Pipfile.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
    "*.min.js",
    "*.min.css",
]

# Git diff context to help judge understand the format
GIT_DIFF_CONTEXT = """This is a git diff showing changes made by the agent.
Lines starting with '+' were ADDED (new state after agent's work)
Lines starting with '-' were REMOVED (old state before agent's work)
Lines without +/- are context lines for reference.

Evaluate whether the final state (after changes) meets the guideline.

---

"""

# Minimum number of parts expected in "diff --git a/path b/path" line
DIFF_GIT_PARTS_MIN = 4

# Index of the "b/path" part in diff --git line
DIFF_GIT_B_PATH_INDEX = 3


def _should_exclude_file(filepath: str) -> bool:
    """Check if file should be excluded from diff evaluation.

    Args:
        filepath: Path to the file from git diff

    Returns:
        True if file should be excluded

    """
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(filepath, pattern):
            return True
    return False


def _process_diff_header(
    line: str, skip_file_content: bool, skipped_count: int
) -> tuple[bool, int]:
    """Process a diff header line and determine if we should skip content.

    Args:
        line: The diff header line starting with "diff --git"
        skip_file_content: Current skip state
        skipped_count: Current count of skipped files

    Returns:
        Tuple of (new_skip_state, new_skipped_count)

    """
    parts = line.split()

    if len(parts) >= DIFF_GIT_PARTS_MIN:
        filepath = (
            parts[DIFF_GIT_B_PATH_INDEX][2:]
            if parts[DIFF_GIT_B_PATH_INDEX].startswith("b/")
            else parts[DIFF_GIT_B_PATH_INDEX]
        )
        new_skip = _should_exclude_file(filepath)
        if new_skip:
            skipped_count += 1
            logger.debug(f"Filtering content for: {filepath}")
        return new_skip, skipped_count

    return skip_file_content, skipped_count


def _filter_git_diff(git_diff: str) -> str:
    """Filter out lock files and generated content from git diff.

    Preserves file headers (to show file creation/deletion) but removes
    the actual diff content for lock files and generated files.

    Args:
        git_diff: Raw git diff string

    Returns:
        Filtered git diff with lock file content removed but headers preserved

    """
    if not git_diff:
        return ""

    lines = git_diff.split("\n")
    filtered_lines = []
    skip_file_content = False
    in_hunk = False
    skipped_count = 0
    content_skipped = 0

    for line in lines:
        is_diff_header = line.startswith("diff --git")
        is_index = line.startswith("index ")
        is_minus = line.startswith("--- ")
        is_plus = line.startswith("+++ ")
        is_metadata = is_index or is_minus or is_plus
        is_hunk_header = line.startswith("@@ ")
        is_no_newline = line.startswith("\\ No newline")

        if is_diff_header:
            skip_file_content, skipped_count = _process_diff_header(
                line, skip_file_content, skipped_count
            )
            in_hunk = False
            filtered_lines.append(line)
        elif is_metadata:
            filtered_lines.append(line)
        elif is_hunk_header:
            in_hunk = True
            if skip_file_content:
                content_skipped += 1
                if content_skipped == 1:
                    filtered_lines.append("# [content omitted - lock/generated file]")
            else:
                filtered_lines.append(line)
        elif in_hunk and skip_file_content:
            continue
        elif is_no_newline and not skip_file_content:
            filtered_lines.append(line)
        elif not (in_hunk and skip_file_content):
            filtered_lines.append(line)

    if skipped_count > 0:
        logger.info(f"Filtered content from {skipped_count} lock/generated files")

    return "\n".join(filtered_lines)


def _extract_feedback_score(feedback: Any) -> float:  # noqa: ANN401
    """Extract score from Guidelines Feedback object.

    Args:
        feedback: Feedback object from Guidelines scorer

    Returns:
        Score as float (0.0-1.0)

    """
    score_attr = getattr(feedback, "score", None)
    if score_attr is not None:
        return float(score_attr)

    value_attr = getattr(feedback, "value", None)
    if value_attr is not None:
        # Binary feedback: yes/pass = 1.0, no/fail = 0.0
        return 1.0 if value_attr in ["yes", "pass", True] else 0.0

    # Fallback: try direct conversion
    try:
        return float(str(feedback))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        logger.warning(f"Could not convert feedback to score: {feedback}")
        return 0.0


def evaluate_run(run_id: str, judge_model: str | None = None) -> dict[str, Any]:
    """Evaluate a single run using Guidelines scorer.

    Evaluates the git diff against each task guideline individually and logs
    results to the existing MLflow run as metrics and an artifact.

    Args:
        run_id: MLflow run ID to evaluate
        judge_model: Model to use for Guidelines (defaults to settings.judge_model)

    Returns:
        Dictionary with scores and rationales for each guideline

    """
    model = judge_model or settings.judge_model
    logger.info(f"Evaluating run {run_id} with model {model}")

    # Extract run data
    judge_input = extract_run_data(run_id)
    task_config = load_task_config(judge_input.task_name)

    if not task_config.guidelines:
        logger.warning(f"No guidelines configured for task '{task_config.name}'")
        return {"score": 0.0, "rationale": "No guidelines configured"}

    # Filter git diff to remove lock files and generated content
    filtered_diff = _filter_git_diff(judge_input.git_diff or "")
    original_length = len(judge_input.git_diff or "")
    filtered_length = len(filtered_diff)

    if filtered_length < original_length:
        logger.info(
            f"Filtered git diff: {original_length:,} chars → {filtered_length:,} chars "
            f"({original_length - filtered_length:,} removed)"
        )

    # Prepare git diff with context explanation
    git_diff_with_context = GIT_DIFF_CONTEXT + filtered_diff

    # Evaluate each guideline separately
    guideline_results = {}
    scores = []

    for guideline_name, guideline_criteria in task_config.guidelines.items():
        logger.info(f"Evaluating guideline '{guideline_name}': {guideline_criteria}")

        # Create Guidelines scorer for this specific guideline
        judge = Guidelines(
            name=f"{task_config.name}_{guideline_name}",
            guidelines=[guideline_criteria],  # Guidelines expects a list
            model=model,
        )

        # Run evaluation
        feedback = judge(
            inputs={"request": judge_input.task_prompt},
            outputs={"response": git_diff_with_context},
        )

        # Extract results
        score = _extract_feedback_score(feedback)
        rationale = getattr(feedback, "rationale", None) or ""

        guideline_results[guideline_name] = {
            "criteria": guideline_criteria,
            "score": score,
            "rationale": rationale,
        }
        scores.append(score)

        logger.info(f"Guideline '{guideline_name}': score={score:.2f}")

    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Prepare evaluation results
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_id": run_id,
        "task_name": task_config.name,
        "model": model,
        "guidelines": guideline_results,
        "average_score": avg_score,
        "git_diff_original_length": original_length,
        "git_diff_filtered_length": filtered_length,
        "git_diff_filtered": filtered_length < original_length,
    }

    # Log to existing run
    with mlflow.start_run(run_id=run_id):
        # Log individual guideline scores
        for guideline_name, result in guideline_results.items():
            metric_name = f"guidelines_{guideline_name}_score"
            mlflow.log_metric(metric_name, result["score"])

        # Log average score
        mlflow.log_metric("guidelines_average_score", avg_score)

        # Log artifact with full results
        artifact_path = Path("evaluation_results.json")
        with artifact_path.open("w") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(artifact_path))
        artifact_path.unlink()  # Clean up temp file

    logger.info(f"Evaluation complete for run {run_id}: avg_score={avg_score:.2f}")
    return {
        "guidelines": guideline_results,
        "average_score": avg_score,
    }


def evaluate_batch(
    run_ids: list[str],
    judge_model: str | None = None,
    continue_on_error: bool = True,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple runs in batch.

    Args:
        run_ids: List of MLflow run IDs to evaluate
        judge_model: Model to use for Guidelines
        continue_on_error: If True, continue evaluating other runs on error

    Returns:
        Dictionary mapping run IDs to their results

    """
    results = {}
    success_count = 0

    for run_id in run_ids:
        try:
            run_results = evaluate_run(run_id, judge_model)
            results[run_id] = run_results
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to evaluate run {run_id}: {e}")
            results[run_id] = {"error": str(e)}
            if not continue_on_error:
                break

    logger.info(
        f"Batch evaluation complete: {success_count}/{len(run_ids)} runs successful"
    )
    return results


class EvaluationRunner:
    """Simple runner for Guidelines-based evaluations.

    Maintains backward compatibility with existing code that instantiates
    a runner class rather than using the module-level functions.

    """

    def __init__(self, judge_model: str | None = None) -> None:
        """Initialize evaluation runner.

        Args:
            judge_model: Override default judge model from settings

        """
        self.judge_model = judge_model or settings.judge_model

    def evaluate_run(self, run_id: str) -> dict[str, Any]:
        """Evaluate a single run."""
        return evaluate_run(run_id, self.judge_model)

    def evaluate_batch(
        self, run_ids: list[str], continue_on_error: bool = True
    ) -> dict[str, dict[str, Any]]:
        """Evaluate multiple runs."""
        return evaluate_batch(run_ids, self.judge_model, continue_on_error)
