"""MLflow run data extraction for evaluation."""

import logging
from typing import Any

import mlflow

from beyond_vibes.evaluations.models import JudgeInput

logger = logging.getLogger(__name__)


def extract_run_data(run_id: str) -> JudgeInput:
    """Extract standardized input from MLflow run."""
    run = mlflow.get_run(run_id)

    task_name = run.data.tags.get("task.name", "")
    archetype = run.data.tags.get("task.archetype", "")
    task_prompt = run.data.params.get("task.prompt", "")

    system_prompt = _load_artifact(run_id, "system_prompt.txt")
    git_diff = _load_artifact(run_id, "git_diff.patch")
    trace = _load_trace_session(run_id)
    final_message = _extract_final_message(trace)

    return JudgeInput(
        run_id=run_id,
        task_name=task_name,
        archetype=archetype,
        system_prompt=system_prompt or "",
        task_prompt=task_prompt,
        final_message=final_message,
        git_diff=git_diff,
        trace=trace,
    )


def _load_artifact(run_id: str, artifact_path: str) -> str | None:
    """Load text artifact from MLflow run."""
    try:
        artifact_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.artifacts.load_text(artifact_uri)
    except Exception:
        logger.debug("Artifact %s not found for run %s", artifact_path, run_id)
        return None


def _load_trace_session(run_id: str) -> dict[str, Any]:
    """Load trace_session.json artifact from run."""
    try:
        artifact_uri = f"runs:/{run_id}/trace_session.json"
        return mlflow.artifacts.load_dict(artifact_uri)
    except Exception as e:
        logger.error("Failed to load trace_session.json for run %s: %s", run_id, e)
        return {
            "total_messages": 0,
            "total_tool_calls": 0,
            "tool_error_count": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "tool_loop_detected": False,
            "tool_loop_threshold": 3,
            "tool_max_consecutive_calls": 0,
            "tool_error_rate": 0.0,
            "token_efficiency": 0.0,
            "cost_efficiency": 0.0,
            "error_message_indices": [],
            "turns": [],
        }


def _extract_final_message(trace: dict[str, Any]) -> str:
    """Extract final message content from trace."""
    turns = trace.get("turns", [])

    if not turns:
        return ""

    last_turn = turns[-1]
    raw_message = last_turn.get("raw_message", {})
    content = raw_message.get("content", [])
    content_parts = []

    for block in content:
        block_type = block.get("type", "")
        if block_type == "text":
            content_parts.append(block.get("text", ""))
        elif block_type == "thinking":
            content_parts.append(block.get("thinking", ""))

    return "\n".join(content_parts)


def query_simulation_runs(
    experiment: str = "beyond-vibes",
    run_ids: list[str] | None = None,
    task_name: str | None = None,
    archetype: str | None = None,
    status: str | None = None,
) -> list[Any]:
    """Query simulation runs matching filters."""
    filters = []

    if task_name:
        filters.append(f"tags.`task.name` = '{task_name}'")

    if archetype:
        filters.append(f"tags.`task.archetype` = '{archetype}'")

    if status:
        if status == "success":
            filters.append("tags.`run.status` = 'success'")
        elif status == "error":
            filters.append("tags.`run.status` = 'error'")

    filter_string = " and ".join(filters) if filters else None

    if run_ids:
        runs = []
        for run_id in run_ids:
            try:
                run = mlflow.get_run(run_id)
                runs.append(run)
            except Exception as e:
                logger.warning("Failed to load run %s: %s", run_id, e)
        return runs

    experiment_obj = mlflow.get_experiment_by_name(experiment)
    if not experiment_obj:
        logger.warning("Experiment '%s' not found", experiment)
        return []

    return mlflow.search_runs(
        experiment_ids=[experiment_obj.experiment_id],
        filter_string=filter_string,
        output_format="list",
    )
