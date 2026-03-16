"""Evaluation runner for orchestrating judge execution.

This module provides the main entry point for running evaluations
on simulation runs.

Example:
    from beyond_vibes.evaluators import EvaluationRunner

    runner = EvaluationRunner()
    results = runner.evaluate_run("run-id-123")

    # Batch evaluation
    runner.evaluate_batch(["run-1", "run-2"])

"""

import json
import logging
import os
from typing import Any

import mlflow

from beyond_vibes.evaluations.extractor import extract_run_data
from beyond_vibes.evaluations.judge_factory import build_judges_for_task
from beyond_vibes.evaluations.models import JudgeInput
from beyond_vibes.settings import settings
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.prompts.loader import load_task_config

logger = logging.getLogger(__name__)

# Configure boto3/AWS credentials for MLflow trace uploads
# MLflow's trace exporter uses boto3 directly for S3 artifact storage.
# We need to ensure AWS credentials are available from our S3 settings.
if settings.s3_access_key:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.s3_access_key)
if settings.s3_secret_key:
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.s3_secret_key)
if settings.s3_endpoint:
    # boto3 uses AWS_ENDPOINT_URL for custom S3 endpoints
    endpoint_url = f"https://{settings.s3_endpoint}"
    os.environ.setdefault("AWS_ENDPOINT_URL", endpoint_url)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# Trace middle positions for truncation (percentage points: 25%, 50%, 75%)
EVAL_TRACE_MIDDLE_POSITIONS = [0.25, 0.5, 0.75]

# Minimum remaining characters to apply complex truncation strategy
MIN_REMAINING_CHARS_FOR_TRUNCATION = 1000


class EvaluationRunner:
    """Runs judges on simulation runs and logs results.

    This class orchestrates the evaluation pipeline:
    1. Extract data from MLflow run
    2. Load task configuration
    3. Build judges from task config
    4. Run each judge with appropriate input artifact
    5. Log results back to the run

    Attributes:
        judge_model: Model to use for judges (overrides settings)

    """

    def __init__(self, judge_model: str | None = None) -> None:
        """Initialize evaluation runner.

        Args:
            judge_model: Override default judge model from settings

        """
        self.judge_model = judge_model or settings.judge_model
        logger.info(f"EvaluationRunner initialized with model: {self.judge_model}")

    def evaluate_run(
        self,
        run_id: str,
        judges: list[tuple[Any, str]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single run with specified judges.

        If judges not provided, loads them from task configuration.

        Args:
            run_id: MLflow run ID to evaluate
            judges: List of (judge_instance, input_artifact) tuples (optional)

        Returns:
            Dictionary mapping judge names to their results

        Raises:
            ValueError: If run cannot be evaluated
            mlflow.exceptions.MlflowException: If MLflow operation fails

        """
        logger.info(f"Evaluating run: {run_id}")

        # Step 1: Extract data from run
        # FAIL FAST: Any exception propagates with full stack trace
        judge_input = extract_run_data(run_id)
        logger.debug(f"Extracted data for task: {judge_input.task_name}")

        # Step 2: Load task config
        # FAIL FAST: Any exception propagates with full stack trace
        task_config = self._load_task_config(judge_input.task_name)

        # Step 3: Build judges if not provided
        if judges is None:
            judges = build_judges_for_task(task_config, self.judge_model)

        if not judges:
            logger.warning(f"No judges configured for run {run_id}")
            return {}

        logger.info(f"Running {len(judges)} judges on run {run_id}")

        # Step 4: Evaluate with each judge
        # FAIL FAST: Any exception from any judge propagates with full stack trace
        all_results = {}

        for judge, input_artifact in judges:
            result = self._evaluate_with_judge(judge, input_artifact, judge_input)
            all_results[judge.name] = result

            # Step 5: Log results
            # FAIL FAST: Any logging error propagates
            self._log_results_to_run(run_id, judge.name, result)

        logger.info(f"Completed evaluation of run {run_id}")
        return all_results

    def evaluate_batch(
        self,
        run_ids: list[str],
        continue_on_error: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate multiple runs in batch.

        Args:
            run_ids: List of MLflow run IDs to evaluate
            continue_on_error: If True, continue evaluating other runs on error

        Returns:
            Dictionary mapping run IDs to their results

        """
        results = {}
        success_count = 0

        for run_id in run_ids:
            try:
                run_results = self.evaluate_run(run_id)
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

    def _evaluate_with_judge(
        self,
        judge: Any,  # noqa: ANN401
        input_artifact: str,
        judge_input: JudgeInput,
    ) -> dict[str, Any]:
        """Evaluate a single judge with appropriate input.

        Args:
            judge: Judge instance
            input_artifact: Which artifact to use (git_diff, final_message, trace)
            judge_input: Standardized input data

        Returns:
            Evaluation result dictionary

        Raises:
            Exception: Any exception from judge evaluation propagates (fail fast)

        """
        # Special handling for ToolCallEfficiency - it needs actual MLflow Trace
        if self._is_tool_call_efficiency_judge(judge):
            return self._evaluate_tool_call_efficiency(judge, judge_input)

        # Prepare evaluation data based on input artifact
        eval_data = self._prepare_eval_data(judge_input, input_artifact)

        logger.debug(f"Running judge '{judge.name}' on {input_artifact}")

        # Call judge directly instead of using mlflow.genai.evaluate()
        # This avoids creating separate evaluation runs
        try:
            # Get inputs and outputs from eval_data
            inputs = eval_data.get("inputs", {})
            outputs = eval_data.get("outputs", {})

            # Call the judge directly with the data
            feedback = judge(inputs=inputs, outputs=outputs)

            # Extract score from Feedback object
            if hasattr(feedback, "value"):
                # Binary or categorical feedback
                score_val = 1.0 if feedback.value in ["yes", "pass", True] else 0.0
                rationale_val = getattr(feedback, "rationale", None)
                return {"score": score_val, "rationale": rationale_val}
            if hasattr(feedback, "score"):
                # Numeric score
                score_val = float(feedback.score)
                rationale_val = getattr(feedback, "rationale", None)
                return {"score": score_val, "rationale": rationale_val}
            # Try to convert to float directly
            return {"score": float(feedback), "rationale": None}

        except Exception as e:
            logger.error(f"Judge '{judge.name}' evaluation failed: {e}")
            raise

    def _is_tool_call_efficiency_judge(self, judge: Any) -> bool:  # noqa: ANN401
        """Check if judge is ToolCallEfficiency type.

        Args:
            judge: Judge instance to check

        Returns:
            True if judge is ToolCallEfficiency

        """
        return (
            hasattr(judge, "__class__")
            and judge.__class__.__name__ == "ToolCallEfficiency"
        )

    def _evaluate_tool_call_efficiency(
        self,
        judge: Any,  # noqa: ANN401
        judge_input: JudgeInput,
    ) -> dict[str, Any]:
        """Evaluate ToolCallEfficiency using actual MLflow Trace.

        ToolCallEfficiency is a session-level scorer that requires the actual
        MLflow Trace object, not the standard inputs/outputs format.

        NOTE: This method does NOT catch exceptions. If trace fetching fails,
        the exception propagates with full details for debugging.

        Args:
            judge: ToolCallEfficiency judge instance
            judge_input: Standardized input data with run_id

        Returns:
            Evaluation result dictionary

        Raises:
            ValueError: If no traces found for the run
            Exception: Any MLflow or judge evaluation error propagates

        """
        logger.debug(
            f"Fetching MLflow trace for ToolCallEfficiency: {judge_input.run_id}"
        )

        # Get the run to find the experiment_id
        run = mlflow.get_run(judge_input.run_id)
        if run is None:
            raise ValueError(
                f"Could not find MLflow run with ID: {judge_input.run_id}. "
                f"Run may have been deleted or tracking server is unavailable."
            )

        if not hasattr(run, "info"):
            raise ValueError(
                f"MLflow run object for {judge_input.run_id} is missing "
                f"'info' attribute. Run type: {type(run)}. This may indicate "
                f"a corrupted run or MLflow version incompatibility."
            )

        experiment_id = run.info.experiment_id

        # Search for traces associated with this run
        # MLflow stores the source run ID in metadata.mlflow.sourceRun
        client = mlflow.MlflowClient()
        traces = client.search_traces(
            experiment_ids=[experiment_id],
            filter_string=f"metadata.mlflow.sourceRun = '{judge_input.run_id}'",
        )

        # If no traces found, raise a clear error immediately
        if len(traces) == 0:
            raise ValueError(
                f"No traces found for run_id={judge_input.run_id}. "
                f"ToolCallEfficiency requires MLflow traces to be enabled "
                f"and stored. Ensure simulations are logging traces via "
                f"mlflow.start_span() and that traces are associated with "
                f"the run_id via metadata.mlflow.sourceRun."
            )

        # Get the first trace (should be the main simulation trace)
        trace = traces[0]

        # Verify trace is not None
        if trace is None:
            raise ValueError(
                f"First trace in results is None for run_id={judge_input.run_id}. "
                f"This may indicate a trace storage issue."
            )

        # Verify trace has required attributes (trace.info should exist)
        if not hasattr(trace, "info"):
            raise ValueError(
                f"Trace object for run_id={judge_input.run_id} is missing "
                f"required 'info' attribute. This indicates incomplete trace "
                f"data. Trace type: {type(trace)}. "
                f"Check MLflow trace configuration."
            )

        logger.debug(
            f"Invoking ToolCallEfficiency with trace for run {judge_input.run_id}"
        )

        # Invoke judge directly with trace parameter
        feedback = judge(trace=trace)

        # Extract score from Feedback object
        if hasattr(feedback, "value"):
            # Binary or categorical feedback
            score = 1.0 if feedback.value in ["yes", "efficient", True] else 0.0
        elif hasattr(feedback, "score"):
            # Numeric score
            score = float(feedback.score)
        else:
            raise ValueError(
                f"Unexpected feedback format from ToolCallEfficiency: "
                f"{type(feedback)}. Expected object with 'value' or "
                f"'score' attribute."
            )

        rationale = getattr(feedback, "rationale", None)

        return {"score": score, "rationale": rationale}

    def _prepare_eval_data(  # noqa: PLR0912
        self,
        judge_input: JudgeInput,
        input_artifact: str,
    ) -> dict[str, Any]:
        """Convert JudgeInput to format expected by mlflow.evaluate.

        Args:
            judge_input: Standardized input data
            input_artifact: Which artifact to use

        Returns:
            Dictionary in mlflow.evaluate format

        Format:
            {
                "inputs": {"request": ..., "system": ...},
                "outputs": {"response": ...},
                "context": ...
            }

        """
        if input_artifact == "git_diff":
            # Truncate git diff to prevent token limit errors
            if judge_input.git_diff:
                output_content = self._truncate_git_diff(judge_input.git_diff)
                # Log truncation if it occurred
                if len(output_content) < len(judge_input.git_diff):
                    original_kb = len(judge_input.git_diff) / 1024
                    truncated_kb = len(output_content) / 1024
                    logger.debug(
                        f"Truncated git diff: {original_kb:.1f}KB -> "
                        f"{truncated_kb:.1f}KB"
                    )
            else:
                output_content = ""
            context = None  # Context not needed when diff is the output

        elif input_artifact == "final_message":
            output_content = judge_input.final_message
            # Use compact trace summary for context
            if judge_input.trace:
                summary = self._create_trace_summary(judge_input.trace)
                context = json.dumps(summary, indent=2)
            else:
                context = None

        elif input_artifact == "trace":
            # Truncate trace for main output
            if judge_input.trace:
                truncated = self._truncate_trace(judge_input.trace)
                output_content = json.dumps(truncated, indent=2)
            else:
                output_content = "{}"
            # Use truncated git_diff as context if available
            if judge_input.git_diff:
                context = self._truncate_git_diff(
                    judge_input.git_diff, max_chars=settings.eval_context_diff_max_chars
                )
            else:
                context = None

        else:
            raise ValueError(f"Unknown input artifact: {input_artifact}")

        return {
            "inputs": {
                "request": judge_input.task_prompt,
                "system": judge_input.system_prompt,
            },
            "outputs": {
                "response": output_content,
            },
            "context": context,
        }

    def _log_results_to_run(
        self,
        run_id: str,
        judge_name: str,
        result: dict[str, Any],
    ) -> None:
        """Log evaluation results back to the simulation run.

        Args:
            run_id: MLflow run ID
            judge_name: Name of judge
            result: Evaluation result dictionary

        Raises:
            Exception: Propagates any MLflow logging errors (fail fast)

        """
        with mlflow.start_run(run_id=run_id):
            # Log score as metric
            score = result.get("score", 0.0)
            metric_name = f"eval_{judge_name}"
            mlflow.log_metric(metric_name, score)

            # Log rationale as tag (truncated to 500 chars for MLflow limits)
            rationale = result.get("rationale")
            if rationale:
                tag_name = f"{metric_name}_rationale"
                mlflow.set_tag(tag_name, rationale[:500])

            logger.debug(f"Logged results for '{judge_name}': score={score}")

    def _truncate_trace(
        self,
        trace: dict[str, Any],
        keep_first: int = -1,
        keep_last: int = -1,
        middle_positions: list[float] | None = None,
    ) -> dict[str, Any]:
        """Truncate trace to reduce token count while preserving context.

        Strategy: Keep first N and last N messages, plus deterministic samples
        from the middle at specified percentage positions (e.g., 25%, 50%, 75%).

        Args:
            trace: Full trace_session data
            keep_first: Number of messages to keep from start
            keep_last: Number of messages to keep from end
            middle_positions: List of float positions (0.0-1.0) for middle sampling

        Returns:
            Truncated trace with metadata about truncation

        """
        # Use settings defaults if sentinel values provided
        if keep_first < 0:
            keep_first = settings.eval_trace_keep_first
        if keep_last < 0:
            keep_last = settings.eval_trace_keep_last
        if middle_positions is None:
            middle_positions = EVAL_TRACE_MIDDLE_POSITIONS

        messages = trace.get("messages", [])
        total_messages = len(messages)

        # If trace is small enough, return as-is
        if total_messages <= keep_first + keep_last + len(middle_positions):
            return trace

        # Build list of indices to keep
        indices_to_keep = set()

        # Add first N
        indices_to_keep.update(range(min(keep_first, total_messages)))

        # Add last N
        indices_to_keep.update(
            range(max(0, total_messages - keep_last), total_messages)
        )

        # Add middle samples at deterministic positions
        for pos in middle_positions:
            idx = int(total_messages * pos)
            # Ensure we don't overlap with first/last sections
            if keep_first <= idx < total_messages - keep_last:
                indices_to_keep.add(idx)

        # Sort indices and build truncated message list
        sorted_indices = sorted(indices_to_keep)
        truncated_messages = []
        last_idx = -1

        for idx in sorted_indices:
            # Add omission marker if there's a gap
            if last_idx >= 0 and idx > last_idx + 1:
                omitted_count = idx - last_idx - 1
                truncated_messages.append(
                    {
                        "type": "truncation_marker",
                        "omitted_count": omitted_count,
                        "note": f"... {omitted_count} messages omitted ...",
                    }
                )

            truncated_messages.append(messages[idx])
            last_idx = idx

        # Build truncated trace
        return {
            **trace,
            "messages": truncated_messages,
            "truncated": True,
            "original_message_count": total_messages,
            "retained_message_count": len(truncated_messages),
            "truncation_config": {
                "keep_first": keep_first,
                "keep_last": keep_last,
                "middle_positions": middle_positions,
            },
        }

    def _create_trace_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
        """Create compact summary of trace for context usage.

        Returns key metrics without full message content.

        Args:
            trace: Full trace_session data

        Returns:
            Dictionary with key metrics and brief message info

        """
        messages = trace.get("messages", [])
        first_msg_role = None
        last_msg_role = None

        if messages:
            first_msg = messages[0]
            last_msg = messages[-1]
            first_raw = (
                first_msg.get("raw_message", {}) if isinstance(first_msg, dict) else {}
            )
            last_raw = (
                last_msg.get("raw_message", {}) if isinstance(last_msg, dict) else {}
            )
            first_msg_role = (
                first_raw.get("info", {}).get("role")
                if isinstance(first_raw, dict)
                else None
            )
            last_msg_role = (
                last_raw.get("info", {}).get("role")
                if isinstance(last_raw, dict)
                else None
            )

        return {
            "total_messages": trace.get("total_messages", 0),
            "total_tool_calls": trace.get("tool_total_calls", 0),
            "tool_loop_detected": trace.get("tool_loop_detected", False),
            "tool_error_rate": trace.get("tool_error_rate", 0.0),
            "total_tokens": trace.get("total_tokens", 0),
            "total_cost": trace.get("total_cost", 0.0),
            "completion_status": trace.get("completion_status"),
            "has_error": bool(trace.get("error")),
            "message_summary": {
                "first_message_role": first_msg_role,
                "last_message_role": last_msg_role,
            },
        }

    def _truncate_git_diff(
        self,
        diff: str,
        max_chars: int = -1,
        keep_header: bool = True,
    ) -> str:
        """Truncate git diff to fit within token limits.

        Strategy:
        1. Keep the header (file list and stats) if keep_header=True
        2. Keep changes from beginning and end
        3. Add clear truncation markers

        Args:
            diff: Full git diff string
            max_chars: Maximum characters to retain
            keep_header: Whether to preserve the diff header/file stats

        Returns:
            Truncated diff with markers indicating what was omitted

        """
        # Use settings default if sentinel value provided
        if max_chars < 0:
            max_chars = settings.eval_git_diff_max_chars

        if not diff or len(diff) <= max_chars:
            return diff

        lines = diff.split("\n")
        total_lines = len(lines)

        # Estimate average line length to calculate how many lines we can keep
        avg_line_len = len(diff) / total_lines if total_lines > 0 else 1

        if keep_header:
            # Extract header (file stats, summary lines)
            header_lines = []
            content_start = 0

            for i, line in enumerate(lines):
                # Header typically ends at first "diff --git" or "@@"
                if line.startswith("diff --git") or line.startswith("@@"):
                    content_start = i
                    break
                header_lines.append(line)

            # Keep header + beginning of content + end of content
            header_len = len("\n".join(header_lines))
            remaining_chars = max_chars - header_len - 500  # Buffer for markers

            if remaining_chars > MIN_REMAINING_CHARS_FOR_TRUNCATION:
                # Split remaining budget 60/40 between beginning and end
                begin_chars = int(remaining_chars * 0.6)
                end_chars = int(remaining_chars * 0.4)

                # Build truncated diff
                result_lines = header_lines.copy()
                result_lines.append("")
                result_lines.append("# === TRUNCATED: Showing partial diff ===")
                result_lines.append("")

                # Add beginning content
                current_chars = 0
                begin_end_idx = content_start
                for i in range(content_start, total_lines):
                    line = lines[i]
                    if current_chars + len(line) > begin_chars:
                        begin_end_idx = i
                        break
                    result_lines.append(line)
                    current_chars += len(line) + 1  # +1 for newline

                # Add truncation marker
                omitted_middle = total_lines - begin_end_idx - 1
                result_lines.append("")
                result_lines.append(f"# ... {omitted_middle} lines omitted ...")
                result_lines.append("")

                # Add end content
                end_start_idx = max(
                    begin_end_idx + 1, total_lines - int(end_chars / avg_line_len)
                )
                for i in range(end_start_idx, total_lines):
                    result_lines.append(lines[i])

                return "\n".join(result_lines)

        # Simple truncation: keep beginning and end
        begin_chars = int(max_chars * 0.6)
        end_chars = int(max_chars * 0.4)

        omitted = len(diff) - begin_chars - end_chars
        return (
            diff[:begin_chars]
            + f"\n\n# ... {omitted} characters omitted ...\n\n"
            + diff[-end_chars:]
        )

    def _load_task_config(self, task_name: str) -> SimulationConfig:
        """Load task configuration by name.

        Args:
            task_name: Name of task

        Returns:
            Task configuration

        Raises:
            ValueError: If task cannot be loaded

        """
        try:
            # Empty context dict for evaluation
            return load_task_config(task_name, "{}")
        except Exception as e:
            raise ValueError(f"Failed to load task '{task_name}': {e}") from e


# Convenience function for simple use cases
def evaluate_run(run_id: str, judge_model: str | None = None) -> dict[str, Any]:
    """Evaluate a single run with default configuration.

    Convenience function that creates a runner and evaluates.

    Args:
        run_id: MLflow run ID
        judge_model: Optional model override

    Returns:
        Evaluation results

    """
    runner = EvaluationRunner(judge_model=judge_model)
    return runner.evaluate_run(run_id)
