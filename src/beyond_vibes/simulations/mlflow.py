"""MLflow tracing integration for simulation runs."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

import mlflow
from mlflow.entities.span import LiveSpan, SpanType
from mlflow.entities.span_event import SpanEvent
from pydantic import BaseModel, Field

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.pi_dev import TurnData

logger = logging.getLogger(__name__)

MAX_RAW_JSON_LEN = 4096


def generate_session_id(model_config: ModelConfig, quant_tag: str | None = None) -> str:
    """Generate a unique session ID from model config and timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    quant = quant_tag or "fp16"
    return f"{model_config.name}_{quant}_{timestamp}"


class SimulationSession(BaseModel):
    """Complete simulation session data."""

    model_config = {"arbitrary_types_allowed": True}

    sim_config: SimulationConfig
    llm_config: ModelConfig
    quant_tag: str | None = None
    session_id: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    total_messages: int = 0
    total_model_compute_time_seconds: float | None = None
    turns: list[TurnData] = Field(default_factory=list)
    git_diff: str | None = None
    system_prompt: str | None = None
    error: str | None = None
    stderr_content: str | None = None
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_tokens: int = 0
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    tool_error_count: int = 0
    completion_status: str | None = None
    tool_loop_threshold: int = 3
    tool_total_calls: int | None = None
    tool_loop_detected: bool | None = None
    tool_error_rate: float | None = None
    token_efficiency: float | None = None
    cost_efficiency: float | None = None
    avg_time_to_first_token_seconds: float | None = None
    avg_prompt_tokens_per_second: float | None = None
    avg_generation_tokens_per_second: float | None = None
    total_generation_time_seconds: float | None = None
    total_prompt_processing_time_seconds: float | None = None


class MlflowTracer:
    """MLflow tracer - captures simulation data as traces for later DSPy eval."""

    def __init__(
        self,
        experiment_name: str = "beyond-vibes",
        quant_tag: str | None = None,
        container_tag: str | None = None,
    ) -> None:
        """Initialize simulation logger with MLflow tracing."""
        self.experiment_name = experiment_name
        self.run_id: str | None = None
        self.session: SimulationSession | None = None
        self.quant_tag = quant_tag
        self.container_tag = container_tag
        self._active_span: LiveSpan | None = None
        self._tool_last_name: str | None = None
        self._tool_consecutive_calls: int = 0
        self._tool_max_consecutive_calls: int = 0

    @contextmanager
    def log_simulation(
        self,
        sim_config: SimulationConfig,
        model_config: ModelConfig,
    ) -> Iterator["MlflowTracer"]:
        """Context manager for tracing a simulation run."""
        session_id = generate_session_id(model_config, self.quant_tag)

        self.session = SimulationSession(
            sim_config=sim_config,
            llm_config=model_config,
            quant_tag=self.quant_tag,
            session_id=session_id,
            started_at=datetime.now(timezone.utc),
        )

        mlflow.set_experiment(self.experiment_name)

        try:
            with mlflow.start_run(run_name=session_id) as run:
                self.run_id = run.info.run_id
                logger.info(
                    "Started MLflow run %s for session: %s",
                    self.run_id,
                    session_id,
                )

                mlflow.log_param("model.name", model_config.name)
                mlflow.log_param("model.provider", model_config.provider)
                mlflow.log_param("model.model_id", model_config.get_model_id())
                if model_config.repo_id:
                    mlflow.log_param("model.repo_id", model_config.repo_id)
                if self.quant_tag:
                    mlflow.log_param("model.quant", self.quant_tag)
                if self.container_tag:
                    mlflow.log_param("runtime.container", self.container_tag)

                mlflow.set_tag("task.name", sim_config.name)
                mlflow.set_tag("task.archetype", sim_config.archetype)
                mlflow.set_tag("repository.url", sim_config.repository.url)
                mlflow.set_tag("repository.branch", sim_config.repository.branch)

                try:
                    yield self
                finally:
                    try:
                        self._flush()
                    except Exception:
                        logger.exception("Failed to flush session data to MLflow")
                        try:
                            mlflow.set_tag("run.status", "flush_error")
                        except Exception:
                            pass
                        raise

        except Exception as e:
            if self.run_id is None:
                logger.error("Failed to start MLflow run: %s", e)
            raise

    def _end_active_span(self) -> None:
        """End the current active span if one exists."""
        if self._active_span is not None:
            self._active_span.end()
            self._active_span = None

    def log_turn(self, turn: TurnData) -> None:  # noqa: PLR0915
        """Log a native pi turn as a span in the trace.

        Keeps the span open so log_error() can mark it failed later.
        """
        if self.session is None:
            logger.warning("No active session to log turn to")
            return

        # Close previous turn's span before starting a new one
        self._end_active_span()

        turn_index = turn.turn_index
        span_kwargs = {
            "name": f"turn_{turn_index}",
            "span_type": SpanType.AGENT,
            "attributes": {"run_id": self.run_id},
            "metadata": {"mlflow.trace.session": self.session.session_id},
        }
        if turn.assistant_message_start_ns is not None:
            span_kwargs["start_time_ns"] = turn.assistant_message_start_ns

        parent_span = mlflow.start_span_no_context(**span_kwargs)
        self._active_span = parent_span

        readable_parts = []
        for block in turn.content:
            block_type = block.get("type", "")
            if block_type == "text" and (text := block.get("text")):
                readable_parts.append({"type": "text", "content": text})
            elif block_type == "thinking" and (text := block.get("thinking")):
                readable_parts.append({"type": "thinking", "content": text})
            else:
                readable_parts.append(block)

        parent_span.set_inputs({"turn_index": turn_index, "role": "assistant"})
        parent_span.set_outputs({"content": readable_parts})

        usage = turn.usage or {}
        input_tokens = usage.get("input", 0)
        output_tokens = usage.get("output", 0)
        cache_read_tokens = usage.get("cacheRead", 0)
        cache_write_tokens = usage.get("cacheWrite", 0)
        total_tokens = usage.get(
            "totalTokens",
            input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        )
        cost_info = usage.get("cost", {})
        cost = cost_info.get("total", 0.0) if isinstance(cost_info, dict) else 0.0

        parent_span.set_attributes(
            {
                "llm.token_usage.input_tokens": input_tokens,
                "llm.token_usage.output_tokens": output_tokens,
                "llm.token_usage.total_tokens": total_tokens,
                "llm.token_usage.cache_read_tokens": usage.get("cacheRead", 0),
                "llm.token_usage.cache_write_tokens": usage.get("cacheWrite", 0),
                "modelID": self.session.llm_config.get_model_id(),
                "providerID": self.session.llm_config.provider,
                "cost": cost,
                "stop_reason": turn.stop_reason or "",
                "response_id": turn.response_id or "",
            }
        )

        results_by_id = {
            r["toolCallId"]: r for r in turn.tool_results if "toolCallId" in r
        }
        for call in turn.tool_calls:
            result = results_by_id.get(call.get("toolCallId", ""), {})
            self._create_tool_span(parent_span, call, result)

        perf_attrs = {
            f"perf.{k}": v
            for k, v in {
                "time_to_first_token_seconds": turn.time_to_first_token_seconds,
                "prompt_tokens_per_second": turn.prompt_tokens_per_second,
                "generation_tokens_per_second": turn.generation_tokens_per_second,
                "generation_time_seconds": turn.generation_time_seconds,
                "has_tool_calls": bool(turn.tool_calls),
            }.items()
            if v is not None
        }
        if perf_attrs:
            parent_span.set_attributes(perf_attrs)

        self.session.total_cost += cost
        self.session.total_input_tokens += input_tokens
        self.session.total_output_tokens += output_tokens
        self.session.total_cache_read_tokens += cache_read_tokens
        self.session.total_cache_write_tokens += cache_write_tokens
        self.session.total_tokens += total_tokens
        self.session.turns.append(turn)

    def _create_tool_span(
        self,
        parent_span: LiveSpan,
        call: dict,
        result: dict,
    ) -> None:
        """Create a child span for a pi tool execution."""
        tool_name = call.get("toolName", "unknown")
        call_id = call.get("toolCallId", "")

        child_span = mlflow.start_span_no_context(
            name=f"tool:{tool_name}:{call_id}",
            span_type=SpanType.TOOL,
            parent_span=parent_span,
            inputs={"tool": tool_name, "call_id": call_id, "input": call.get("args")},
        )

        if (tool_output := result.get("result")) is not None:
            child_span.set_outputs({"output": tool_output})

        if result.get("isError", False):
            child_span.set_status("ERROR")
            if self.session is not None:
                self.session.tool_error_count += 1

            error_message = result.get("result", "Tool execution failed")
            child_span.add_event(
                SpanEvent(
                    "exception",
                    attributes={
                        "error.message": str(error_message),
                        "error.type": "tool_execution_error",
                        "tool.name": tool_name,
                        "tool.call_id": call_id,
                    },
                )
            )

        self._accumulate_tool_call(tool_name)
        child_span.end()

    def _accumulate_tool_call(self, tool_name: str) -> None:
        """Increment tool call count and track consecutive calls for loop detection."""
        if self.session is None:
            return

        if tool_name == self._tool_last_name:
            self._tool_consecutive_calls += 1
            self._tool_max_consecutive_calls = max(
                self._tool_max_consecutive_calls,
                self._tool_consecutive_calls,
            )
        else:
            self._tool_consecutive_calls = 1
            self._tool_last_name = tool_name

        self.session.tool_call_counts.setdefault(tool_name, 0)
        self.session.tool_call_counts[tool_name] += 1

    def log_git_diff(self, diff_content: str) -> None:
        """Log git diff as an artifact."""
        if self.session is None:
            logger.warning("No active session to log git diff to")
            return

        self.session.git_diff = diff_content

    def log_system_prompt(self, prompt: str) -> None:
        """Log the system prompt for curation and traceability."""
        if self.session is None:
            logger.warning("No active session to log system prompt to")
            return

        self.session.system_prompt = prompt

    def log_error(self, error_message: str) -> None:
        """Log an error that occurred during the simulation.

        Marks the active turn span as ERROR so the trace shows red in MLflow.
        """
        if self.session is None:
            logger.warning("No active session to log error to")
            return

        self.session.error = error_message
        if self.session.turns:
            self.session.turns[-1].simulation_error = error_message
        if self._active_span is not None:
            self._active_span.set_status("ERROR")
            self._active_span.add_event(
                SpanEvent(
                    "exception",
                    attributes={
                        "error.message": error_message,
                        "error.type": "simulation_error",
                    },
                )
            )

    def log_stderr(self, stderr_content: str) -> None:
        """Log pi stderr output for debugging process failures."""
        if self.session is None:
            logger.warning("No active session to log stderr to")
            return

        if self.session.stderr_content:
            self.session.stderr_content += (
                "\n\n--- stderr continuation ---\n\n" + stderr_content
            )
        else:
            self.session.stderr_content = stderr_content

        if self.session.stderr_content:
            mlflow.log_text(self.session.stderr_content, "pi_stderr.log")

    def set_completion_status(self, status: str) -> None:
        """Set the completion status of the simulation."""
        if self.session is None:
            logger.warning("No active session to set completion status")
            return

        self.session.completion_status = status

    def _aggregate_performance(self) -> None:
        """Compute performance aggregates from session turns."""
        turns = self.session.turns

        def _mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        self.session.avg_time_to_first_token_seconds = _mean(
            [
                t.time_to_first_token_seconds
                for t in turns
                if t.time_to_first_token_seconds is not None
            ]
        )
        self.session.avg_prompt_tokens_per_second = _mean(
            [
                t.prompt_tokens_per_second
                for t in turns
                if t.prompt_tokens_per_second is not None
            ]
        )
        self.session.avg_generation_tokens_per_second = _mean(
            [
                t.generation_tokens_per_second
                for t in turns
                if t.generation_tokens_per_second is not None
            ]
        )
        gen_times = [
            t.generation_time_seconds
            for t in turns
            if t.generation_time_seconds is not None
        ]
        self.session.total_generation_time_seconds = (
            sum(gen_times) if gen_times else None
        )
        prompt_times = [
            t.prompt_processing_time_seconds
            for t in turns
            if t.prompt_processing_time_seconds is not None
        ]
        self.session.total_prompt_processing_time_seconds = (
            sum(prompt_times) if prompt_times else None
        )

        # Total inference time = prompt processing + generation (actual model compute)
        if (
            self.session.total_prompt_processing_time_seconds is not None
            and self.session.total_generation_time_seconds is not None
        ):
            self.session.total_model_compute_time_seconds = (
                self.session.total_prompt_processing_time_seconds
                + self.session.total_generation_time_seconds
            )

    def _log_tool_metrics(self) -> int:
        """Log tool call counts and return total."""
        total = 0
        for tool_name, count in self.session.tool_call_counts.items():
            mlflow.log_metric(f"tool_calls.{tool_name}", count)
            total += count
        mlflow.log_metric("tool_total_calls", total)
        mlflow.log_metric("tool_error_count", self.session.tool_error_count)
        return total

    def _log_performance_metrics(self) -> None:
        """Log computed performance metrics to MLflow."""
        metrics = [
            (
                "avg_time_to_first_token_seconds",
                self.session.avg_time_to_first_token_seconds,
            ),
            (
                "avg_prompt_tokens_per_second",
                self.session.avg_prompt_tokens_per_second,
            ),
            (
                "avg_generation_tokens_per_second",
                self.session.avg_generation_tokens_per_second,
            ),
            (
                "total_generation_time_seconds",
                self.session.total_generation_time_seconds,
            ),
            (
                "total_prompt_processing_time_seconds",
                self.session.total_prompt_processing_time_seconds,
            ),
            (
                "total_model_compute_time_seconds",
                self.session.total_model_compute_time_seconds,
            ),
        ]
        for name, value in metrics:
            if value is not None:
                mlflow.log_metric(name, value)

    def _log_derived_metrics(self, total_tool_calls: int, total_messages: int) -> None:
        """Compute and log derived session metrics."""
        self.session.tool_total_calls = total_tool_calls
        self.session.tool_loop_detected = (
            self._tool_max_consecutive_calls > self.session.tool_loop_threshold
        )
        self.session.tool_error_rate = self.session.tool_error_count / max(
            total_tool_calls, 1
        )
        total_messages = max(total_messages, 1)
        self.session.token_efficiency = self.session.total_tokens / total_messages
        self.session.cost_efficiency = self.session.total_cost / total_messages

        mlflow.log_metric(
            "tool_loop_detected", 1 if self.session.tool_loop_detected else 0
        )
        mlflow.log_metric(
            "tool_max_consecutive_calls", self._tool_max_consecutive_calls
        )
        mlflow.log_metric("tool_error_rate", self.session.tool_error_rate)
        mlflow.log_metric("token_efficiency", self.session.token_efficiency)
        mlflow.log_metric("cost_efficiency", self.session.cost_efficiency)

    def _flush(self) -> None:
        """Flush all accumulated data to MLflow."""
        if not self.run_id or self.session is None:
            return

        try:
            self.session.completed_at = datetime.now(timezone.utc)
            self.session.total_messages = len(self.session.turns)

            mlflow.log_metric("total_messages", self.session.total_messages)
            self._aggregate_performance()
            self._log_performance_metrics()

            if isinstance(self.session.started_at, datetime) and isinstance(
                self.session.completed_at, datetime
            ):
                wall = (
                    self.session.completed_at - self.session.started_at
                ).total_seconds()
                if wall > 0:
                    mlflow.log_metric("total_wall_clock_time_seconds", wall)
                    if self.session.total_tokens > 0:
                        mlflow.log_metric(
                            "total_tokens_per_second",
                            self.session.total_tokens / wall,
                        )

            compute_time = self.session.total_model_compute_time_seconds
            if (
                isinstance(compute_time, int | float)
                and compute_time > 0
                and self.session.total_tokens > 0
            ):
                mlflow.log_metric(
                    "model_compute_tokens_per_second",
                    self.session.total_tokens / compute_time,
                )

            mlflow.log_metric("total_cost", self.session.total_cost)
            mlflow.log_metric("total_input_tokens", self.session.total_input_tokens)
            mlflow.log_metric("total_output_tokens", self.session.total_output_tokens)
            mlflow.log_metric(
                "total_cache_read_tokens", self.session.total_cache_read_tokens
            )
            mlflow.log_metric(
                "total_cache_write_tokens", self.session.total_cache_write_tokens
            )
            mlflow.log_metric("total_tokens", self.session.total_tokens)

            if self.session.error:
                mlflow.log_metric("has_error", 1)

            total_tool_calls = self._log_tool_metrics()
            self._log_derived_metrics(total_tool_calls, self.session.total_messages)

            trace_session = self.session.model_dump(mode="json")
            mlflow.log_dict(trace_session, "trace_session.json")

            mlflow.set_tag("run.status", "error" if self.session.error else "success")
            mlflow.set_tag(
                "run.completion_status", self.session.completion_status or "unknown"
            )
            mlflow.set_tag("has_git_diff", "true" if self.session.git_diff else "false")
            date_str = self.session.completed_at.strftime("%Y-%m-%d")
            mlflow.set_tag("simulation.date", date_str)

            if self.session.git_diff:
                mlflow.log_text(self.session.git_diff, "git_diff.patch")

            if self.session.system_prompt:
                mlflow.log_text(self.session.system_prompt, "system_prompt.txt")

            logger.info("Flushed session data to MLflow run %s", self.run_id)
        finally:
            self._end_active_span()
