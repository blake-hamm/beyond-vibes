"""MLflow tracing integration for simulation runs."""

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.span_event import SpanEvent
from pydantic import BaseModel, Field

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.pi_dev import TurnData

logger = logging.getLogger(__name__)


def generate_session_id(model_config: ModelConfig, quant_tag: str | None = None) -> str:
    """Generate a unique session ID from model config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant = quant_tag or "fp16"
    return f"{model_config.name}_{quant}_{timestamp}"


@dataclass
class MessageData:
    """Data for a single message in the simulation."""

    message_index: int
    timestamp: datetime
    raw_message: dict | None = None


@dataclass
class MessagePerformanceMetrics:
    """Performance metrics for a single message."""

    ttft_ms: float | None = None
    generation_time_ms: float | None = None
    generation_tps: float | None = None
    prompt_tps: float | None = None
    prompt_processing_ms: float | None = None
    output_tokens: int = 0
    has_tool_calls: bool = False


class SimulationSession(BaseModel):
    """Complete simulation session data."""

    model_config = {"arbitrary_types_allowed": True}

    sim_config: SimulationConfig
    llm_config: ModelConfig
    quant_tag: str | None = None
    session_id: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    total_messages: int = 0
    total_time_seconds: float | None = None
    messages: list[MessageData] = Field(default_factory=list)
    git_diff: str | None = None
    system_prompt: str | None = None
    error: str | None = None
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    tool_error_count: int = 0
    completion_status: str | None = None
    tool_loop_threshold: int = 3
    tool_last_name: str | None = None
    tool_consecutive_calls: int = 0
    tool_max_consecutive_calls: int = 0
    error_message_indices: list[int] = Field(default_factory=list)
    tool_total_calls: int | None = None
    tool_loop_detected: bool | None = None
    tool_error_rate: float | None = None
    token_efficiency: float | None = None
    cost_efficiency: float | None = None
    message_metrics: list[MessagePerformanceMetrics] = Field(default_factory=list)
    avg_ttft_ms: float | None = None
    avg_prompt_tps: float | None = None
    avg_generation_tps: float | None = None
    total_generation_time_ms: float | None = None
    total_prompt_processing_ms: float | None = None


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

    @contextmanager
    def log_simulation(
        self,
        sim_config: SimulationConfig,
        model_config: ModelConfig,
    ) -> Generator["MlflowTracer", Any, None]:
        """Context manager for tracing a simulation run."""
        session_id = generate_session_id(model_config, self.quant_tag)

        self.session = SimulationSession(
            sim_config=sim_config,
            llm_config=model_config,
            quant_tag=self.quant_tag,
            session_id=session_id,
            started_at=datetime.now(),
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

                # Parameters (performance analysis)
                mlflow.log_param("model.name", model_config.name)
                mlflow.log_param("model.provider", model_config.provider)
                mlflow.log_param("model.model_id", model_config.get_model_id())
                if model_config.repo_id:
                    mlflow.log_param("model.repo_id", model_config.repo_id)
                if self.quant_tag:
                    mlflow.log_param("model.quant", self.quant_tag)
                if self.container_tag:
                    mlflow.log_param("runtime.container", self.container_tag)

                # Tags (filtering)
                mlflow.set_tag("task.name", sim_config.name)
                mlflow.set_tag("task.archetype", sim_config.archetype)
                mlflow.set_tag("repository.url", sim_config.repository.url)
                mlflow.set_tag("repository.branch", sim_config.repository.branch)

                yield self

                self._flush()

        except Exception as e:
            logger.error("Failed to start MLflow run: %s", e)
            raise

    def log_message(self, message: dict) -> None:  # noqa: PLR0912, PLR0915
        """Log a raw message as a span in the trace with accurate timestamps.

        Creates a parent span for the message and child spans for each tool call,
        with proper error handling and latency tracking.
        """
        if self.session is None:
            logger.warning("No active session to log message to")
            return

        message_index = len(self.session.messages)

        # Extract timestamps from message metadata using helper
        time_info = message.get("info", {}).get("time", {})
        parent_start_ns, parent_end_ns = self._extract_timestamps_ns(
            time_info, "created", "completed"
        )

        # Create parent span with session_id in metadata
        parent_span = mlflow.start_span_no_context(
            name=f"message_{message_index}",
            span_type=SpanType.AGENT,
            start_time_ns=parent_start_ns,
            attributes={"run_id": self.run_id},
            metadata={"mlflow.trace.session": self.session.session_id},
        )

        # Extract readable content from message parts
        readable_parts = []
        has_tool_error = False

        # Process message parts and create tool child spans
        parts = message.get("parts", [])
        for part in parts:
            part_type = part.get("type", "")

            # Collect readable content (text and reasoning)
            if part_type == "text":
                text = part.get("text", "")
                if text:
                    readable_parts.append({"type": "text", "content": text})
            elif part_type == "reasoning":
                text = part.get("text", "")
                if text:
                    readable_parts.append({"type": "reasoning", "content": text})

            # Create child spans for tool calls
            elif part_type == "tool":
                self._create_tool_child_span(
                    parent_span,
                    part,
                    parent_start_ns,
                    parent_end_ns,
                    message_index,
                )

                # Check for tool errors
                state = part.get("state", {})
                if state.get("status") == "error":
                    has_tool_error = True

        # Set parent span inputs
        parent_span.set_inputs(
            {
                "message_index": message_index,
                "message_id": message.get("info", {}).get("id", ""),
                "role": message.get("info", {}).get("role", ""),
            }
        )

        # Set readable outputs (assistant text and reasoning only)
        parent_span.set_outputs({"content": readable_parts})

        # Set turn-specific configuration attributes
        info = message.get("info", {})
        cost = info.get("cost", 0.0)
        tokens = info.get("tokens", {})
        input_tokens = tokens.get("input", 0)
        output_tokens = tokens.get("output", 0)
        total_tokens = input_tokens + output_tokens

        parent_span.set_attributes(
            {
                "llm.token_usage.input_tokens": input_tokens,
                "llm.token_usage.output_tokens": output_tokens,
                "llm.token_usage.total_tokens": total_tokens,
                "modelID": self.session.llm_config.get_model_id(),
                "providerID": self.session.llm_config.provider,
                "cost": cost,
            }
        )

        # Calculate and log per-message performance metrics
        perf_metrics = self._calculate_message_metrics(message, output_tokens)
        if perf_metrics:
            perf_attrs = {}
            if perf_metrics.ttft_ms is not None:
                perf_attrs["perf.ttft_ms"] = perf_metrics.ttft_ms
            if perf_metrics.generation_time_ms is not None:
                perf_attrs["perf.generation_time_ms"] = perf_metrics.generation_time_ms
            if perf_metrics.generation_tps is not None:
                perf_attrs["perf.generation_tps"] = perf_metrics.generation_tps
            perf_attrs["perf.has_tool_calls"] = perf_metrics.has_tool_calls
            if perf_attrs:
                parent_span.set_attributes(perf_attrs)
            self.session.message_metrics.append(perf_metrics)

        # Propagate tool errors to parent span
        if has_tool_error:
            parent_span.set_status("ERROR")

        # Store raw message as attribute
        parent_span.set_attributes({"raw_message_json": json.dumps(message)})

        # End parent span with custom timestamp
        parent_span.end(end_time_ns=parent_end_ns)

        # Accumulate cost and token totals from message info
        self.session.total_cost += cost
        self.session.total_input_tokens += input_tokens
        self.session.total_output_tokens += output_tokens
        self.session.total_tokens += total_tokens

        # Append message data to session
        message_data = MessageData(
            message_index=message_index,
            timestamp=datetime.now(),
            raw_message=message,
        )
        self.session.messages.append(message_data)

    def log_turn(self, turn: TurnData) -> None:  # noqa: PLR0912, PLR0915
        """Log a native pi turn as a span in the trace.

        Creates a parent span for the turn and child spans for each tool
        execution. Accumulates session metrics from turn data.
        """
        if self.session is None:
            logger.warning("No active session to log turn to")
            return

        turn_index = turn.turn_index
        span_name = f"turn_{turn_index}"

        parent_span = mlflow.start_span_no_context(
            name=span_name,
            span_type=SpanType.AGENT,
            attributes={"run_id": self.run_id},
            metadata={"mlflow.trace.session": self.session.session_id},
        )

        # Extract readable content from turn.content blocks
        readable_parts = []
        for block in turn.content:
            block_type = block.get("type", "")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    readable_parts.append({"type": "text", "content": text})
            elif block_type == "thinking":
                text = block.get("thinking", "")
                if text:
                    readable_parts.append({"type": "thinking", "content": text})

        parent_span.set_inputs({"turn_index": turn_index, "role": "assistant"})
        parent_span.set_outputs({"content": readable_parts})

        # Token usage from pi-native usage dict
        usage = turn.usage or {}
        input_tokens = usage.get("input", 0)
        output_tokens = usage.get("output", 0)
        total_tokens = usage.get("totalTokens", input_tokens + output_tokens)
        cost_info = usage.get("cost", {})
        cost = cost_info.get("total", 0.0) if isinstance(cost_info, dict) else 0.0

        cache_read = usage.get("cacheRead", 0)
        cache_write = usage.get("cacheWrite", 0)
        response_id = turn.raw_message.get("responseId", "") if turn.raw_message else ""

        parent_span.set_attributes(
            {
                "llm.token_usage.input_tokens": input_tokens,
                "llm.token_usage.output_tokens": output_tokens,
                "llm.token_usage.total_tokens": total_tokens,
                "llm.token_usage.cache_read_tokens": cache_read,
                "llm.token_usage.cache_write_tokens": cache_write,
                "modelID": self.session.llm_config.get_model_id(),
                "providerID": self.session.llm_config.provider,
                "cost": cost,
                "stop_reason": turn.stop_reason or "",
                "response_id": response_id,
            }
        )

        # Create child spans for tool executions
        results_by_id = {
            r["toolCallId"]: r for r in turn.tool_results if "toolCallId" in r
        }
        for call in turn.tool_calls:
            call_id = call.get("toolCallId", "")
            tool_name = call.get("toolName", "unknown")
            result = results_by_id.get(call_id, {})
            self._create_tool_span_from_pi(
                parent_span, tool_name, call_id, call.get("args"), result, turn_index
            )

        raw_json = json.dumps(turn.raw_message) if turn.raw_message else ""
        max_raw_len = 4096
        if len(raw_json) > max_raw_len:
            raw_json = raw_json[:max_raw_len] + "... [truncated]"
        parent_span.set_attributes({"raw_message_json": raw_json})

        # Compute and log performance metrics from TurnData
        perf = MessagePerformanceMetrics(
            ttft_ms=turn.ttft_ms,
            generation_time_ms=turn.generation_time_ms,
            generation_tps=turn.generation_tps,
            prompt_tps=turn.prompt_tps,
            prompt_processing_ms=turn.prompt_processing_ms,
            output_tokens=output_tokens,
            has_tool_calls=bool(turn.tool_calls),
        )
        self.session.message_metrics.append(perf)

        perf_attrs = {}
        if perf.ttft_ms is not None:
            perf_attrs["perf.ttft_ms"] = perf.ttft_ms
        if perf.prompt_tps is not None:
            perf_attrs["perf.prompt_tps"] = perf.prompt_tps
        if perf.generation_tps is not None:
            perf_attrs["perf.generation_tps"] = perf.generation_tps
        if perf.generation_time_ms is not None:
            perf_attrs["perf.generation_time_ms"] = perf.generation_time_ms
        perf_attrs["perf.has_tool_calls"] = perf.has_tool_calls
        if perf_attrs:
            parent_span.set_attributes(perf_attrs)

        parent_span.end()

        # Accumulate session totals
        self.session.total_cost += cost
        self.session.total_input_tokens += input_tokens
        self.session.total_output_tokens += output_tokens
        self.session.total_tokens += total_tokens

        # Append message data to session for trace_session.json
        message_data = MessageData(
            message_index=turn_index,
            timestamp=datetime.now(),
            raw_message=turn.raw_message or {},
        )
        self.session.messages.append(message_data)

    def _create_tool_span_from_pi(  # noqa: PLR0913
        self,
        parent_span: Any,  # noqa: ANN401
        tool_name: str,
        call_id: str,
        tool_input: Any,  # noqa: ANN401
        result: dict,
        turn_index: int,
    ) -> None:
        """Create a child span for a pi tool execution."""
        span_name = f"tool:{tool_name}:{call_id}"

        child_span = mlflow.start_span_no_context(
            name=span_name,
            span_type=SpanType.TOOL,
            parent_span=parent_span,
            inputs={"tool": tool_name, "call_id": call_id, "input": tool_input},
        )

        tool_output = result.get("result")
        if tool_output is not None:
            child_span.set_outputs({"output": tool_output})

        is_error = result.get("isError", False)
        if is_error:
            child_span.set_status("ERROR")
            if self.session is not None:
                self.session.tool_error_count += 1
                self.session.error_message_indices.append(turn_index)

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

        self._accumulate_tool_call(tool_name, turn_index)
        child_span.end()

    def _extract_timestamps_ns(
        self,
        time_info: dict,
        start_key: str,
        end_key: str,
        fallback_start_ns: int | None = None,
        fallback_end_ns: int | None = None,
    ) -> tuple[int | None, int | None]:
        """Extract timestamps from time info and convert ms to ns with fallbacks."""
        start_ms = time_info.get(start_key)
        end_ms = time_info.get(end_key)

        start_ns = (
            int(start_ms * 1_000_000) if start_ms is not None else fallback_start_ns
        )
        end_ns = int(end_ms * 1_000_000) if end_ms is not None else fallback_end_ns

        return start_ns, end_ns

    def _handle_tool_errors(  # noqa: PLR0913
        self,
        child_span: Any,  # noqa: ANN401
        state: dict,
        tool_name: str,
        call_id: str,
        tool_output: Any,  # noqa: ANN401
        message_index: int,
    ) -> None:
        """Check for tool errors and add exception event if found."""
        status = state.get("status", "")
        metadata = state.get("metadata", {})
        exit_code = metadata.get("exit", 0)

        is_explicit_error = status == "error"
        is_nonzero_exit = exit_code > 0

        if not is_explicit_error and not is_nonzero_exit:
            return

        child_span.set_status("ERROR")

        if self.session is not None:
            self.session.tool_error_count += 1
            self.session.error_message_indices.append(message_index)

        if is_explicit_error:
            error_type = "execution_error"
            error_message = state.get("error", "")
            if not error_message:
                error_message = tool_output if tool_output else "Tool execution failed"
        else:
            error_type = "non_zero_exit"
            error_message = (
                tool_output if tool_output else f"Command exited with code {exit_code}"
            )

        event_attributes = {
            "error.message": error_message,
            "error.type": error_type,
            "tool.name": tool_name,
            "tool.call_id": call_id,
            "tool.status": status,
        }

        if exit_code > 0:
            event_attributes["tool.exit_code"] = exit_code

        if "error_code" in state:
            event_attributes["error.code"] = state["error_code"]

        child_span.add_event(SpanEvent("exception", attributes=event_attributes))

    def _create_tool_child_span(
        self,
        parent_span: Any,  # noqa: ANN401
        part: dict,
        parent_start_ns: int | None,
        parent_end_ns: int | None,
        message_index: int,
    ) -> None:
        """Create a child span for a tool call with latency and error tracking."""
        tool_name = part.get("tool", "unknown")
        call_id = part.get("callID", "")
        state = part.get("state", {})
        span_name = f"tool:{tool_name}:{call_id}"

        # Extract timestamps with fallbacks to parent span times
        time_info = state.get("time", {})
        child_start_ns, child_end_ns = self._extract_timestamps_ns(
            time_info, "start", "end", parent_start_ns, parent_end_ns
        )

        # Create child span with explicit parent and timestamp control
        tool_input = state.get("input", {})
        child_span = mlflow.start_span_no_context(
            name=span_name,
            span_type=SpanType.TOOL,
            parent_span=parent_span,
            inputs={"tool": tool_name, "call_id": call_id, "input": tool_input},
            start_time_ns=child_start_ns,
        )

        tool_output = state.get("output")
        if tool_output is not None:
            child_span.set_outputs({"output": tool_output})

        # Handle errors and track tool call count
        self._handle_tool_errors(
            child_span, state, tool_name, call_id, tool_output, message_index
        )
        self._accumulate_tool_call(tool_name, message_index)

        child_span.end(end_time_ns=child_end_ns)

    def _accumulate_tool_call(
        self,
        tool_name: str,
        message_index: int,  # noqa: ARG002
    ) -> None:
        """Increment tool call count and track consecutive calls for loop detection."""
        if self.session is None:
            return

        # Track consecutive calls for loop detection
        if tool_name == self.session.tool_last_name:
            self.session.tool_consecutive_calls += 1
            self.session.tool_max_consecutive_calls = max(
                self.session.tool_max_consecutive_calls,
                self.session.tool_consecutive_calls,
            )
        else:
            self.session.tool_consecutive_calls = 1
            self.session.tool_last_name = tool_name

        # Original count tracking
        if tool_name not in self.session.tool_call_counts:
            self.session.tool_call_counts[tool_name] = 0

        self.session.tool_call_counts[tool_name] += 1

    def _calculate_message_metrics(
        self,
        message: dict,
        output_tokens: int,
    ) -> MessagePerformanceMetrics | None:
        """Calculate performance metrics (TTFT, TPS, generation time) for a message.

        Excludes tool execution time from calculations.
        Returns None if timing data is insufficient.
        """
        info = message.get("info", {})
        time_info = info.get("time", {})
        message_created_ms = time_info.get("created")
        message_completed_ms = time_info.get("completed")

        if message_created_ms is None or message_completed_ms is None:
            return None

        # Find the first content part (text or reasoning) with timing
        parts = message.get("parts", [])
        first_content_start_ms = None
        has_tool_calls = False

        for part in parts:
            part_type = part.get("type", "")
            if part_type == "tool":
                has_tool_calls = True

            if part_type in ("text", "reasoning"):
                part_time = part.get("time")
                if part_time and part_time.get("start"):
                    first_content_start_ms = part_time.get("start")
                    break

        if first_content_start_ms is None:
            return None

        # Calculate TTFT: time from message created to first content
        ttft_ms = first_content_start_ms - message_created_ms

        # For TPS calculation, exclude messages with tool calls
        # (tool execution contaminates the generation time)
        tps = None
        generation_time_ms = None

        if not has_tool_calls and output_tokens > 0:
            generation_time_ms = message_completed_ms - first_content_start_ms
            if generation_time_ms > 0:
                tps = (output_tokens / generation_time_ms) * 1000  # tokens per second

        return MessagePerformanceMetrics(
            ttft_ms=ttft_ms,
            generation_time_ms=generation_time_ms,
            generation_tps=tps,
            output_tokens=output_tokens,
            has_tool_calls=has_tool_calls,
        )

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
        """Log an error that occurred during the simulation."""
        if self.session is None:
            logger.warning("No active session to log error to")
            return

        self.session.error = error_message

    def set_completion_status(self, status: str) -> None:
        """Set the completion status of the simulation.

        Args:
            status: One of 'completed', 'max_turns', or 'error'

        """
        if self.session is None:
            logger.warning("No active session to set completion status")
            return

        self.session.completion_status = status

    def _flush(self) -> None:  # noqa: PLR0912, PLR0915
        """Flush all accumulated data to MLflow."""
        if not self.run_id or self.session is None:
            return

        self.session.completed_at = datetime.now()
        if self.session.started_at and self.session.completed_at:
            self.session.total_time_seconds = (
                self.session.completed_at - self.session.started_at
            ).total_seconds()

        self.session.total_messages = len(self.session.messages)

        mlflow.log_metric("total_messages", self.session.total_messages)

        if self.session.total_time_seconds is not None:
            mlflow.log_metric("total_time_seconds", self.session.total_time_seconds)

        # Log accumulated cost and token metrics
        mlflow.log_metric("total_cost", self.session.total_cost)
        mlflow.log_metric("total_input_tokens", self.session.total_input_tokens)
        mlflow.log_metric("total_output_tokens", self.session.total_output_tokens)
        mlflow.log_metric("total_tokens", self.session.total_tokens)

        if self.session.error:
            mlflow.log_metric("has_error", 1)

        # Log tool call counts per tool and total
        total_tool_calls = 0
        for tool_name, count in self.session.tool_call_counts.items():
            mlflow.log_metric(f"tool_calls.{tool_name}", count)
            total_tool_calls += count
        mlflow.log_metric("tool_total_calls", total_tool_calls)

        # Log tool error count
        mlflow.log_metric("tool_error_count", self.session.tool_error_count)

        # Compute derived metrics and set on session
        self.session.total_messages = len(self.session.messages)
        self.session.tool_total_calls = total_tool_calls
        self.session.tool_loop_detected = (
            self.session.tool_max_consecutive_calls > self.session.tool_loop_threshold
        )
        self.session.tool_error_rate = self.session.tool_error_count / max(
            total_tool_calls, 1
        )
        self.session.token_efficiency = self.session.total_tokens / max(
            self.session.total_messages, 1
        )
        self.session.cost_efficiency = self.session.total_cost / max(
            self.session.total_messages, 1
        )

        # Calculate aggregated performance metrics
        if self.session.message_metrics:
            ttft_values = [
                m.ttft_ms for m in self.session.message_metrics if m.ttft_ms is not None
            ]
            generation_tps_values = [
                m.generation_tps
                for m in self.session.message_metrics
                if m.generation_tps is not None
            ]
            prompt_tps_values = [
                m.prompt_tps
                for m in self.session.message_metrics
                if m.prompt_tps is not None
            ]
            generation_times = [
                m.generation_time_ms
                for m in self.session.message_metrics
                if m.generation_time_ms is not None
            ]
            prompt_processing_times = [
                m.prompt_processing_ms
                for m in self.session.message_metrics
                if m.prompt_processing_ms is not None
            ]

            if ttft_values:
                self.session.avg_ttft_ms = sum(ttft_values) / len(ttft_values)
            if generation_tps_values:
                self.session.avg_generation_tps = sum(generation_tps_values) / len(
                    generation_tps_values
                )
            if prompt_tps_values:
                self.session.avg_prompt_tps = sum(prompt_tps_values) / len(
                    prompt_tps_values
                )
            if generation_times:
                self.session.total_generation_time_ms = sum(generation_times)
            if prompt_processing_times:
                self.session.total_prompt_processing_ms = sum(prompt_processing_times)

        # Log trace summary as JSON artifact for evaluators
        trace_session = self.session.model_dump(mode="json")
        mlflow.log_dict(trace_session, "trace_session.json")

        # Log computed metrics for MLflow UI
        mlflow.log_metric(
            "tool_loop_detected", 1 if self.session.tool_loop_detected else 0
        )
        mlflow.log_metric(
            "tool_max_consecutive_calls", self.session.tool_max_consecutive_calls
        )
        mlflow.log_metric("tool_error_rate", self.session.tool_error_rate)
        mlflow.log_metric("token_efficiency", self.session.token_efficiency)
        mlflow.log_metric("cost_efficiency", self.session.cost_efficiency)

        # Log performance metrics
        if self.session.avg_ttft_ms is not None:
            mlflow.log_metric("avg_ttft_ms", self.session.avg_ttft_ms)
        if self.session.avg_prompt_tps is not None:
            mlflow.log_metric("avg_prompt_tps", self.session.avg_prompt_tps)
        if self.session.avg_generation_tps is not None:
            mlflow.log_metric("avg_generation_tps", self.session.avg_generation_tps)
        if self.session.total_generation_time_ms is not None:
            mlflow.log_metric(
                "total_generation_time_ms", self.session.total_generation_time_ms
            )
        if self.session.total_prompt_processing_ms is not None:
            mlflow.log_metric(
                "total_prompt_processing_ms",
                self.session.total_prompt_processing_ms,
            )

        # Set tags for filtering (in addition to metrics)
        mlflow.set_tag("run.status", "error" if self.session.error else "success")
        # completion_status: completed, max_turns, or error
        completion_status = self.session.completion_status or "unknown"
        mlflow.set_tag("run.completion_status", completion_status)
        mlflow.set_tag("has_git_diff", "true" if self.session.git_diff else "false")
        date_str = self.session.completed_at.strftime("%Y-%m-%d")
        mlflow.set_tag("simulation.date", date_str)

        if self.session.git_diff:
            mlflow.log_text(self.session.git_diff, "git_diff.patch")

        if self.session.system_prompt:
            mlflow.log_text(self.session.system_prompt, "system_prompt.txt")

        logger.info("Flushed session data to MLflow run %s", self.run_id)
