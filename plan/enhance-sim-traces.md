# Enhance Simulation Traces Plan

## Overview

Compute trace summaries during simulation execution and log them as artifacts. This eliminates the need for evaluators to re-parse full MLflow traces, reducing memory overhead and evaluation time.

## Goals

1. Detect tool loops in real-time during simulation
2. Pre-compute all trace metrics needed by judges
3. Store summary as portable JSON artifact
4. Enable zero-overhead evaluation (just read artifact)

## Changes Required

### 1. Update SimulationSession Dataclass

**File**: `src/beyond_vibes/simulations/mlflow.py`

Add tracking fields for loop detection and error indices:

```python
@dataclass
class SimulationSession:
    """Complete simulation session data."""

    sim_config: SimulationConfig
    model_config: ModelConfig
    quant_tag: str | None = None
    session_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    total_messages: int = 0
    total_time_seconds: float | None = None
    messages: list[MessageData] = field(default_factory=list)
    git_diff: str | None = None
    system_prompt: str | None = None
    error: str | None = None
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_error_count: int = 0
    completion_status: str | None = None
    # NEW: Loop detection tracking
    tool_loop_threshold: int = 3  # Configurable threshold for loop detection
    last_tool_name: str | None = None
    consecutive_tool_calls: int = 0
    max_consecutive_calls: int = 0
    # NEW: Error tracking for analysis
    error_message_indices: list[int] = field(default_factory=list)
```

### 2. Update _accumulate_tool_call Method

**File**: `src/beyond_vibes/simulations/mlflow.py`

Track consecutive tool calls to detect loops:

```python
def _accumulate_tool_call(self, tool_name: str, message_index: int) -> None:
    """Increment tool call count and track consecutive calls for loop detection.

    Args:
        tool_name: Name of the tool
        message_index: Index of the message containing this tool call

    """
    if self.session is None:
        return

    # Track consecutive calls for loop detection
    if tool_name == self.session.last_tool_name:
        self.session.consecutive_tool_calls += 1
        self.session.max_consecutive_calls = max(
            self.session.max_consecutive_calls,
            self.session.consecutive_tool_calls
        )
    else:
        self.session.consecutive_tool_calls = 1
        self.session.last_tool_name = tool_name

    # Original count tracking
    if tool_name not in self.session.tool_call_counts:
        self.session.tool_call_counts[tool_name] = 0

    self.session.tool_call_counts[tool_name] += 1
```

### 3. Update _handle_tool_errors Method

**File**: `src/beyond_vibes/simulations/mlflow.py`

Track which message indices contain errors:

```python
def _handle_tool_errors(
    self,
    child_span: Any,
    state: dict,
    tool_name: str,
    call_id: str,
    tool_output: Any,
    message_index: int,  # NEW parameter
) -> None:
    """Check for tool errors and add exception event if found.

    Args:
        child_span: The span to mark with error status
        state: Tool state dictionary
        tool_name: Name of the tool
        call_id: Unique call identifier
        tool_output: Output from the tool execution
        message_index: Index of the parent message (NEW)

    """
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
        # Track which message had the error
        self.session.error_message_indices.append(message_index)

    # ... rest of existing error handling logic ...
```

### 4. Update _create_tool_child_span Method

**File**: `src/beyond_vibes/simulations/mlflow.py`

Pass message_index to error handler and tool accumulator:

```python
def _create_tool_child_span(
    self,
    parent_span: Any,
    part: dict,
    parent_start_ns: int | None,
    parent_end_ns: int | None,
    message_index: int,  # NEW parameter
) -> None:
    """Create a child span for a tool call with latency and error tracking.

    Args:
        parent_span: Parent span for this tool call
        part: Tool part dictionary from message
        parent_start_ns: Parent span start timestamp
        parent_end_ns: Parent span end timestamp
        message_index: Index of the parent message (NEW)

    """
    tool_name = part.get("tool", "unknown")
    call_id = part.get("callID", "")
    state = part.get("state", {})
    span_name = f"tool:{tool_name}:{call_id}"

    # ... existing timestamp extraction ...

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

    # Handle errors and track tool call count (with message_index)
    self._handle_tool_errors(child_span, state, tool_name, call_id, tool_output, message_index)
    self._accumulate_tool_call(tool_name, message_index)

    child_span.end(end_time_ns=child_end_ns)
```

### 5. Update log_message Method

**File**: `src/beyond_vibes/simulations/mlflow.py`

Pass message_index to child span creation:

```python
def log_message(self, message: dict) -> None:
    """Log a raw message as a span in the trace with accurate timestamps."""
    if self.session is None:
        logger.warning("No active session to log message to")
        return

    message_index = len(self.session.messages)

    # ... existing timestamp extraction ...

    # Create parent span with session_id in metadata
    parent_span = mlflow.start_span_no_context(
        name=f"message_{message_index}",
        span_type=SpanType.AGENT,
        start_time_ns=parent_start_ns,
        metadata={"mlflow.trace.session": self.session.session_id},
    )

    # ... existing content extraction ...

    # Process message parts and create tool child spans
    parts = message.get("parts", [])
    for part in parts:
        part_type = part.get("type", "")

        # ... existing text/reasoning handling ...

        # Create child spans for tool calls (with message_index)
        elif part_type == "tool":
            self._create_tool_child_span(
                parent_span,
                part,
                parent_start_ns,
                parent_end_ns,
                message_index,  # NEW
            )

            # Check for tool errors
            state = part.get("state", {})
            if state.get("status") == "error":
                has_tool_error = True

    # ... rest of existing method ...
```

### 6. Update _flush Method

**File**: `src/beyond_vibes/simulations/mlflow.py`

Compute and log trace summary as JSON artifact:

```python
def _flush(self) -> None:
    """Flush all accumulated data to MLflow."""
    if not self.run_id or self.session is None:
        return

    self.session.completed_at = datetime.now()
    if self.session.started_at and self.session.completed_at:
        self.session.total_time_seconds = (
            self.session.completed_at - self.session.started_at
        ).total_seconds()

    self.session.total_messages = len(self.session.messages)

    # ... existing metric logging ...

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
    mlflow.log_metric("total_tool_calls", total_tool_calls)

    # Log tool error count
    mlflow.log_metric("tool_error_count", self.session.tool_error_count)

    # NEW: Compute and log trace summary
    trace_summary = {
        "total_messages": self.session.total_messages,
        "total_tool_calls": total_tool_calls,
        "tool_error_count": self.session.tool_error_count,
        "total_tokens": self.session.total_tokens,
        "total_cost": self.session.total_cost,
        "tool_loop_detected": self.session.max_consecutive_calls > self.session.tool_loop_threshold,
        "tool_loop_threshold": self.session.tool_loop_threshold,
        "max_consecutive_calls": self.session.max_consecutive_calls,
        "error_rate": (
            self.session.tool_error_count / max(total_tool_calls, 1)
        ),
        "token_efficiency": (
            self.session.total_tokens / max(self.session.total_messages, 1)
        ),
        "cost_efficiency": (
            self.session.total_cost / max(total_tool_calls, 1)
        ),
        "error_message_indices": self.session.error_message_indices,
    }

    # Log trace summary as JSON artifact for evaluators
    mlflow.log_dict(trace_summary, "trace_summary.json")

    # ... existing tag setting ...

    mlflow.set_tag("run.status", "error" if self.session.error else "success")
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
```

### 7. Update Extractor to Read Trace Summary

**File**: `src/beyond_vibes/evaluators/extractor.py`

Read pre-computed summary instead of analyzing spans:

```python
def extract_run_data(run_id: str) -> JudgeInput:
    """Extract standardized input from MLflow run."""
    # Load run
    run = mlflow.get_run(run_id)
    
    # Load artifacts
    system_prompt = _load_artifact(run_id, "system_prompt.txt")
    git_diff = _load_artifact(run_id, "git_diff.patch")
    
    # Load pre-computed trace summary (NEW)
    trace_summary = _load_trace_summary(run_id)
    
    # Extract final message from last span (still needed)
    final_message = _extract_final_message(run_id)
    
    # Build JudgeInput
    return JudgeInput(
        run_id=run_id,
        task_name=run.data.tags.get("task.name", ""),
        archetype=run.data.tags.get("task.archetype", ""),
        system_prompt=system_prompt or "",
        task_prompt=run.data.params.get("task.prompt", ""),
        final_message=final_message,
        git_diff=git_diff,
        trace_summary=trace_summary,  # NEW: Use pre-computed summary
    )


def _load_trace_summary(run_id: str) -> dict:
    """Load pre-computed trace summary from run artifacts."""
    try:
        return mlflow.artifacts.load_dict(
            f"runs:/{run_id}/trace_summary.json"
        )
    except Exception:
        # Fallback: return empty summary if not found
        logger.warning(f"No trace_summary.json found for run {run_id}")
        return {
            "total_messages": 0,
            "total_tool_calls": 0,
            "tool_error_count": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "tool_loop_detected": False,
            "tool_loop_threshold": 3,
            "max_consecutive_calls": 0,
            "error_rate": 0.0,
            "token_efficiency": 0.0,
            "cost_efficiency": 0.0,
            "error_message_indices": [],
        }
```

## Trace Summary Schema

The `trace_summary.json` artifact contains:

```json
{
  "total_messages": 25,
  "total_tool_calls": 42,
  "tool_error_count": 3,
  "total_tokens": 15420,
  "total_cost": 0.05,
  "tool_loop_detected": true,
  "tool_loop_threshold": 3,
  "max_consecutive_calls": 5,
  "error_rate": 0.071,
  "token_efficiency": 616.8,
  "cost_efficiency": 0.00119,
  "error_message_indices": [5, 12, 23]
}
```

**Field Descriptions:**

- `total_messages`: Number of messages in the simulation
- `total_tool_calls`: Total number of tool invocations
- `tool_error_count`: Number of tool calls that resulted in errors
- `total_tokens`: Total tokens consumed (input + output)
- `total_cost`: Total cost of the simulation in USD
- `tool_loop_detected`: True if any tool was called >threshold times consecutively
- `tool_loop_threshold`: Configurable threshold for loop detection (default: 3)
- `max_consecutive_calls`: Maximum consecutive calls of the same tool
- `error_rate`: tool_error_count / total_tool_calls
- `token_efficiency`: total_tokens / total_messages (tokens per message)
- `cost_efficiency`: total_cost / total_tool_calls (cost per tool call)
- `error_message_indices`: Message indices where errors occurred (for analysis)

## Benefits

1. **Zero evaluation overhead**: Evaluators read JSON artifact instead of parsing spans
2. **Real-time loop detection**: Detects tool loops as they happen during simulation
3. **Portable metrics**: Summary travels with the run, works across MLflow instances
4. **Extensible**: Easy to add new metrics to summary without changing evaluators
5. **Backward compatible**: Falls back gracefully if summary is missing (legacy runs)

## Testing Strategy

1. **Unit test**: Verify loop detection logic with mock tool sequences
2. **Integration test**: Run simulation, verify trace_summary.json artifact exists
3. **Validation**: Confirm summary values match actual trace data
4. **Regression test**: Ensure existing runs without summary still work (fallback)

## Success Criteria

- [ ] `trace_summary.json` artifact logged to every simulation run
- [ ] Tool loops detected in real-time (max_consecutive_calls > threshold)
- [ ] Error indices correctly tracked per message
- [ ] Evaluator reads summary without parsing full trace
- [ ] Fallback works for runs created before this change
- [ ] All existing tests pass
