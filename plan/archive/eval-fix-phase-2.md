# Evaluation Fix Phase 2: Smart Trace Truncation

## Overview

**Problem**: When `input_artifact` is "trace" or when trace is used as context, the entire `trace_session.json` is serialized to JSON, resulting in 293,922+ tokens which exceeds gpt-4o-mini's 128K token limit.

**Root Cause**: The current implementation in `runner.py:_prepare_eval_data()` dumps the full trace:
```python
output_content = json.dumps(judge_input.trace, indent=2)  # Too large!
```

**Solution**: Implement intelligent truncation that keeps:
- First 3 messages (setup/system/context)
- Last 3 messages (final output/conclusion)
- 2-3 deterministic middle messages (at 25%, 50%, 75% positions)

This reduces token count by ~90% while preserving conversation flow and key decision points.

## Changes Required

### File: `src/beyond_vibes/evaluations/runner.py`

#### 1. Add Truncation Helper Method

```python
def _truncate_trace(
    self,
    trace: dict[str, Any],
    keep_first: int = 3,
    keep_last: int = 3,
    middle_positions: list[float] = [0.25, 0.5, 0.75],
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
    indices_to_keep.update(range(max(0, total_messages - keep_last), total_messages))
    
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
            truncated_messages.append({
                "type": "truncation_marker",
                "omitted_count": omitted_count,
                "note": f"... {omitted_count} messages omitted ..."
            })
        
        truncated_messages.append(messages[idx])
        last_idx = idx
    
    # Build truncated trace
    truncated_trace = {
        **trace,
        "messages": truncated_messages,
        "truncated": True,
        "original_message_count": total_messages,
        "retained_message_count": len(truncated_messages),
        "truncation_config": {
            "keep_first": keep_first,
            "keep_last": keep_last,
            "middle_positions": middle_positions,
        }
    }
    
    return truncated_trace
```

#### 2. Modify `_prepare_eval_data()` to Use Truncation

```python
def _prepare_eval_data(
    self,
    judge_input: JudgeInput,
    input_artifact: str,
) -> dict[str, Any]:
    """Convert JudgeInput to format expected by mlflow.evaluate."""
    
    if input_artifact == "git_diff":
        output_content = judge_input.git_diff or ""
        context = None
        
    elif input_artifact == "final_message":
        output_content = judge_input.final_message
        # Truncate trace for context (keep first/last/middle)
        if judge_input.trace:
            truncated = self._truncate_trace(judge_input.trace)
            context = json.dumps(truncated, indent=2)
        else:
            context = None
            
    elif input_artifact == "trace":
        # Truncate trace for main output
        if judge_input.trace:
            truncated = self._truncate_trace(judge_input.trace)
            output_content = json.dumps(truncated, indent=2)
        else:
            output_content = "{}"
        context = judge_input.git_diff
        
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
```

#### 3. Add Trace Summary for Non-Trace Artifacts

When trace is NOT the primary artifact but might be useful context, create a compact summary instead of full (truncated) trace:

```python
def _create_trace_summary(self, trace: dict[str, Any]) -> dict[str, Any]:
    """Create compact summary of trace for context usage.
    
    Returns key metrics without full message content.
    """
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
            "first_message_role": trace.get("messages", [{}])[0].get("raw_message", {}).get("info", {}).get("role") if trace.get("messages") else None,
            "last_message_role": trace.get("messages", [{}])[-1].get("raw_message", {}).get("info", {}).get("role") if trace.get("messages") else None,
        }
    }
```

Then update `_prepare_eval_data()` for `final_message` case:
```python
elif input_artifact == "final_message":
    output_content = judge_input.final_message
    # Use compact summary for context, not full trace
    if judge_input.trace:
        summary = self._create_trace_summary(judge_input.trace)
        context = json.dumps(summary, indent=2)
    else:
        context = None
```

## Prerequisites

1. **Message Structure**: The `trace_session.json` must have a `messages` field that is a list of message objects with `raw_message` structure.

2. **Deterministic Output**: The truncation positions must be deterministic (same input → same output) for reproducible evaluations.

3. **Message Content Access**: Messages should have extractable content via `raw_message.info.role` and other standard fields.

## Downstream Services Affected

### 1. LLM Judge Models (gpt-4o-mini, etc.)
- **Impact**: Significantly reduced token usage (~70-90% reduction)
- **Benefit**: Faster evaluation, lower costs, no token limit errors
- **Trade-off**: Less context for judges, but key decision points preserved

### 2. Faithfulness Judge (DeepEval)
- **Impact**: Reduced context when using `git_diff` as input with trace as context
- **Mitigation**: Use `_create_trace_summary()` instead of truncated trace for context
- **Benefit**: Maintains efficiency while providing relevant metadata

### 3. Guidelines Judge
- **Impact**: Reduced context when evaluating trace-based outputs
- **Benefit**: Still sufficient to detect guideline adherence patterns
- **Note**: Guidelines judge typically evaluates git_diff, not trace, so minimal impact

### 4. Evaluation Results
- **Impact**: Slightly reduced judge accuracy due to truncated context
- **Mitigation**: First/last/middle sampling captures most important information
- **Monitoring**: Log truncation metadata to assess impact

## Testing Strategy

1. **Unit Test**: Test truncation logic with various message counts
   - 5 messages (no truncation needed)
   - 10 messages (minimal truncation)
   - 100 messages (heavy truncation)
   - 0 messages (edge case)

2. **Token Count Test**: Verify truncated output is under 100K tokens
```python
def test_truncation_reduces_tokens():
    large_trace = {"messages": [...100 messages...]}
    truncated = runner._truncate_trace(large_trace)
    json_str = json.dumps(truncated)
    estimated_tokens = len(json_str) / 4  # Rough estimate
    assert estimated_tokens < 100000
```

3. **Determinism Test**: Same input should produce same output
```python
def test_truncation_is_deterministic():
    trace = {...}
    result1 = runner._truncate_trace(trace)
    result2 = runner._truncate_trace(trace)
    assert result1 == result2
```

4. **Integration Test**: Full evaluation with truncated traces
   - Run evaluation on a real simulation
   - Verify no token limit errors
   - Check that scores are reasonable

## Configuration Options

Consider making truncation parameters configurable via environment variables:

```python
# settings.py
class Settings(BaseSettings):
    # ... existing settings ...
    eval_trace_keep_first: int = 3
    eval_trace_keep_last: int = 3
    eval_trace_middle_positions: list[float] = [0.25, 0.5, 0.75]
    eval_max_trace_tokens: int = 100000  # Safety threshold
```

## Success Criteria

- [ ] Token count for trace-based evaluation < 100K tokens (well under 128K limit)
- [ ] No more `BadRequestError: maximum context length` errors
- [ ] Evaluation completes successfully for runs with 50+ messages
- [ ] Truncation metadata logged to help debug any accuracy issues
- [ ] Deterministic output (reproducible evaluations)
- [ ] Backwards compatible (small traces work as before)

## Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Avg Tokens (100 msg run) | ~300K | ~30K | 90% reduction |
| Evaluation Time | Fails/timeout | ~5-10s | Functional |
| Cost per Evaluation | N/A (fails) | ~$0.02 | Now possible |

## Implementation Notes

- The truncation positions [0.25, 0.5, 0.75] are chosen to capture:
  - Early development (25%): Initial exploration phase
  - Middle work (50%): Main implementation phase
  - Final approach (75%): Convergence phase
- First 3 messages typically contain system prompt and initial context
- Last 3 messages contain the final output and any wrap-up
- Omission markers make it clear to judges that content was skipped

## Related Issues

- Phase 1: ToolCallEfficiency fix (ToolCallEfficiency needs actual trace, not truncated)
- Phase 3: Git diff truncation (similar token limit issues)
