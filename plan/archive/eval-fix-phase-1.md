# Evaluation Fix Phase 1: ToolCallEfficiency Judge Fix

## Overview

**Problem**: ToolCallEfficiency judge is failing with `'NoneType' object has no attribute 'info'` because it's being invoked via `mlflow.genai.evaluate()` with the standard `inputs/outputs/context` format, but ToolCallEfficiency is a **session-level scorer** that requires an actual MLflow Trace object via the `trace=` parameter.

**Root Cause**: From MLflow 3.10.0 documentation, ToolCallEfficiency evaluates "the agent's trajectory for redundancy in tool usage" and expects:
```python
trace = mlflow.get_trace("<trace-id>")
feedback = ToolCallEfficiency()(trace=trace)
```

The current implementation in `runner.py` incorrectly passes it to `mlflow.genai.evaluate()` which uses the wrong data format.

## Changes Required

### File: `src/beyond_vibes/evaluations/runner.py`

#### 1. Add Special Handling in `_evaluate_with_judge()`

Modify the method to detect ToolCallEfficiency judges and handle them separately:

```python
def _evaluate_with_judge(
    self,
    judge: Any,
    input_artifact: str,
    judge_input: JudgeInput,
) -> dict[str, Any]:
    """Evaluate a single judge with appropriate input."""
    
    # Special handling for ToolCallEfficiency - it needs an actual MLflow Trace
    if self._is_tool_call_efficiency_judge(judge):
        return self._evaluate_tool_call_efficiency(judge, judge_input)
    
    # Standard evaluation for other judges
    eval_data = self._prepare_eval_data(judge_input, input_artifact)
    # ... rest of existing code
```

#### 2. Add Helper Method `_is_tool_call_efficiency_judge()`

```python
def _is_tool_call_efficiency_judge(self, judge: Any) -> bool:
    """Check if judge is ToolCallEfficiency type."""
    return (
        hasattr(judge, '__class__') 
        and judge.__class__.__name__ == 'ToolCallEfficiency'
    )
```

#### 3. Add ToolCallEfficiency Evaluation Method (Fail Fast Version)

**NO ERROR HANDLING** - If something fails, let it propagate with full stack trace:

```python
def _evaluate_tool_call_efficiency(
    self,
    judge: Any,
    judge_input: JudgeInput,
) -> dict[str, Any]:
    """Evaluate ToolCallEfficiency using actual MLflow Trace.
    
    ToolCallEfficiency is a session-level scorer that requires the actual
    MLflow Trace object, not the standard inputs/outputs format.
    
    NOTE: This method does NOT catch exceptions. If trace fetching fails,
    the exception propagates with full details for debugging.
    """
    # Search for traces associated with this run
    # If this fails, we get a loud error with full stack trace
    traces_df = mlflow.search_traces(
        filter_string=f"attributes.run_id = '{judge_input.run_id}'"
    )
    
    # If no traces found, raise a clear error immediately
    if traces_df is None or len(traces_df) == 0:
        raise ValueError(
            f"No traces found for run_id={judge_input.run_id}. "
            f"ToolCallEfficiency requires MLflow traces to be enabled and stored. "
            f"Ensure simulations are logging traces via mlflow.start_span()."
        )
    
    # Get the first trace (should be the main simulation trace)
    trace = traces_df.iloc[0]['trace']
    
    # Invoke judge directly with trace parameter
    # If this fails, we get the full MLflow/LiteLLM error with details
    feedback = judge(trace=trace)
    
    # Extract score from Feedback object
    if hasattr(feedback, 'value'):
        # Binary or categorical feedback
        score = 1.0 if feedback.value in ['yes', 'efficient', True] else 0.0
    elif hasattr(feedback, 'score'):
        # Numeric score
        score = float(feedback.score)
    else:
        raise ValueError(
            f"Unexpected feedback format from ToolCallEfficiency: {type(feedback)}. "
            f"Expected object with 'value' or 'score' attribute."
        )
    
    rationale = getattr(feedback, 'rationale', None)
    
    return {"score": score, "rationale": rationale}
```

## Prerequisites

1. **MLflow Tracing Must Be Enabled**: ToolCallEfficiency requires traces to be properly stored in MLflow.
   - Verify: `mlflow.search_traces()` should return traces for completed runs
   - Traces must be associated with run IDs

2. **No Fallback**: If traces are not available, the evaluation FAILS LOUDLY with a clear error message explaining:
   - What went wrong (no traces found)
   - Why it matters (ToolCallEfficiency requires traces)
   - How to fix it (enable MLflow tracing in simulations)

## Downstream Services Affected

### 1. MLflow Tracking Server
- **Impact**: Additional `mlflow.search_traces()` calls per evaluation
- **Load**: 1 query per ToolCallEfficiency evaluation
- **Mitigation**: Ensure proper indexing on trace run_id attributes

### 2. Judge Model (LLM)
- **Impact**: ToolCallEfficiency invokes LLM to analyze tool usage patterns
- **Cost**: One LLM call per evaluation (when traces exist)
- **Latency**: Additional 1-2 seconds

### 3. Simulation Tracing
- **Requirement**: Traces must be properly associated with run IDs
- **Verification**: Check that `mlflow.start_span()` sets run metadata

## Testing Strategy

1. **Integration Test**: Run evaluation on a real simulation with traces
   ```bash
   uv run python -m beyond_vibes.cli evaluate --run-id <trace-enabled-run>
   ```

2. **Failure Test**: Run evaluation on run WITHOUT traces
   ```bash
   # Should fail with clear error:
   # ValueError: No traces found for run_id=xxx. ToolCallEfficiency requires...
   ```

3. **Edge Cases**:
   - Multiple traces (should use first one)
   - Trace without tool calls (should score appropriately)
   - Corrupted trace data (should fail with MLflow error)

## Success Criteria

- [ ] ToolCallEfficiency no longer returns `'NoneType' object has no attribute 'info'` error
- [ ] ToolCallEfficiency produces meaningful scores using actual MLflow traces
- [ ] Clear error message when traces are missing
- [ ] Full stack trace on any failure for debugging
- [ ] Scores logged correctly to MLflow run metrics

## Error Messages

When things fail, you'll see clear errors:

**No traces found:**
```
ValueError: No traces found for run_id=d10fdff6b15240ea8f88fe2a3f390237. 
ToolCallEfficiency requires MLflow traces to be enabled and stored. 
Ensure simulations are logging traces via mlflow.start_span().
```

**Trace fetch fails:**
```
mlflow.exceptions.MlflowException: Permission denied accessing traces
[Full stack trace showing exact call site]
```

**Unexpected feedback format:**
```
ValueError: Unexpected feedback format from ToolCallEfficiency: <class 'dict'>. 
Expected object with 'value' or 'score' attribute.
[Full stack trace]
```

## Implementation Notes

- No fallback mechanism - failures are loud and clear
- Users must ensure MLflow tracing is properly configured
- Clear error messages guide users to the solution
- Full stack traces help identify integration issues

## Related Issues

- Token limit errors (153K tokens) - fixed by Phase 2 trace truncation
- AWS S3 warnings - not blocking, can be ignored
