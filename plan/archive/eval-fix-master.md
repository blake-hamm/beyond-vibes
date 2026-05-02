# Evaluation System Fix - Master Plan

## Executive Summary

This plan addresses critical failures in the evaluation system when running judges on simulation traces. The issues include token limit errors (153K tokens requested vs 128K limit) and incorrect ToolCallEfficiency usage.

**Philosophy**: Fail fast with loud, detailed errors. No graceful degradation - if something breaks, we want to know immediately with full context.

**Total Implementation**: 3 phases
**Estimated Effort**: 1-2 days
**Risk Level**: Low-Medium

## Quick Reference

| Phase | Issue | Solution | File(s) Modified |
|-------|-------|----------|------------------|
| [Phase 1](eval-fix-phase-1.md) | ToolCallEfficiency fails with NoneType error | Use actual MLflow Trace object via `trace=` parameter | `runner.py` |
| [Phase 2](eval-fix-phase-2.md) | Token limit exceeded (293K tokens) | Truncate traces: first 3 + last 3 + middle samples | `runner.py` |
| [Phase 3](eval-fix-phase-3.md) | Large git diffs cause token issues | Truncate diffs to 30KB with header preservation | `runner.py` |

## Problem Analysis

### Current Failure Mode

When running:
```bash
uv run python -m beyond_vibes.cli evaluate --run-id d10fdff6b15240ea8f88fe2a3f390237
```

The following errors occur:

1. **ToolCallEfficiency**: `'NoneType' object has no attribute 'info'`
   - Root cause: ToolCallEfficiency requires MLflow Trace object, not standard evaluate format

2. **Token Limit**: `maximum context length is 128000 tokens. However, you requested about 153919 tokens`
   - Root cause: Full trace_session.json serialized (293,922 tokens)

3. **AWS S3 Warnings**: Non-blocking but noisy
   - These are warnings about trace upload, not blocking errors

## Error Handling Philosophy: FAIL FAST

**We do NOT implement graceful error handling.**

Instead:
- Let exceptions propagate with full stack traces
- No try/except blocks that swallow errors
- No fallback mechanisms that hide issues
- If a judge fails, the entire evaluation fails loudly
- Full error details in logs for debugging

**Rationale**: 
- Errors indicate real problems that need fixing
- Silent failures lead to incorrect conclusions
- Full stack traces help identify root causes
- Better to catch issues early than have bad data

## Implementation Order

### Recommended Sequence

1. **Phase 2** (Trace Truncation)
   - Fixes the most critical token limit issue
   - Enables evaluations to complete
   - Benefits all trace-based judges
   - Independent of other changes

2. **Phase 3** (Git Diff Truncation)
   - Secondary token reduction
   - Important for Guidelines judge which evaluates git_diff
   - Independent of other changes

3. **Phase 1** (ToolCallEfficiency Fix)
   - Most complex change
   - Requires actual MLflow traces to test
   - Depends on Phases 2-3 to prevent token errors

### Parallel Development

Phases 2 and 3 can be developed in parallel since they're independent:
- Phase 2: Trace truncation logic
- Phase 3: Git diff truncation logic

Phase 1 should be done after 2 and 3 are working.

## Key Design Decisions

### 1. ToolCallEfficiency Approach

**Decision**: Use actual MLflow Trace object (Option B)

**Rationale**:
- ToolCallEfficiency is designed to analyze actual MLflow traces
- Using pre-computed metrics would defeat the purpose of the judge
- Provides more nuanced analysis (e.g., similar tool call detection)

**Implementation**:
- Fetch trace using `mlflow.search_traces()`
- Call `judge(trace=trace)` directly
- If trace fetching fails, let the exception propagate (fail fast)

### 2. Trace Truncation Strategy

**Decision**: First 3 + Last 3 + Deterministic Middle (25%, 50%, 75%)

**Rationale**:
- First 3: Setup, system prompt, initial context
- Last 3: Final output, conclusion, wrap-up
- Middle samples: Capture key decision points
- Deterministic: Reproducible evaluations

**Trade-offs**:
- Loses some context from omitted messages
- May miss important mid-conversation changes
- Compromise between completeness and token limits

### 3. No Error Handling

**Decision**: Remove all try/except blocks that handle evaluation errors

**What this means**:
- If `mlflow.genai.evaluate()` returns None, we crash with AttributeError
- If token limit exceeded, we crash with BadRequestError
- If trace fetching fails, we crash with MlflowException
- All exceptions propagate to the CLI with full stack traces

**Benefits**:
- Immediate visibility into problems
- No silent failures
- Forces proper fixes rather than workarounds
- Stack traces show exact failure points

## Dependencies and Prerequisites

### System Requirements

1. **MLflow 3.10.0+** (already required)
   - Tracing support for ToolCallEfficiency
   - Stable genai.evaluate() API

2. **MLflow Tracing Enabled**
   - Traces must be stored and searchable
   - Run ID association in trace metadata
   - Verify: `mlflow.search_traces()` returns results

3. **Artifact Storage**
   - `trace_session.json` with required fields
   - `git_diff.patch` for Guidelines judge
   - Both accessible via `mlflow.artifacts`

### Code Dependencies

```python
# Required imports (already present)
import mlflow
from mlflow.genai.scorers import Guidelines, ToolCallEfficiency
```

### Data Requirements

**trace_session.json must contain**:
```json
{
  "messages": [...],
  "total_messages": 100,
  "tool_total_calls": 25,
  "tool_loop_detected": false,
  "tool_error_rate": 0.05,
  "total_tokens": 50000,
  "total_cost": 0.50,
  "completion_status": "completed"
}
```

## Testing Strategy

### Unit Tests

For each phase, create tests in `tests/unit/evaluations/`:

```python
# Phase 1: ToolCallEfficiency
test_tool_call_efficiency_with_trace()

# Phase 2: Trace Truncation
test_truncate_small_trace()
test_truncate_large_trace()
test_truncation_determinism()

# Phase 3: Git Diff Truncation
test_truncate_large_diff()
test_diff_header_preservation()
```

### Integration Tests

1. **End-to-End Evaluation**
   ```bash
   # Run evaluation on a known good run
   uv run python -m beyond_vibes.cli evaluate --run-id <test-run-id>
   
   # Verify:
   # - Evaluation completes successfully
   # - All judges produce scores
   # - Scores logged to MLflow
   ```

2. **Large Trace Test**
   - Use a run with 50+ messages
   - Verify token count < 100K
   - Verify evaluation completes without token errors

3. **Failure Test**
   - Test with missing trace data
   - Verify we get a loud, clear error with full stack trace
   - No silent failures or swallowed exceptions

### Manual Testing Checklist

- [ ] Run evaluation on existing run with all 3 judges
- [ ] Verify ToolCallEfficiency produces non-zero scores
- [ ] Verify no token limit errors in logs
- [ ] Check MLflow UI for logged metrics
- [ ] Verify git diff truncation visible in context
- [ ] Test error case: verify full stack trace appears

## Rollback Plan

If issues arise post-deployment:

1. **Immediate**: Revert to previous runner.py version
   ```bash
   git checkout HEAD~1 -- src/beyond_vibes/evaluations/runner.py
   ```

2. **Debug**: Enable full Python tracebacks
   ```bash
   export PYTHONVERBOSE=1
   uv run python -m beyond_vibes.cli evaluate --run-id <id>
   ```

## Cost Impact

### Before Fixes
- Token limit errors: **FAILS** (evaluation doesn't complete)
- 100KB git diff: **~$0.05** per judge (if it worked)
- Large trace (300K tokens): **FAILS**

### After Fixes
- 30KB git diff: **~$0.015** per judge
- Truncated trace (30K tokens): **~$0.01** per judge
- ToolCallEfficiency trace fetch: **Minimal** (1 API call)

**Estimated Savings**: 60-70% cost reduction for large runs

## Future Enhancements (Optional)

### Advanced Truncation
- Smart content selection based on message importance
- Dynamic truncation based on judge requirements
- Caching of truncated traces

### Judge-Specific Optimization
- Custom truncation per judge type
- Guidelines judge: Focus on file changes
- Faithfulness judge: Focus on task completion
- ToolCallEfficiency: Full trace (no truncation)

## Success Criteria

### Must Have
- [ ] All evaluations complete successfully (no token errors)
- [ ] ToolCallEfficiency produces valid scores using MLflow traces
- [ ] Token usage reduced by 50%+
- [ ] Failures produce loud, detailed error messages

### Should Have
- [ ] Evaluation latency < 30 seconds per judge
- [ ] Test coverage for truncation logic
- [ ] Documentation updated

### Nice to Have
- [ ] Configurable truncation parameters
- [ ] Real-time truncation preview

## Related Documentation

- [Original Judges Implementation Plan](judges.md)
- [Phase 2 Data Models](judge-phase-2.md)
- [Phase 6 Runner Module](judge-phase-6.md)
- MLflow GenAI Docs: https://mlflow.org/docs/latest/genai-metrics.html

## Debugging Failed Evaluations

When an evaluation fails, you'll see:

1. **Full Python traceback** showing exactly where it failed
2. **Exception type and message** with all details
3. **Local variables** (if running with `PYTHONVERBOSE=1`)
4. **MLflow trace information** (if trace fetching failed)

Example:
```python
Traceback (most recent call last):
  File ".../runner.py", line 188, in _evaluate_tool_call_efficiency
    traces_df = mlflow.search_traces(...)
  File ".../mlflow/tracking/client.py", line 1234, in search_traces
    raise MlflowException("No traces found for run_id=...")
mlflow.exceptions.MlflowException: No traces found for run_id=d10fdff6b15240ea8f88fe2a3f390237
```

This immediately tells you:
- Which method failed (`_evaluate_tool_call_efficiency`)
- Which line failed (line 188)
- What the error is (no traces found)
- Which run_id was being evaluated

## Approval and Sign-off

Before implementation:
- [ ] Review all 3 phase documents
- [ ] Confirm MLflow tracing is enabled and working
- [ ] Identify test runs for validation
- [ ] Acknowledge "fail fast" philosophy

After implementation:
- [ ] Run full test suite
- [ ] Verify with real simulation runs
- [ ] Test that errors produce loud failures
- [ ] Update CHANGELOG
- [ ] Notify team of changes
