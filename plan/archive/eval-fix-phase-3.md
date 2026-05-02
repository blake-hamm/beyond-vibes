# Evaluation Fix Phase 3: Git Diff Truncation

## Overview

**Problem**: Git diffs for substantial code changes can be extremely large (50K+ characters), contributing to token limit errors when included as output or context in judge evaluation.

**Root Cause**: In `runner.py:_prepare_eval_data()`, the full git_diff is passed directly:
```python
output_content = judge_input.git_diff or ""  # Can be massive!
context = judge_input.git_diff  # Same issue
```

**Solution**: Implement intelligent git diff truncation that preserves the most important changes while staying within token budgets.

## Changes Required

### File: `src/beyond_vibes/evaluations/runner.py`

#### 1. Add Git Diff Truncation Helper

```python
def _truncate_git_diff(
    self,
    diff: str,
    max_chars: int = 30000,
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
    if not diff or len(diff) <= max_chars:
        return diff
    
    lines = diff.split('\n')
    total_lines = len(lines)
    
    # Estimate average line length to calculate how many lines we can keep
    avg_line_len = len(diff) / total_lines
    target_lines = int(max_chars / avg_line_len)
    
    if keep_header:
        # Extract header (file stats, summary lines)
        header_lines = []
        content_start = 0
        
        for i, line in enumerate(lines):
            # Header typically ends at first "diff --git" or "@@"
            if line.startswith('diff --git') or line.startswith('@@'):
                content_start = i
                break
            header_lines.append(line)
        
        # Keep header + beginning of content + end of content
        header_len = len('\n'.join(header_lines))
        remaining_chars = max_chars - header_len - 500  # Buffer for markers
        
        if remaining_chars > 1000:
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
            result_lines.append(f"# ... {len(diff) - current_chars - sum(len(l) for l in lines[-int(end_chars/avg_line_len):])} characters omitted ...")
            result_lines.append("")
            
            # Add end content
            end_start_idx = max(begin_end_idx + 1, total_lines - int(end_chars / avg_line_len))
            for i in range(end_start_idx, total_lines):
                result_lines.append(lines[i])
            
            return '\n'.join(result_lines)
    
    # Simple truncation: keep beginning and end
    begin_chars = int(max_chars * 0.6)
    end_chars = int(max_chars * 0.4)
    
    truncated = (
        diff[:begin_chars] +
        f"\n\n# ... {len(diff) - begin_chars - end_chars} characters omitted ...\n\n" +
        diff[-end_chars:]
    )
    
    return truncated
```

#### 2. Add Diff Statistics Helper

```python
def _extract_diff_stats(self, diff: str) -> dict[str, Any]:
    """Extract key statistics from git diff for summary.
    
    Returns:
        Dictionary with file count, line changes, etc.
    """
    lines = diff.split('\n')
    
    stats = {
        "total_lines": len(lines),
        "files_changed": 0,
        "insertions": 0,
        "deletions": 0,
        "file_list": [],
    }
    
    for line in lines:
        if line.startswith('diff --git'):
            stats["files_changed"] += 1
            # Extract filename from "diff --git a/path b/path"
            parts = line.split()
            if len(parts) >= 4:
                filename = parts[2][2:]  # Remove 'a/' prefix
                stats["file_list"].append(filename)
        elif line.startswith('+') and not line.startswith('+++'):
            stats["insertions"] += 1
        elif line.startswith('-') and not line.startswith('---'):
            stats["deletions"] += 1
    
    return stats
```

#### 3. Modify `_prepare_eval_data()` to Apply Truncation

```python
def _prepare_eval_data(
    self,
    judge_input: JudgeInput,
    input_artifact: str,
) -> dict[str, Any]:
    """Convert JudgeInput to format expected by mlflow.evaluate."""
    
    if input_artifact == "git_diff":
        # Truncate git diff to prevent token limit errors
        if judge_input.git_diff:
            output_content = self._truncate_git_diff(judge_input.git_diff)
            # Log truncation if it occurred
            if len(output_content) < len(judge_input.git_diff):
                original_kb = len(judge_input.git_diff) / 1024
                truncated_kb = len(output_content) / 1024
                logger.debug(
                    f"Truncated git diff: {original_kb:.1f}KB → {truncated_kb:.1f}KB"
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
        # Truncate trace for main output (from Phase 2)
        if judge_input.trace:
            truncated = self._truncate_trace(judge_input.trace)
            output_content = json.dumps(truncated, indent=2)
        else:
            output_content = "{}"
        # Use truncated git_diff as context if available
        if judge_input.git_diff:
            context = self._truncate_git_diff(judge_input.git_diff, max_chars=15000)
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
```

## Prerequisites

1. **Git Diff Format**: Standard unified diff format with:
   - `diff --git` headers
   - `@@` hunk headers
   - `+` and `-` line prefixes
   - `---` and `+++` file markers

2. **Diff Availability**: The simulation must generate and log `git_diff.patch` artifact

3. **Reasonable Size Expectations**: Some migrations may legitimately have massive diffs (>100KB), which is why we need truncation

## Downstream Services Affected

### 1. Guidelines Judge (Primary Consumer)
- **Impact**: Guidelines judge evaluates git_diff against criteria
- **Trade-off**: Truncation may miss violations in omitted sections
- **Mitigation**: 
  - Keep header with file list so judges know what files changed
  - Keep beginning (often has important config changes)
  - Keep end (often has final adjustments)
  - Judges can request re-evaluation with full diff if needed

### 2. Faithfulness Judge (DeepEval)
- **Impact**: Uses git_diff as context (when evaluating final_message)
- **Benefit**: Compact summary is sufficient for relevance check
- **Alternative**: Use `_extract_diff_stats()` instead of full diff for context

### 3. LLM Judge Models
- **Impact**: Reduced token usage from large diffs
- **Benefit**: Faster evaluation, lower costs
- **Example Cost Savings**:
  - 100KB diff ≈ 25K tokens ≈ $0.05 (gpt-4o-mini input)
  - 30KB diff ≈ 7.5K tokens ≈ $0.015
  - 70% cost reduction for large diffs

### 4. Evaluation Storage
- **Impact**: Truncated diffs stored in evaluation results
- **Consideration**: May lose some fidelity in historical data
- **Mitigation**: Original diff still stored in run artifacts

## Testing Strategy

1. **Unit Test: Truncation Logic**
```python
def test_git_diff_truncation():
    # Create large diff (100 lines)
    large_diff = "\n".join([f"+line {i}" for i in range(100)])
    
    truncated = runner._truncate_git_diff(large_diff, max_chars=500)
    
    assert len(truncated) <= 1000  # Reasonable limit
    assert "omitted" in truncated.lower() or "truncated" in truncated.lower()
    assert truncated.startswith("+line 0")  # Beginning preserved
    assert "line 99" in truncated  # End preserved
```

2. **Test: Header Preservation**
```python
def test_git_diff_header_preserved():
    diff_with_header = """diff --git a/file.py b/file.py
index abc..def 100644
--- a/file.py
+++ b/file.py
@@ -1,5 +1,5 @@
-context
+changed"""
    
    truncated = runner._truncate_git_diff(diff_with_header, max_chars=100)
    
    assert "diff --git" in truncated
    assert "file.py" in truncated
```

3. **Integration Test: Large Migration**
   - Run evaluation on a real poetry-to-uv migration
   - Verify evaluation completes without token errors
   - Check that Guidelines scores are reasonable

4. **Edge Cases**
   - Empty diff (should return empty string)
   - Single file diff (should not truncate)
   - Binary diff (should handle gracefully)
   - Diff with only deletions

## Configuration Options

Add to settings for customization:

```python
# settings.py
class Settings(BaseSettings):
    # ... existing settings ...
    eval_git_diff_max_chars: int = 30000  # ~7.5K tokens
    eval_git_diff_keep_header: bool = True
    eval_context_diff_max_chars: int = 15000  # Smaller for context usage
```

## Heuristic Guidelines for Truncation

When deciding what to keep:

1. **Priority 1: Header Information**
   - File list (what changed)
   - Statistics (how much changed)
   - This helps judges understand scope

2. **Priority 2: Beginning of Diff**
   - Configuration files (pyproject.toml, package.json)
   - Import statements
   - Class/function definitions
   - Often contains the most important changes

3. **Priority 3: End of Diff**
   - Final adjustments
   - Test additions
   - Documentation updates

4. **Priority 4: Middle (Sampled)**
   - Implementation details
   - Less critical for high-level evaluation

## Success Criteria

- [ ] Git diffs > 30KB are truncated to < 30KB
- [ ] File list and statistics preserved in header
- [ ] Truncation markers clearly indicate omitted content
- [ ] No token limit errors from large git diffs
- [ ] Guidelines judge can still evaluate effectively
- [ ] Truncation is logged for debugging
- [ ] Backwards compatible (small diffs unchanged)

## Performance Impact

| Diff Size | Before | After | Improvement |
|-----------|--------|-------|-------------|
| 10KB | 2.5K tokens | 2.5K tokens | None (no truncation) |
| 50KB | 12.5K tokens | 7.5K tokens | 40% reduction |
| 100KB | 25K tokens | 7.5K tokens | 70% reduction |
| 500KB | 125K tokens → FAIL | 7.5K tokens | From fail to success |

## Integration with Other Phases

- **Phase 1 (ToolCallEfficiency)**: ToolCallEfficiency doesn't use git_diff, so no impact
- **Phase 2 (Trace Truncation)**: Both truncations work together to keep total context under limit

## Implementation Notes

- Use `#` comments for truncation markers (git diff comment style)
- Preserve unified diff format structure
- Consider configurable truncation strategy per judge type
- Log truncation events at DEBUG level to avoid noise
- Original full diff remains available in run artifacts for manual inspection
