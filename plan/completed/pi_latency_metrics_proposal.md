# Proposal: Deriving Latency Metrics from pi.dev JSONL Stream

## What the Script Revealed

Ran `scripts/analyze_pi_timing.py` against fixtures and a **live** `pi --mode json` capture.

### Key Finding: JSON Timestamps Are Useless for Streaming Latency

The `timestamp` field inside `message` objects (e.g. `1777502812949`) is identical for `message_start` and `message_end` of the same assistant response. In the live capture a 2-second generation had the same start/end timestamp. These are API-level completion timestamps, not streaming event timestamps.

**Conclusion:** We cannot derive TTFT, TPS, or generation time from JSON fields alone.

### What Does Work: Wall-Clock Timing in the Reader

`PiDevClient` reads stdout line-by-line as the JSONL stream arrives. The wall-clock interval between lines **is** the real latency.

Live capture data (prompt: "Write a 3-sentence paragraph about Python programming."):

| Boundary | Wall-Clock Time | Derived Metric |
|----------|-----------------|----------------|
| Subprocess spawn | t=0 | Baseline |
| `message_end(user)` | t=1.775s | Prompt submitted |
| `message_start(assistant)` | t=5.541s | **Prompt processing = 3,766 ms** |
| First `message_update` | t=5.541s | **TTFT ≈ 3,766 ms** |
| `message_end(assistant)` | t=7.622s | **Generation time = 2,081 ms** |
| Output tokens | 135 | **TPS = 64.9 tok/s** |

`message_start(assistant)` and first `message_update` arrive within 0.1 ms of each other. `message_start` is effectively the first-token signal.

## Reference: llama.cpp Benchmark Metrics

**llama-bench** (the canonical local LLM benchmark) measures:

| Metric | Name | Formula |
|--------|------|---------|
| Prompt processing throughput | `pp t/s` | `n_prompt / prompt_processing_time` |
| Generation throughput | `tg t/s` | `n_gen / generation_time` |
| Raw time | `avg_ns` | Total nanoseconds |
| Variance | `stddev_ns`, `stddev_ts` | Standard deviation across repetitions |

We adapt `pp t/s` and `tg t/s` to pi.dev's streaming JSONL model.

## Council Review Outcome

Spawned **architect** and **product** reviewers. Converged on:

- **Ship 6 core fields only.** Defer everything else.
- **Do not bloat `TurnData`.** Capture raw wall-clock timestamps in a nested `TurnTimestamps` dataclass. Compute derived metrics via a pure helper function before yielding.
- **Keep `_read_turns()` small.** No metric formulas inside the event loop.

---

## Proposed Metrics (v1 — Shipping)

### Per-Turn Metrics

| Metric | Derivation | Source Field | Benchmark Parallel |
|--------|-----------|-------------|-------------------|
| **Prompt Processing Time (ms)** | `assistant_msg_start.wall - user_msg_end.wall` | `TurnData.prompt_processing_ms` | Raw latency |
| **TTFT (ms)** | Same as prompt processing (`message_start` is first-token proxy) | `TurnData.ttft_ms` | — |
| **Generation Time (ms)** | `assistant_msg_end.wall - assistant_msg_start.wall` | `TurnData.generation_time_ms` | `avg_ns` (gen phase) |
| **End-to-End Turn Time (ms)** | `assistant_msg_end.wall - user_msg_end.wall` | `TurnData.e2e_turn_ms` | Total request latency |
| **Prompt Throughput (t/s)** | `usage.input / (prompt_processing_ms / 1000)` | `TurnData.prompt_tps` | `pp t/s` |
| **Generation Throughput (t/s)** | `usage.output / (generation_time_ms / 1000)` | `TurnData.generation_tps` | `tg t/s` |

### Session-Level Metrics

| Metric | Source |
|--------|--------|
| `avg_ttft_ms` | Mean of turn `ttft_ms` |
| `avg_prompt_tps` | Mean of turn `prompt_tps` |
| `avg_generation_tps` | Mean of turn `generation_tps` |
| `avg_tps` | Alias for `avg_generation_tps` |
| `total_generation_time_ms` | Sum of all turn `generation_time_ms` |
| `total_prompt_processing_ms` | Sum of all turn `prompt_processing_ms` |
| `total_e2e_time_seconds` | `session.completed_at - session.started_at` (already works) |

### Deferred to v2

These were cut by council consensus. Revisit only after user demand or reasoning-model support:

- `tpot_ms` — inverse of `generation_tps`; redundant
- Inter-token latency distribution (`avg_`, `min_`, `max_`, `stddev_`) — diagnostic noise for v1
- `thinking_time_ms` / `thinking_to_text_ratio` — only for reasoning models; not current provider
- `tool_execution_ms` — measures local tool speed, not model performance
- `framework_overhead_ms` — residual metric; semantically shaky
- `cache_hit_ratio` — `usage.cacheRead` existence unconfirmed across providers

---

## Implementation Plan

### 1. Add `TurnTimestamps` Dataclass

Capture only raw wall-clock timestamps inside `_read_turns()`. No metric formulas in the event loop.

```python
@dataclass
class TurnTimestamps:
    """Raw wall-clock timestamps captured during JSONL streaming."""
    user_message_end: float | None = None
    assistant_message_start: float | None = None
    assistant_message_end: float | None = None
    first_update: float | None = None
```

### 2. Extend `TurnData`

Attach `timestamps` as a single nested field. Keep 6 derived metrics as top-level fields.

```python
@dataclass
class TurnData:
    turn_index: int
    content: list[dict] = field(default_factory=list)
    usage: dict | None = None
    stop_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    raw_message: dict | None = None
    timestamp: int | None = None

    # Raw timestamps (captured during streaming)
    timestamps: TurnTimestamps | None = None

    # Derived latency metrics (computed before yield)
    prompt_processing_ms: float | None = None
    ttft_ms: float | None = None
    generation_time_ms: float | None = None
    e2e_turn_ms: float | None = None
    prompt_tps: float | None = None
    generation_tps: float | None = None
```

### 3. Pure Helper for Metric Computation

```python
def compute_latency_metrics(turn: TurnData) -> None:
    """Compute wall-clock derived metrics from TurnData.timestamps.

    Mutates the TurnData in place. Called before yielding a turn.
    """
    ts = turn.timestamps
    if ts is None:
        return

    usage = turn.usage or {}
    input_tokens = usage.get("input", 0)
    output_tokens = usage.get("output", 0)

    if ts.assistant_message_start and ts.user_message_end:
        turn.prompt_processing_ms = (
            ts.assistant_message_start - ts.user_message_end
        ) * 1000
        turn.ttft_ms = turn.prompt_processing_ms

    if ts.assistant_message_end and ts.assistant_message_start:
        turn.generation_time_ms = (
            ts.assistant_message_end - ts.assistant_message_start
        ) * 1000

    if ts.assistant_message_end and ts.user_message_end:
        turn.e2e_turn_ms = (
            ts.assistant_message_end - ts.user_message_end
        ) * 1000

    if turn.prompt_processing_ms and input_tokens > 0:
        turn.prompt_tps = input_tokens / (turn.prompt_processing_ms / 1000)

    if turn.generation_time_ms and output_tokens > 0:
        turn.generation_tps = output_tokens / (turn.generation_time_ms / 1000)
```

### 4. Wire Timing into `_read_turns()`

Record `time.perf_counter()` per line. Set timestamps on `current_turn` / `pending_turn` at key events:

- `message_end(role=user)` → `timestamps.user_message_end = t_recv`
- `message_start(role=assistant)` → `timestamps.assistant_message_start = t_recv`
- First `message_update` after `message_start` → `timestamps.first_update = t_recv`
- `message_end(role=assistant)` → `timestamps.assistant_message_end = t_recv`; call `compute_latency_metrics(turn)`; yield

No per-update lists. No O(n) state growth.

### 5. Wire into `MlflowTracer.log_turn()`

Populate `MessagePerformanceMetrics` from `TurnData`:

```python
perf = MessagePerformanceMetrics(
    ttft_ms=turn.ttft_ms,
    generation_time_ms=turn.generation_time_ms,
    tps=turn.generation_tps,
    output_tokens=usage.get("output", 0),
    has_tool_calls=bool(turn.tool_calls),
)
self.session.message_metrics.append(perf)
```

Set span attributes:
- `perf.ttft_ms`
- `perf.prompt_tps`
- `perf.generation_tps`
- `perf.generation_time_ms`

### 6. Verify Aggregation in `_flush()`

Extend existing aggregation loop to compute session-level means for the 6 core fields.

### 7. MLflow Logging

Log session-level metrics:

```python
mlflow.log_metric("avg_ttft_ms", self.session.avg_ttft_ms)
mlflow.log_metric("avg_prompt_tps", self.session.avg_prompt_tps)
mlflow.log_metric("avg_generation_tps", self.session.avg_generation_tps)
mlflow.log_metric("avg_tps", self.session.avg_generation_tps)
mlflow.log_metric("total_generation_time_ms", self.session.total_generation_time_ms)
mlflow.log_metric("total_prompt_processing_ms", self.session.total_prompt_processing_ms)
```

---

## Edge Cases

| Case | Handling |
|------|----------|
| `output_tokens == 0` | `generation_tps = None` |
| `input_tokens == 0` | `prompt_tps = None` |
| Tool call turns | `has_tool_calls = True`; metrics cover the full turn wall-clock. No special splitting. |
| `turn_end` only present with tools | Timing is event-driven, not `turn_end`-dependent. |
| Premature EOF | Raise `PiDevError` as before; partially-built turn discarded. |

## Known Limitations

| Missing | Why | Workaround |
|---------|-----|------------|
| KV cache size / pressure | pi doesn't expose this | Infer from `prompt_tps` degradation across turns |
| GPU/CPU utilization | pi is a client, not the inference engine | Run `nvidia-smi` / `nvtop` alongside if local |
| Batch size / concurrency | Single-request architecture | N/A — this is per-request benchmarking |
| Tokenizer latency | pi includes tokenization in prompt processing | Included in `prompt_processing_ms` |
| Sampling overhead | pi includes sampling in generation | Included in `generation_time_ms` |
| `message_start` as TTFT proxy | Unvalidated across all providers | Documented assumption; refine if divergence observed |

## Files to Touch

1. `src/beyond_vibes/simulations/pi_dev.py` — `TurnTimestamps`, `compute_latency_metrics()`, timing in `_read_turns()`
2. `src/beyond_vibes/simulations/mlflow.py` — `log_turn()` metric wiring, `_flush()` aggregation
3. `tests/test_pi_dev.py` — test `timestamps` and derived fields on `TurnData`
4. `tests/test_mlflow_tracer.py` — verify `log_turn()` sets perf span attributes and populates `message_metrics`

## Next Step

Approve this proposal and I will implement the changes.
