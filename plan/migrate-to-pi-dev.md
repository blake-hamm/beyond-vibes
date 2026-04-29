# Migration: OpenCode Server → pi.dev CLI

## How to Use This Document

This is a **living document**. It exists to guide direction, not to dictate
implementation. Each phase is a hypothesis. As we build, we learn, and we
update this plan accordingly.

**Principles:**
- Taste and elegance over exhaustive specification.
- Simple implementation over complex abstraction.
- Speed of development over fidelity to this document.
- When in doubt, build the smallest thing that could work, run it, and inspect
the result.

**Process:**
1. Read the current phase.
2. Implement the smallest viable version.
3. Run it. Look at the output (MLflow traces, logs, errors).
4. Update this document to reflect what you learned.
5. Remove implementation details from completed phases. Summarize only what
   subsequent phases need to know.

---

## Context

OpenCode did not have the `run` feature when this project started. The current
architecture uses `opencode serve` (HTTP server on :4096) and polls
`GET /session/{id}/message` every 5 seconds. This server approach has proven
unreliable.

**pi.dev** is preferred because:
- Smaller footprint (standalone Node.js CLI, no daemon)
- Purpose-built for headless/CI execution
- Native JSONL event stream (`--mode json`) with no polling required
- One-shot invocation: spawn, stream stdout, reap
- Aligns with the project's day-to-day workflow (pi is the primary coding agent)

## Core Decision

We will **not** reconstruct OpenCode-compatible message dicts. `MlflowTracer` is
the only consumer of those dicts, and pi.dev's event stream is richer than
OpenCode's polling model. We will redesign the tracer interface to consume native
pi turn structures (`TurnData`) and iterate on the schema after inspecting real
traces in MLflow.

**MVP first:** Build the client and a minimal `log_turn()` tracer method, run
one end-to-end simulation, inspect the MLflow trace hierarchy, then refine the
schema before writing full tests.

---

## pi.dev JSONL Format Reference

Confirmed from live runs of `pi --mode json --no-session` (pi v0.68.1):

```jsonl
{"type":"session","version":3,"id":"uuid","timestamp":"...","cwd":"/path"}
{"type":"agent_start"}
{"type":"turn_start"}
{"type":"message_start","message":{"role":"user","content":[],"timestamp":...}}
{"type":"message_end","message":{"role":"user","content":[],"timestamp":...}}
{"type":"message_start","message":{"role":"assistant","content":[],"usage":{...},"stopReason":"stop",...}}
{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"Hello",...}}
{"type":"message_end","message":{"role":"assistant","content":[],"usage":{...},"stopReason":"stop",...}}
{"type":"tool_execution_start","toolCallId":"...","toolName":"bash","args":{...}}
{"type":"tool_execution_end","toolCallId":"...","toolName":"bash","result":{...},"isError":false}
{"type":"turn_end","message":{...},"toolResults":[...]}
```

Event types we care about:
- **Lifecycle:** `agent_start` (no `agent_end` emitted in practice)
- **Turn:** `turn_start`, `turn_end` (only when tools used)
- **Message:** `message_start`, `message_update`, `message_end`
- **Tool:** `tool_execution_start`, `tool_execution_end`

Key findings from live capture:
- `message_end` includes `usage: {input, output, cacheRead, cacheWrite, totalTokens, cost: {...}}`
- Multiple `message_end` events per turn (user + assistant)
- **`agent_end` is NOT emitted.** Natural completion = EOF on stdout after last `message_end`
- **`turn_end` is only emitted when tools are used.** For no-tool responses, stream ends after assistant `message_end`
- **Turn counting:** Count assistant `message_end` events (one per turn)
- **Completion detection:** EOF on stdout = done. Cannot distinguish crash from clean exit except by absence of expected events.

---

## Phase 1: Foundation & Nix

**Goal:** `pi` is available in the dev shell. We can spawn it, read JSONL events,
and buffer them into turns.

**Key outcomes needed for Phase 2:**
- `PiDevClient` exists with subprocess lifecycle management
- `TurnData` schema defined (native pi structure, not OpenCode-compatible)
- Real fixture captured and committed
- Exception boundary: `PiDevError`, `PiDevTimeoutError`

**Design notes (confirmed from live runs):**
- Prompt can be passed as positional arg, piped stdin, or `@file.md`
- Keep the `llm-agents.nix` revision pin. Reproducibility matters.
- Stderr: `stderr=open(log_path, "w")` to avoid pipe deadlock.
- Timeout: `threading.Timer` → `killpg` + `popen.stdout.close()` to unblock reader.
- Process cleanup: `try/finally` in orchestrator run is sufficient.
- Premature EOF (no `message_end` seen) must raise `PiDevError`.
- POSIX-only (`killpg`). Document this assumption.
- `max_turns` enforced by counting assistant `message_end` events, then killing subprocess.

---

## Phase 2: Configuration & Mapping

**Goal:** Configs use pi-native provider names. The `agent` parameter is gone.

**Key outcomes needed for Phase 3:**
- `ModelConfig` has a `provider: str` field
- `models.yaml` updated with pi-native names (e.g., `kimi-for-coding` → `kimi-coding`)
- `agent` removed from `SimulationConfig`, orchestrator, `run_simulation()`, CLI
- Model ID and provider passed separately to pi CLI

---

## Phase 3: Core Refactor

**Goal:** Orchestrator streams turns instead of polling messages. Tracer consumes
native turns.

**Key outcomes needed for Phase 4:**
- `SimulationOrchestrator` iterates over `PiDevClient.messages()` yielding pi-native message dicts
- `MlflowTracer.log_message(message: dict)` adapted for pi-native assistant `message_end` events
- Completion detected by stdout EOF or `max_turns` (no `agent_end` in practice)
- Remove the old `finish == "stop"` heuristic (pi signals completion via stream end)
- Git diff capture remains unchanged

**MVP scope for `log_message()` (pi-native):**
- One parent span per assistant `message_end`
- Map assistant content (text + thinking) to span outputs
- Tool executions as child spans under their parent message
- Token usage from `message_end.message.usage`
- Raw message JSON as debug attribute (truncated to 4KB)

**Dropped (pi.dev lacks timing data):**
- Per-message performance metrics (TTFT, TPS)
- `avg_ttft_ms`, `avg_tps`, `total_generation_time_ms` from `trace_session.json`
- May re-add approximate turn durations in later phase if needed

---

## Phase 4: Testing & Validation

**Goal:** Tests exist. MVP trace looks correct in MLflow.

**Phase 4.1: E2E First (Human Checkpoint)**
Run one end-to-end simulation with `kimi-coding`. Inspect MLflow UI. Refine
message span schema based on observations. This is the FIRST gate.

**Phase 4.2–4.5: Unit tests (after schema is locked)**
- `test_pi_dev.py`: Client lifecycle, timeout, premature EOF, abort
- `test_mlflow_tracer.py`: Span creation, hierarchy, attributes
- `test_orchestration.py`: Streaming semantics, completion status
- `test_cli.py`: Settings wired correctly

---

## Phase 5: Cleanup

**Goal:** Old code archived. Docs updated.

- Archive (do not delete) `opencode.py` and `test_opencode.py` to `plan/archive/`
- Remove `opencode_url` from settings
- Remove `opencode` from `flake.nix`
- Update README and completed plan docs

---

## Effort Estimate

| Phase | Effort | Checkpoint |
|-------|--------|------------|
| 1 | 0.5–1 day | Client + fixture committed |
| 2 | 0.5 day | Config committed |
| 3 | 0.5–1 day | Core refactor committed |
| 4.1 | 0.5 day | **MVP e2e validated in MLflow** |
| 4.2–4.5 | 0.5–1 day | Unit tests green |
| 5 | 0.5 day | Archive + docs |
| **Total** | **~2–3 days** | |

---

## Open Questions / Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Native `TurnData` schema, no OpenCode adapter | Tracer is the only consumer; pi's model is richer |
| 2 | Keep Nix pin | Reproducibility > convenience |
| 3 | Drop `agent` parameter | pi.dev has no agent concept |
| 4 | `stderr=open(log_path, "w")` | Avoids pipe deadlock |
| 5 | `try/finally` cleanup | Subprocess lifetime matches one simulation run |
| 6 | Completion by EOF on stdout | `agent_end` not emitted in practice |
| 7 | `max_turns` via assistant `message_end` count | `turn_end` only emitted with tools |
| 8 | POSIX-only (`killpg`) | Documented assumption |
| 9 | MVP e2e before unit tests | Inspect MLflow first, then lock schema |
| 10 | Drop TTFT/TPS metrics | pi.dev lacks per-message timing data |
| 11 | Message-level spans, not turn-level | One parent span per assistant `message_end` |

---

## Changelog

Use this section to record what actually happened as each phase is completed.

### Phase 1 — COMPLETE

**Implemented:**

1. `flake.nix` — `opencode` removed, `pi` added from `llm-agents.nix`
2. `src/beyond_vibes/simulations/pi_dev.py` — `PiDevClient` class
   - Spawns `pi --mode json --no-session --provider kimi-coding --model kimi-for-coding --print <prompt>`
   - Reads stdout JSONL line-by-line
   - Buffers events into `TurnData` objects (assistant `message_end` events)
   - `max_turns` enforced by counting assistant `message_end` events, then `killpg`
   - `stderr` redirected to log file to avoid pipe deadlock
3. Exception boundary: `PiDevError`, `PiDevTimeoutError`
4. `tests/fixtures/pi_dev_output.jsonl` — real captured fixture from live run
5. `tests/test_pi_dev.py` — 14 unit tests, all passing
6. `scripts/test_phase1.py` — manual verification script

**Learnings:**

- Prompt passed as positional arg after `--print`. Piping stdin also works but positional is simpler.
- `message_end` is source of truth for content. Deltas in `message_update` are streaming-only.
- Content blocks: `thinking` (with `thinkingSignature`) and `text` (with `index`).
- Usage schema: `{input, output, cacheRead, cacheWrite, totalTokens, cost: {input, output, cacheRead, cacheWrite, total}}`.
- `agent_end` is indeed not emitted. EOF on stdout after last `message_end` = clean completion.
- `--system-prompt` flag supported; passed before positional prompt.

**Manual test result:** PASS. `scripts/test_phase1.py` yields correct `TurnData` with stop_reason, usage, and content blocks. Subprocess reaped cleanly.

### Phase 2 — COMPLETE

**Implemented:**

1. `src/beyond_vibes/simulations/models.py` — Removed `agent` field from `SimulationConfig`
2. `src/beyond_vibes/simulations/orchestration.py` — Removed `agent` parameter from `SimulationOrchestrator.run()` and `run_simulation()`
3. `models.yaml` — Updated provider names to pi-native:
   - `k2p6` → `provider: kimi-coding`, `model_id: k2p6`
   - `minimax-m2.5-free` → `provider: minimax`, `model_id: minimax-m2.5-free`
   - `qwen3-0.6B` → `provider: local` (unchanged, no pi equivalent yet)
4. Task YAMLs — Removed `agent: "orchestrator"` from `poetry_to_uv.yaml` and `unit_tests.yaml`
5. `tests/test_orchestration.py` — Removed all `agent="orchestrator"` references
6. `tests/test_cli.py` — Removed all `sim_config.agent = "build"` references

**Learnings:**

- `k2p6` maps to pi provider `kimi-coding`, model `k2p6` (confirmed via `pi --list-models`)
- `minimax-m2.5-free` maps to pi provider `minimax`, model `minimax-m2.5-free`
- No pi-native `local` provider for GGUF models (no ollama/lmstudio provider found in pi v0.68.1). Keeping `local` as a sentinel for now; will need custom handling or `huggingface` provider if running local inference through pi.
- `ModelConfig.provider` and `ModelConfig.model_id` pass directly to `--provider` and `--model` flags.

**Tests:** All 186 tests pass. Ruff clean.

### Phase 3 — COMPLETE

**Implemented:**

1. `src/beyond_vibes/simulations/orchestration.py` — Rewritten for streaming
   - `SimulationOrchestrator` now takes `PiDevClient` instead of `OpenCodeClient`
   - `run()` iterates over `PiDevClient.run()` yielding `TurnData` directly
   - No polling loop, no message deduplication, no `time.sleep(5)`
   - Completion status: `completed` (natural EOF), `max_turns` (`pi_client.max_turns_reached`), `error` (exception)
   - Git diff capture remains in `finally` block
   - `system_prompt` passed through to pi CLI

2. `src/beyond_vibes/simulations/mlflow.py` — Added `log_turn()` method
   - Consumes native `TurnData` instead of OpenCode message dicts
   - Parent span per turn (`turn_{index}`) with token usage from `turn.usage`
   - Content blocks mapped to span outputs (`text`, `thinking`)
   - Tool calls/results matched by `toolCallId`, rendered as child `TOOL` spans
   - Raw message JSON stored as attribute (truncated to 4KB)
   - Accumulates session totals for cost, tokens, tool calls
   - Old `log_message()` kept intact for backward compatibility (will archive in Phase 5)

3. `src/beyond_vibes/cli.py` — Wired to `PiDevClient`
   - `PiDevClient(provider=model_config.provider, model=model_config.get_model_id())` constructed per simulation
   - Passed to `run_simulation()` which passes it to `SimulationOrchestrator`

4. `src/beyond_vibes/simulations/pi_dev.py` — Added `max_turns_reached` property and `timestamp` field to `TurnData`

5. `tests/test_orchestration.py` — Completely rewritten for PiDevClient
   - Tests cover: success, sandbox failure, max_turns, exception, git diff capture, system prompt propagation
   - `run_simulation` tests verify `log_turn()` calls and error handling

6. `tests/test_cli.py` — Patches updated from `OpenCodeClient` to `PiDevClient`

**Learnings:**

- `PiDevClient` is pre-configured with provider/model, so `orchestrator.run()` no longer needs `model_id`/`provider` params
- `max_turns_reached` flag is needed to distinguish natural completion from limit hit
- `_create_tool_span_from_pi()` reuses existing `_accumulate_tool_call()` for loop detection
- `TurnData.timestamp` captured from `message_end.message.timestamp` (useful for ordering)

**Tests:** 184 tests pass. Ruff clean.

### Phase 4 — COMPLETE

**E2E Validation Results:**

1. `uv run python -m beyond_vibes.cli simulate --task e2e_test --model k2p6`
   - Completed in ~15s, 1 turn, natural stop
   - MLflow trace shows `thinking` + `text` content blocks correctly
   - `stop_reason: stop`, usage populated

2. `uv run python -m beyond_vibes.cli simulate --task unit_tests --model k2p6`
   - Ran for ~5min, 14 turns, hit max_turns
   - **BUG FOUND:** No tool calls recorded in MLflow traces

**Root Cause Analysis:**

In `PiDevClient._read_turns()`, when `message_end` (assistant) was received, the code did:
```python
yield current_turn
assistant_count += 1
current_turn = None  # BUG!
```

But pi's JSONL stream emits tool events **after** `message_end`:
```
message_end(assistant) ← turn yielded here, current_turn = None
tool_execution_start   ← skipped because current_turn is None
tool_execution_end     ← skipped because current_turn is None
turn_end               ← optional
```

So all tool calls were silently dropped.

**Fix committed:**
- Introduced `pending_turn` buffer
- Yield deferred to next `message_start(assistant)` or EOF, not at `message_end`
- Tool events append to `pending_turn` when `current_turn` is None
- `turn_end` event does not change behavior (it was already optional)

**Unit tests added:**
- `TestLogTurn` (8 tests): no session, text content, thinking content, usage attributes, tool calls, tool errors, session accumulation, message data append
- `TestPiDevClientToolRun` (2 tests): tool call capture with fixture, tool call without message_start
- `tests/fixtures/pi_dev_tool_output.jsonl`: real tool call sequence fixture

**Additional improvements:**
- Added `cache_read_tokens`, `cache_write_tokens`, `response_id` to MLflow span attributes
- Removed `agent="build"` from `mock_simulation_config` fixture (field no longer exists)

**Verified pi data structure for tool calls:**
- `message_end.content` includes `{"type": "toolCall", "id": "...", "name": "...", "arguments": {...}}`
- `tool_execution_start` has `toolCallId`, `toolName`, `args`
- `tool_execution_end` has `toolCallId`, `toolName`, `result`, `isError`
- `turn_end` includes `toolResults` array (redundant with separate events)

**What's in pi but NOT in MLflow traces:**
- `session` event (session ID, cwd)
- `agent_start` / `agent_end` lifecycle
- `turn_start` / `turn_end` boundaries
- `thinkingSignature` on thinking blocks
- `api`, `provider`, `model` on message (redundant with run params)
- `cost` breakdown per category (only total logged)

**Tests:** 194 tests pass. Ruff clean.

### Phase 5
_(Update after implementation)_
