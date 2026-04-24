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

From the pi.dev docs (`docs/json.md`), `pi --mode json --no-session --print` emits:

```jsonl
{"type":"session","version":3,"id":"uuid","timestamp":"...","cwd":"/path"}
{"type":"agent_start"}
{"type":"turn_start"}
{"type":"message_start","message":{"role":"assistant","content":[],...}}
{"type":"message_update","message":{...},"assistantMessageEvent":{"type":"text_delta","delta":"Hello",...}}
{"type":"message_end","message":{...}}
{"type":"tool_execution_start","toolCallId":"...","toolName":"bash","args":{...}}
{"type":"tool_execution_end","toolCallId":"...","toolName":"bash","result":{...},"isError":false}
{"type":"turn_end","message":{...},"toolResults":[...]}
{"type":"agent_end","messages":[...]}
```

Event types we care about:
- **Lifecycle:** `agent_start`, `agent_end`
- **Turn:** `turn_start`, `turn_end`
- **Message:** `message_start`, `message_update`, `message_end`
- **Tool:** `tool_execution_start`, `tool_execution_update`, `tool_execution_end`

Key findings from fixture capture:
- `message_end` includes `usage: {input, output, totalTokens, cost: {...}}`
- A single turn emits multiple `message_end` events
- The stream ends naturally after `agent_end`

---

## Phase 1: Foundation & Nix

**Goal:** `pi` is available in the dev shell. We can spawn it, read JSONL events,
and buffer them into turns.

**Key outcomes needed for Phase 2:**
- `PiDevClient` exists with subprocess lifecycle management
- `TurnData` schema defined (native pi structure, not OpenCode-compatible)
- Real fixture captured and committed
- Exception boundary: `PiDevError`, `PiDevTimeoutError`

**Design notes (implement, then update this list):**
- Keep the `llm-agents.nix` revision pin. Reproducibility matters.
- Stderr: `stderr=open(log_path, "w")` to avoid pipe deadlock.
- Timeout: `threading.Timer` → `killpg` + `popen.stdout.close()` to unblock reader.
- Process cleanup: `weakref.WeakValueDictionary` + `atexit.register`.
- Premature EOF (no `agent_end`) must raise `PiDevError`.
- POSIX-only (`killpg`). Document this assumption.

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
- `SimulationOrchestrator` iterates over `PiDevClient.turns()` yielding `TurnData`
- `MlflowTracer.log_turn(turn: TurnData)` creates parent spans + tool child spans
- Completion detected by iterator exhaustion (after `agent_end`) or `max_turns`
- Remove the old `finish == "stop"` heuristic (pi signals completion via stream end)
- Git diff capture remains unchanged

**MVP scope for `log_turn()`:**
- One parent span per turn
- Map assistant content to span outputs
- Tool executions as child spans
- Token usage from aggregated `turn.usage`
- Raw turn JSON as debug attribute

**Deferred until Phase 4.5:**
- Per-message performance metrics (TTFT, TPS)
- Complex latency attribution

---

## Phase 4: Testing & Validation

**Goal:** Tests exist. MVP trace looks correct in MLflow.

**Phase 4.1–4.4: Unit tests**
- `test_pi_dev.py`: Client lifecycle, timeout, premature EOF, abort
- `test_turn_data.py`: Event mapping, usage aggregation edge cases
- `test_mlflow_tracer.py`: Span creation, hierarchy, attributes
- `test_orchestration.py`: Streaming semantics, completion status
- `test_cli.py`: Settings wired correctly

**Phase 4.5: MVP Validation (Human Checkpoint)**
Run one end-to-end simulation with a cheap model. Inspect MLflow UI. Refine
`TurnData` schema and `log_turn()` based on observations. Then complete remaining
tests against the finalized schema.

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
| 4.1–4.4 | 0.5–1 day | Unit tests green |
| 4.5 | 0.5 day | **MVP e2e validated in MLflow** |
| 4.6 | 0.5 day | Remaining tests green |
| 5 | 0.5 day | Archive + docs |
| **Total** | **~3–4 days** | |

---

## Open Questions / Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Native `TurnData` schema, no OpenCode adapter | Tracer is the only consumer; pi's model is richer |
| 2 | Keep Nix pin | Reproducibility > convenience |
| 3 | Drop `agent` parameter | pi.dev has no agent concept |
| 4 | `stderr=open(log_path, "w")` | Avoids pipe deadlock |
| 5 | `weakref` + `atexit` cleanup | No signal handlers |
| 6 | Completion by stream exhaustion | `agent_end` is the source of truth |
| 7 | POSIX-only (`killpg`) | Documented assumption |
| 8 | MVP before full tests | Inspect MLflow first, then finalize schema |

---

## Changelog

Use this section to record what actually happened as each phase is completed.

### Phase 1
_(Update after implementation)_

### Phase 2
_(Update after implementation)_

### Phase 3
_(Update after implementation)_

### Phase 4
_(Update after implementation)_

### Phase 5
_(Update after implementation)_
