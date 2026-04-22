# Migration: OpenCode Server → pi.dev CLI

## Context

OpenCode did not have the `run` feature when this project started. The current
architecture uses `opencode serve` (HTTP server on :4096) and polls
`GET /session/{id}/message` every 5 seconds. This server approach has proven
unreliable.

The `opencode run --format json` feature now exists, but **pi.dev** is preferred
because:
- Smaller footprint (standalone Node.js CLI, no daemon)
- Purpose-built for headless/CI execution
- Rich JSONL event stream with no polling required
- Native one-shot invocation via `pi --mode json`

## Overview

Replace the HTTP polling architecture with a subprocess-based pi.dev CLI
execution using `pi --mode json`.

## Phase 1: Foundation (No Breaking Changes)

### 1.1 Create `PiDevClient`

New file: `src/beyond_vibes/simulations/pi_dev.py`

- Run `pi --mode json --no-session` as subprocess
- Stream JSONL events from stdout
- Handle timeout, stderr, process lifecycle
- Configurable: `pi_dev_path`, `provider`, `model`, `tools`, `thinking`

### 1.2 Create Message Adapter

New module or within `pi_dev.py`

- Map pi.dev events → OpenCode-like `message` dicts for `MlflowTracer`
- Key mappings:
  - `message_end` event → `{"info": {...}, "parts": [...]}`
  - `tool_execution_end` → tool `parts` with `state`
  - `turn_end` → assistant message with finish signal

### 1.3 Add Settings

- `pi_dev_path: str = "pi"`
- `pi_dev_session_dir: str | None = None`
- Remove/deprecate `opencode_url`

## Phase 2: Core Refactor

### 2.1 Refactor `SimulationOrchestrator`

- Replace HTTP polling loop with event streaming
- Read pi.dev JSONL events as they arrive (no `time.sleep(5)`)
- Rebuild same semantics:
  - Deduplicate by message ID
  - Count assistant turns with meaningful content
  - Detect stop/finish signals
  - Handle max turns
  - Capture git diff

### 2.2 Update `run_simulation()`

- Swap `OpenCodeClient` for `PiDevClient`
- Pass through model/provider from `models.yaml`
- Keep `MlflowTracer` integration unchanged

### 2.3 Update CLI

- Instantiate `PiDevClient` instead of `OpenCodeClient`
- Wire up new settings

## Phase 3: Configuration & Mapping

### 3.1 Update `models.yaml`

Map provider names to pi.dev `--provider` values.

Examples:
- `kimi-for-coding` → `kimi` or keep as-is if pi.dev supports it
- `opencode` → ??? (check if pi.dev has equivalent)
- `local` → `local` or custom endpoint

### 3.2 Add Provider Mapping Config

May need a lookup table if pi.dev uses different provider names.

## Phase 4: Testing

### 4.1 Unit Tests for `PiDevClient`

- Mock subprocess stdout with captured pi.dev JSONL events
- Test event parsing, error handling, timeout

### 4.2 Unit Tests for Message Adapter

- Test each pi.dev event type maps correctly
- Verify `MlflowTracer`-compatible output

### 4.3 Update Existing Tests

- `test_opencode.py` → rename to `test_pi_dev.py`
- `test_orchestration.py` → update mocks from HTTP to subprocess

### 4.4 Integration Test

- Run pi.dev against a real repo with a cheap model

## Phase 5: Cleanup

### 5.1 Remove Old Code

- Delete `src/beyond_vibes/simulations/opencode.py`
- Remove `opencode_url` from settings

### 5.2 Update Documentation

- `plan/completed/simulations.md` or new doc
- README if applicable

## Effort Estimate

| Phase | Effort | Files Touched |
|-------|--------|---------------|
| 1 | 1-2 days | 2 new files, settings.py |
| 2 | 2-3 days | orchestration.py, cli.py, pi_dev.py |
| 3 | 0.5 day | models.yaml, model_config.py |
| 4 | 2-3 days | test files |
| 5 | 0.5 day | docs, cleanup |
| **Total** | **~6-9 days** | |

## Open Questions

1. **Provider mapping**: What are the exact pi.dev `--provider` values? Does
   `kimi-for-coding` map to `kimi`? What's the equivalent of the `opencode`
   provider?

2. **Session state**: pi.dev `--no-session` is ephemeral. Do we need session
   persistence for any reason, or is each simulation fully independent?

3. **Model identifiers**: pi.dev uses `--model provider/id` pattern. How do
   current `model_id` values map? (e.g., `k2p6` → what in pi.dev?)

4. **Tool control**: pi.dev defaults to `read,bash,edit,write`. Do we want to
   restrict tools for safety (e.g., `--tools read,bash`)?

5. **Parallel execution**: With the server, multiple simulations could share one
   process. With CLI subprocesses, each simulation spawns its own `pi` process.
   Is resource usage a concern?

6. **Argo workflows**: Will pi.dev be pre-installed in the container image, or
   installed per-run?

## Recommendation

Start with **Phase 1** (build `PiDevClient` + adapter + tests) and validate it
works end-to-end before touching the orchestrator.
