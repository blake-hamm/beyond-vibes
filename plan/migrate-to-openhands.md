# Alternative Migration: OpenHands SDK (Python + OTel)

## Context

The primary migration plan (`migrate-to-pi-dev.md`) targets pi.dev as a CLI replacement for the
OpenCode server. This document explores an alternative: using a Python-native SDK with built-in
OpenTelemetry support.

## Important Correction

The "Open Agent SDK" referenced in earlier discussion is **not** the tool it appeared to be:

- `pip install open-agent-sdk` installs an unrelated local-LLM SDK by `slb350`
- The actual CodeAny package is `pip install open-agent-sdk-py` (GitHub: `codeany-ai/open-agent-sdk-python`)
- This is a **very new project** (~26 stars, created April 2026) with limited maturity
- The dominant "Open Agent SDK" in the ecosystem is actually **`openai-agents`** (`pip install openai-agents`, 24K+ stars)

**Neither CodeAny's SDK nor OpenAI's Agents SDK are purpose-built for coding tasks.**

## Recommendation: OpenHands SDK

After re-evaluating, **OpenHands** is the best alternative for this architecture because it uniquely
combines:
- ✅ First-class **Python SDK**
- ✅ Native **OpenTelemetry** support (via Laminar SDK)
- ✅ Purpose-built for **coding tasks**
- ✅ **Headless** mode (no TUI)
- ✅ **Docker sandboxing** support
- ✅ Mature and active (50K+ stars)

### OpenHands vs Other Options

| Tool | Python SDK | OTel | Coding-Focused | Maturity |
|------|:----------:|:----:|:--------------:|:--------:|
| **OpenHands** | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| pi.dev | ❌ CLI only | ❌ | ✅ | ⭐⭐⭐⭐ |
| OpenCode server | ❌ HTTP API | ❌ | ✅ | ⭐⭐⭐ |
| Aider | ⚠️ Unofficial | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Claude Code | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| CodeAny SDK | ✅ | ❓ Unknown | ❌ | ⭐⭐ |
| OpenAI Agents | ✅ | ❓ Unknown | ❌ | ⭐⭐⭐⭐⭐ |

### OTel Support Details

OpenHands exports traces via standard OTel environment variables:
```bash
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://api.honeycomb.io/v1/traces
OTEL_EXPORTER_OTLP_TRACES_HEADERS=x-honeycomb-team=YOUR_API_KEY
```

Auto-traced events:
- Agent execution steps
- Tool calls and executions
- LLM API calls (via LiteLLM)

This means **MLflow could consume OpenHands traces directly** via its OTel integration,
eliminating the need for manual span creation entirely.

## Overview

Replace the HTTP polling architecture with OpenHands' Python SDK, leveraging native
OpenTelemetry for trace export to MLflow.

## Phase 1: Foundation (No Breaking Changes)

### 1.1 Install OpenHands SDK

```bash
pip install openhands-sdk openhands-tools
```

### 1.2 Create `OpenHandsClient`

New file: `src/beyond_vibes/simulations/openhands.py`

```python
from pathlib import Path
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool

class OpenHandsClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.conversation: Conversation | None = None

    def create_session(self, working_dir: Path) -> Conversation:
        llm = LLM(
            model="anthropic/claude-sonnet-4-5",
            api_key=SecretStr(self.api_key) if self.api_key else None,
        )
        agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
            ],
        )
        self.conversation = Conversation(
            agent=agent,
            workspace=str(working_dir),
        )
        return self.conversation

    async def send_prompt(self, prompt: str) -> None:
        self.conversation.send_message(prompt)
        await self.conversation.run()

    def get_events(self) -> list:
        return self.conversation.state.events

    def close(self) -> None:
        # OpenHands handles cleanup
        pass
```

### 1.3 Configure OTel Export

Add to settings or environment:
```bash
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=<mlflow-otlp-endpoint>
OTEL_EXPORTER_OTLP_TRACES_HEADERS=<auth-headers>
```

MLflow can ingest OTel traces directly, potentially **replacing** custom `MlflowTracer` logic.

## Phase 2: Core Refactor

### 2.1 Refactor `SimulationOrchestrator`

Replace polling loop with event-driven completion:

```python
async def run(self, ...):
    with self.sandbox.sandbox(url=repo_url, branch=branch) as working_dir:
        client = OpenHandsClient()
        conversation = client.create_session(working_dir)
        await client.send_prompt(prompt)

        # OpenHands handles turn limits, tool execution, etc.
        # Events are auto-traced via OTel
        events = client.get_events()

        # Yield events for backward compatibility
        for event in events:
            yield self._adapt_event(event)
```

### 2.2 Simplify or Remove `MlflowTracer`

**Option A — Full OTel replacement:**
- Remove `MlflowTracer` entirely
- Configure OpenHands OTel export to MLflow's OTel endpoint
- MLflow auto-ingests traces

**Option B — Hybrid:**
- Keep `MlflowTracer` for custom metrics (cost, tool loop detection)
- Let OpenHands handle execution traces via OTel
- Merge data at evaluation time

### 2.3 Update CLI

```python
# Replace:
with OpenCodeClient() as opencode_client:
    error_occurred = run_simulation(..., opencode_client, ...)

# With:
client = OpenHandsClient(api_key=settings.llm_api_key)
error_occurred = await run_simulation(..., client, ...)
```

## Phase 3: Configuration

### 3.1 Update `models.yaml`

OpenHands uses LiteLLM for model routing. Map providers:

```yaml
models:
  - name: k2p6
    provider: kimi
    model_id: kimi/k2p6
  - name: gpt-4o
    provider: openai
    model_id: openai/gpt-4o
  - name: claude-sonnet
    provider: anthropic
    model_id: anthropic/claude-sonnet-4-5
```

### 3.2 Update Settings

- Remove `opencode_url`
- Add `openhands_model`, `openhands_api_key`
- Add OTel endpoint settings (or use standard env vars)

## Phase 4: Testing

### 4.1 Unit Tests

Mock `Conversation` and `Agent` objects:
```python
@pytest.mark.asyncio
async def test_openhands_client():
    client = OpenHandsClient()
    conversation = client.create_session(Path("/tmp/test"))
    # Mock conversation.run()
    # Assert events are captured
```

### 4.2 Integration Test

Run against a real repo with a cheap model (e.g., `gpt-4o-mini`).

### 4.3 OTel Validation

Verify traces appear in MLflow with correct structure.

## Phase 5: Cleanup

- Delete `opencode.py`
- Simplify or delete `orchestration.py` polling logic
- Update `settings.py`
- Document OTel configuration

## Effort Estimate

| Phase | Effort | Files Touched |
|-------|--------|---------------|
| 1 | 1-2 days | 1 new file, deps |
| 2 | 2-3 days | orchestration.py, cli.py, mlflow.py |
| 3 | 0.5 day | models.yaml, settings.py |
| 4 | 2-3 days | test files, OTel validation |
| 5 | 0.5 day | cleanup, docs |
| **Total** | **~6-9 days** | |

## Comparison: OpenHands vs pi.dev

| Dimension | OpenHands | pi.dev |
|-----------|-----------|--------|
| **Python SDK** | ✅ Native | ❌ Node.js CLI |
| **OTel support** | ✅ Native | ❌ None |
| **Installation** | `pip install` | `npm install -g` |
| **Sandboxing** | ✅ Docker built-in | ⚠️ Manual |
| **Message adapter** | Minimal (Python objects) | Complex (JSONL → dict) |
| **Maturity** | 50K+ stars, research-backed | Active, smaller |
| **Subprocess mgmt** | None (in-process) | Required |
| **Argo Workflows** | ✅ Python container | ⚠️ Node.js container |

## Open Questions

1. **OTel endpoint**: Does the current MLflow setup expose an OTel ingest endpoint, or would traces need manual forwarding?
2. **Custom metrics**: Can OpenHands' OTel traces capture tool loop detection, cost tracking, and other custom metrics, or is a hybrid tracer needed?
3. **Model mapping**: How do current provider strings (`kimi-for-coding`, `opencode`, `local`) map to LiteLLM/OpenHands model identifiers?
4. **Async compatibility**: Is the existing codebase ready for `async/await`, or does it need broader refactoring?

## Recommendation

**OpenHands is the better long-term choice** if:
- OTel integration is valued
- You want to stay in Python (no Node.js dependency)
- Docker sandboxing is desirable
- You prefer in-process execution over subprocess management

**pi.dev remains viable** if:
- You prefer CLI invocation (CI/workflow style)
- Node.js is acceptable
- OTel is not a priority
- You want the smallest possible footprint

For an evaluation pipeline running in Argo workflows with MLflow observability,
**OpenHands + OTel is architecturally cleaner** despite similar implementation effort.
