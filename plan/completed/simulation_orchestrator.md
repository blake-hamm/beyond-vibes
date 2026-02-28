# Simulation Orchestration Refactor Plan

## Overview

Split the current monolithic flow into clean components with a new orchestration layer that handles polling and message deduplication.

---

## 1. `src/beyond_vibes/opencode_client.py` — Simplify to HTTP Wrapper

**Keep:**
- `__init__` - base_url, httpx client setup
- `create_session(working_dir) -> str` - POST /session
- `close()` - cleanup

**New methods:**
- `send_prompt(session_id, prompt, model_id, agent)` → POST /session/{id}/prompt_async
- `get_messages(session_id)` → GET /session/{id}/message (returns raw list)

**Remove:**
- `run_prompt()` - all polling logic moves to orchestration
- `_session_id` tracking - not needed (caller manages)

---

## 2. `src/beyond_vibes/simulations/orchestration.py` — NEW FILE

**Contains:**
- `SimulationOrchestrator` class (polling + deduplication)
- `_run_simulation()` function (moved from cli.py)

**New class: `SimulationOrchestrator`**

```python
class SimulationOrchestrator:
    def __init__(self, opencode_client, sim_logger, sandbox_manager):
        self.opencode = opencode_client
        self.logger = sim_logger
        self.sandbox = sandbox_manager
        self._last_message_id: str | None = None

    def run(
        self,
        repo_url: str,
        branch: str,
        prompt: str,
        model_id: str,
        agent: str,
    ) -> Generator[dict, None, None]:
        """Yield new messages as they arrive."""
        with self.sandbox.sandbox(url=repo_url, branch=branch) as working_dir:
            session_id = self.opencode.create_session(working_dir)
            self.opencode.send_prompt(session_id, prompt, model_id, agent)

            while True:
                messages = self.opencode.get_messages(session_id)

                for msg in messages:
                    if self._last_message_id is None or msg["id"] > self._last_message_id:
                        yield msg
                        self._last_message_id = msg["id"]

                if self._is_complete(messages):
                    break

                time.sleep(5)
```

**Helper:**
- `_is_complete(messages) -> bool` - checks if latest assistant message has `info.time.completed`

---

## 3. `src/beyond_vibes/simulations/mlflow.py` — Log Raw Data Only

**Rename:**
- `TurnData` → `MessageData`
- `SimulationLogger` → `MlflowTracer`
- `SimulationSession.turns` → `messages`

**Changes:**
- Keep `log_simulation()` context manager
- Add `log_message(message: dict)` method that:
  - Stores raw message in `session.messages`
  - Logs as MLflow span with raw content
  - Does **not** calculate tps/ttft/tokens
- Remove any metric calculations from `_flush()`
- Keep `_flush()` for final artifacts (git_diff, final metadata)

---

## 4. `src/beyond_vibes/cli.py` — Thin Entry Point

**Changes:**
- Import `SimulationOrchestrator` from `beyond_vibes.simulations.orchestration`
- Replace `_run_simulation()` logic with orchestrator:

```python
def _run_simulation(...) -> bool:
    try:
        with sim_logger.log_simulation(sim_config, model_config) as logger_ctx:
            orchestrator = SimulationOrchestrator(
                opencode_client, sim_logger, sandbox
            )
            
            for message in orchestrator.run(
                repo_url=sim_config.repository.url,
                branch=sim_config.repository.branch,
                prompt=prompt,
                model_id=model_config.name,
                agent=sim_config.agent,
            ):
                logger_ctx.log_message(message)
                
            logger.info("Simulation completed")
            
    except Exception as e:
        # error handling
```

---

## Migration Steps

1. Create `src/beyond_vibes/simulations/orchestration.py` with `SimulationOrchestrator` and `_run_simulation()`
2. Add `send_prompt()` and `get_messages()` to `src/beyond_vibes/opencode_client.py`
3. Rename `SimulationLogger` → `MlflowTracer` in `src/beyond_vibes/simulations/mlflow.py`
4. Rename `TurnData` → `MessageData` and `SimulationSession.turns` → `messages`
5. Add `log_message()` to `mlflow.py`, remove metric calculations
6. Update `src/beyond_vibes/cli.py` to use orchestrator (import `_run_simulation` from orchestration)
7. Test end-to-end
8. Run `nix develop -c ruff check --fix . && nix develop -c ruff format .`
9. Run `nix develop -c mypy src/`
