"""Simulation orchestration - polling and message deduplication."""

import logging
import time
from typing import Generator

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.mlflow import MlflowTracer
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.opencode import OpenCodeClient
from beyond_vibes.simulations.sandbox import SandboxManager

logger = logging.getLogger(__name__)


def _has_meaningful_content(msg: dict) -> bool:
    """Check if a message has any non-empty parts (text, reasoning, or tool)."""
    parts = msg.get("parts", [])
    if not parts:
        return False
    for part in parts:
        part_type = part.get("type", "")
        if part_type in ("text", "reasoning") and part.get("text"):
            return True
        if part_type == "tool":
            return True
    return False


class SimulationOrchestrator:
    """Handles polling and message deduplication for simulation runs."""

    def __init__(
        self,
        opencode_client: OpenCodeClient,
        tracer: MlflowTracer,
        sandbox_manager: SandboxManager,
    ) -> None:
        """Initialize the orchestrator."""
        self.opencode = opencode_client
        self.tracer = tracer
        self.sandbox = sandbox_manager
        self._seen_message_ids: set[str] = set()
        self._assistant_message_count: int = 0
        self._session_id: str | None = None
        self._completion_status: str | None = None

    @property
    def completion_status(self) -> str | None:
        """Return the completion status of the simulation."""
        return self._completion_status

    def run(  # noqa: PLR0912, PLR0913
        self,
        repo_url: str,
        branch: str,
        prompt: str,
        model_id: str,
        provider: str,
        max_turns: int = 75,
        capture_git_diff: bool = False,
    ) -> Generator[dict, None, None]:
        """Run simulation and yield new messages as they arrive."""
        with self.sandbox.sandbox(url=repo_url, branch=branch) as working_dir:
            if working_dir is None:
                raise RuntimeError("Failed to create sandbox")

            logger.info("Running simulation in %s", working_dir)

            self._session_id = self.opencode.create_session(working_dir)
            self.opencode.send_prompt(self._session_id, prompt, model_id, provider)

            try:
                while True:
                    messages = self.opencode.get_messages(self._session_id)

                    logger.debug(
                        "Got %d messages, seen=%d",
                        len(messages),
                        len(self._seen_message_ids),
                    )

                    for msg in messages:
                        msg_id = msg.get("info", {}).get("id", "")
                        # Only yield messages with completed timestamp
                        is_complete = (
                            msg.get("info", {}).get("time", {}).get("completed")
                            is not None
                        )
                        if msg_id not in self._seen_message_ids and is_complete:
                            logger.debug("Yielding completed message id=%s", msg_id)
                            yield msg
                            self._seen_message_ids.add(msg_id)
                            # Count assistant messages
                            if msg.get("info", {}).get("role") == "assistant":
                                has_content = _has_meaningful_content(msg)
                                if has_content:
                                    self._assistant_message_count += 1
                                # Check if this message signals completion
                                if (
                                    msg.get("info", {}).get("finish") == "stop"
                                    and not has_content
                                ):
                                    logger.info(
                                        "Stop signal in empty msg %s, "
                                        "ending session %s",
                                        msg_id,
                                        self._session_id,
                                    )
                                    self._completion_status = "completed"
                                    self.opencode.abort_session(self._session_id)
                                    return

                    # Check if max turns exceeded
                    if self._assistant_message_count >= max_turns:
                        logger.warning(
                            "Max turns (%d) reached, aborting session %s",
                            max_turns,
                            self._session_id,
                        )
                        self._completion_status = "max_turns"
                        self.opencode.abort_session(self._session_id)
                        break

                    time.sleep(5)
            except Exception:
                logger.exception("Simulation interrupted, aborting session")
                self._completion_status = "error"
                if self._session_id:
                    self.opencode.abort_session(self._session_id)
                raise
            finally:
                if capture_git_diff:
                    diff = self.sandbox.get_git_diff()
                    if diff:
                        self.tracer.log_git_diff(diff)
                        logger.info("Git diff captured (%d bytes)", len(diff))
                    else:
                        logger.debug("No git diff to capture (no changes)")


def run_simulation(  # noqa: PLR0913
    sim_config: SimulationConfig,
    model_config: ModelConfig,
    sandbox: SandboxManager,
    opencode_client: OpenCodeClient,
    tracer: MlflowTracer,
    prompt: str,
) -> bool:
    """Execute the simulation and return True if error occurred."""
    error_occurred = False
    try:
        with tracer.log_simulation(sim_config, model_config) as logger_ctx:
            logger_ctx.log_system_prompt(prompt)
            orchestrator = SimulationOrchestrator(opencode_client, tracer, sandbox)

            for message in orchestrator.run(
                repo_url=sim_config.repository.url,
                branch=sim_config.repository.branch,
                prompt=prompt,
                model_id=model_config.get_model_id(),
                provider=model_config.provider,
                max_turns=sim_config.max_turns,
                capture_git_diff=sim_config.capture_git_diff,
            ):
                logger_ctx.log_message(message)

            # Capture completion status for MLflow tagging
            if orchestrator.completion_status:
                tracer.set_completion_status(orchestrator.completion_status)

            logger.info("Simulation completed")

    except Exception as e:
        logger.error("Simulation failed: %s", e)
        error_occurred = True
        if tracer.session:
            tracer.log_error(str(e))

    return error_occurred
