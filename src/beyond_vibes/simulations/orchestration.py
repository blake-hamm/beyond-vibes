"""Simulation orchestration - polling and message deduplication."""

import logging
import time
from typing import Generator

from beyond_vibes.model_downloader.models import ModelConfig
from beyond_vibes.opencode_client import OpenCodeClient
from beyond_vibes.settings import settings
from beyond_vibes.simulations.mlflow import MlflowTracer
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.sandbox import SandboxManager

logger = logging.getLogger(__name__)


class SimulationOrchestrator:
    """Handles polling and message deduplication for simulation runs."""

    def __init__(
        self,
        opencode_client: OpenCodeClient,
        sim_logger: MlflowTracer,
        sandbox_manager: SandboxManager,
    ) -> None:
        """Initialize the orchestrator."""
        self.opencode = opencode_client
        self.logger = sim_logger
        self.sandbox = sandbox_manager
        self._seen_message_ids: set[str] = set()
        self._assistant_message_count: int = 0

    def run(  # noqa: PLR0913
        self,
        repo_url: str,
        branch: str,
        prompt: str,
        model_id: str,
        agent: str,
        max_turns: int = 75,
    ) -> Generator[dict, None, None]:
        """Run simulation and yield new messages as they arrive."""
        with self.sandbox.sandbox(url=repo_url, branch=branch) as working_dir:
            if working_dir is None:
                raise RuntimeError("Failed to create sandbox")

            logger.info("Running simulation in %s", working_dir)

            session_id = self.opencode.create_session(working_dir)
            self.opencode.send_prompt(session_id, prompt, model_id, agent)

            while True:
                messages = self.opencode.get_messages(session_id)

                logger.debug(
                    "Got %d messages, seen=%d",
                    len(messages),
                    len(self._seen_message_ids),
                )

                for msg in messages:
                    msg_id = msg.get("info", {}).get("id", "")
                    if msg_id not in self._seen_message_ids:
                        logger.debug("Yielding new message id=%s", msg_id)
                        yield msg
                        self._seen_message_ids.add(msg_id)
                        # Count assistant messages
                        if msg.get("info", {}).get("role") == "assistant":
                            self._assistant_message_count += 1

                # Check if max turns exceeded
                if self._assistant_message_count >= max_turns:
                    logger.warning(
                        "Max turns (%d) reached, aborting session %s",
                        max_turns,
                        session_id,
                    )
                    self.opencode.abort_session(session_id)
                    break

                # Check if simulation is complete
                if self._is_complete(messages):
                    logger.info("Simulation completed, aborting session %s", session_id)
                    self.opencode.abort_session(session_id)
                    break

                time.sleep(5)

    def _is_complete(self, messages: list[dict]) -> bool:
        """Check if the simulation is complete.

        Args:
            messages: List of messages from the session.

        Returns:
            True if the latest assistant message has finish="stop".

        """
        assistant_messages = [
            m for m in messages if m.get("info", {}).get("role") == "assistant"
        ]

        if not assistant_messages:
            return False

        # API returns messages in reverse chronological order (newest first)
        latest = assistant_messages[0]
        finish = latest.get("info", {}).get("finish")
        return finish == "stop"


def _run_simulation(
    sim_config: SimulationConfig,
    model_config: ModelConfig,
    sandbox: SandboxManager,
    opencode_client: OpenCodeClient,
    sim_logger: MlflowTracer,
) -> bool:
    """Execute the simulation and return True if error occurred."""
    error_occurred = False
    try:
        with sim_logger.log_simulation(sim_config, model_config) as logger_ctx:
            prompt = sim_config.prompt
            if settings.system_prompt:
                prompt = f"{settings.system_prompt}\n\n---\n\n{prompt}"
            if sim_config.system_prompt:
                prompt = f"{sim_config.system_prompt}\n\n---\n\n{prompt}"

            orchestrator = SimulationOrchestrator(opencode_client, sim_logger, sandbox)

            for message in orchestrator.run(
                repo_url=sim_config.repository.url,
                branch=sim_config.repository.branch,
                prompt=prompt,
                model_id=model_config.name,
                agent=sim_config.agent,
                max_turns=sim_config.max_turns,
            ):
                logger_ctx.log_message(message)

            logger.info("Simulation completed")

    except Exception as e:
        logger.error("Simulation failed: %s", e)
        error_occurred = True
        if sim_logger.session:
            sim_logger.log_error(str(e))

    return error_occurred
