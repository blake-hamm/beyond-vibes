"""Simulation orchestration - stream turns from pi.dev CLI."""

import logging
import traceback
from typing import Generator

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.mlflow import MlflowTracer
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.pi_dev import (
    PiDevClient,
    PiDevError,
    PiDevTimeoutError,
    TurnData,
)
from beyond_vibes.simulations.sandbox import SandboxManager

logger = logging.getLogger(__name__)


class SimulationOrchestrator:
    """Streams turns from pi.dev CLI during simulation runs."""

    def __init__(
        self: "SimulationOrchestrator",
        pi_client: PiDevClient,
        tracer: MlflowTracer,
        sandbox_manager: SandboxManager,
    ) -> None:
        """Initialize the orchestrator."""
        self.pi = pi_client
        self.tracer = tracer
        self.sandbox = sandbox_manager
        self._completion_status: str | None = None
        self._turns: list[TurnData] = []

    @property
    def completion_status(self: "SimulationOrchestrator") -> str | None:
        """Return the completion status of the simulation."""
        return self._completion_status

    def run(  # noqa: PLR0913
        self: "SimulationOrchestrator",
        repo_url: str,
        branch: str,
        prompt: str,
        max_turns: int = 75,
        capture_git_diff: bool = False,
        system_prompt: str | None = None,
    ) -> Generator[TurnData, None, None]:
        """Run simulation and yield turns as they arrive from pi.dev."""
        self._turns = []
        with self.sandbox.sandbox(url=repo_url, branch=branch) as working_dir:
            if working_dir is None:
                raise RuntimeError("Failed to create sandbox")

            logger.info("Running simulation in %s", working_dir)

            try:
                for turn in self.pi.run(
                    prompt=prompt,
                    working_dir=working_dir,
                    max_turns=max_turns,
                    system_prompt=system_prompt,
                ):
                    self._turns.append(turn)
                    logger.debug(
                        "Yielding turn %d (stop_reason=%s)",
                        turn.turn_index,
                        turn.stop_reason,
                    )
                    yield turn

                logger.debug(
                    "pi.run() generator exhausted: max_turns_reached=%s turns=%d",
                    self.pi.max_turns_reached,
                    len(self._turns),
                )
                if self.pi.max_turns_reached:
                    logger.warning("Max turns (%d) reached for simulation", max_turns)
                    self._completion_status = "max_turns"
                else:
                    self._completion_status = "completed"
                    logger.info("Simulation completed naturally")

            except PiDevTimeoutError:
                logger.warning("Simulation timed out after %ds", self.pi.timeout)
                self._completion_status = "timeout"
                raise
            except Exception:
                logger.exception("Simulation interrupted")
                self._completion_status = "error"
                raise
            finally:
                if capture_git_diff:
                    diff = self.sandbox.get_git_diff()
                    if diff:
                        self.tracer.log_git_diff(diff)
                        logger.info("Git diff captured (%d bytes)", len(diff))
                    else:
                        logger.debug("No git diff to capture (no changes)")

    def check_turn_errors(self: "SimulationOrchestrator") -> str | None:
        """Return error message if any turn had stop_reason == 'error'."""
        for turn in self._turns:
            if turn.stop_reason == "error" and turn.error_message:
                return turn.error_message
        return None


def run_simulation(  # noqa: PLR0913
    sim_config: SimulationConfig,
    model_config: ModelConfig,
    sandbox: SandboxManager,
    pi_client: PiDevClient,
    tracer: MlflowTracer,
    prompt: str,
) -> None:
    """Execute the simulation. Raises on failure so MLflow marks the run FAILED."""
    with tracer.log_simulation(sim_config, model_config) as logger_ctx:
        logger_ctx.log_system_prompt(prompt)
        orchestrator = SimulationOrchestrator(pi_client, tracer, sandbox)

        try:
            for turn in orchestrator.run(
                repo_url=sim_config.repository.url,
                branch=sim_config.repository.branch,
                prompt=prompt,
                max_turns=sim_config.max_turns,
                capture_git_diff=sim_config.capture_git_diff,
                system_prompt=sim_config.system_prompt,
            ):
                logger_ctx.log_turn(turn)

            if orchestrator.completion_status:
                tracer.set_completion_status(orchestrator.completion_status)

            if orchestrator.completion_status == "max_turns":
                msg = f"Max turns ({sim_config.max_turns}) reached"
                logger.error(msg)
                tracer.log_error(msg)
                raise RuntimeError(msg)

        except PiDevError as exc:
            # Log error + stderr BEFORE re-raising so MLflow flushes them
            exc_str = traceback.format_exc()
            logger.error("Simulation failed:\n%s", exc_str)
            tracer.log_error(exc_str)
            if exc.stderr:
                tracer.log_stderr(exc.stderr)
            raise
        except Exception:
            exc_str = traceback.format_exc()
            logger.error("Simulation failed:\n%s", exc_str)
            tracer.log_error(exc_str)
            raise

        # Detect failures even when pi exits 0 (e.g. API errors)
        logger.debug(
            "Post-run check: completion_status=%s turns=%d",
            orchestrator.completion_status,
            len(orchestrator._turns),
        )
        turn_error = orchestrator.check_turn_errors()
        if turn_error:
            msg = f"pi turn failed: {turn_error}"
            logger.error(msg)
            tracer.log_error(msg)
            raise RuntimeError(msg)

        logger.info("Simulation completed successfully")
