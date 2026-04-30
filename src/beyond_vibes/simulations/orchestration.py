"""Simulation orchestration - stream turns from pi.dev CLI."""

import logging
import traceback
from typing import Generator

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.mlflow import MlflowTracer
from beyond_vibes.simulations.models import SimulationConfig
from beyond_vibes.simulations.pi_dev import PiDevClient, TurnData
from beyond_vibes.simulations.sandbox import SandboxManager

logger = logging.getLogger(__name__)

# Keywords that indicate a failure even when pi exits 0
_STDERR_ERROR_KEYWORDS = (
    "error",
    "404",
    "not found",
    "failed",
    "unauthorized",
    "forbidden",
    "timeout",
    "connection refused",
    "bad request",
)


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
                    logger.debug("Yielding turn %d", turn.turn_index)
                    yield turn

                if self.pi.max_turns_reached:
                    logger.warning("Max turns (%d) reached for simulation", max_turns)
                    self._completion_status = "max_turns"
                else:
                    self._completion_status = "completed"

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

    def check_stderr_for_errors(self: "SimulationOrchestrator") -> str | None:
        """Return stderr content if it contains error indicators."""
        stderr = getattr(self.pi, "_last_stderr", None)
        if not isinstance(stderr, str) or not stderr:
            return None
        lower = stderr.lower()
        for kw in _STDERR_ERROR_KEYWORDS:
            if kw in lower:
                return stderr
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

        except Exception:
            # Log error + stderr BEFORE re-raising so MLflow flushes them
            exc_str = traceback.format_exc()
            logger.error("Simulation failed:\n%s", exc_str)
            tracer.log_error(exc_str)
            stderr = getattr(pi_client, "_last_stderr", None)
            if stderr:
                tracer.log_stderr(stderr)
            raise

        # Detect failures even when pi exits 0 (e.g. fake model)
        stderr_errors = orchestrator.check_stderr_for_errors()
        if stderr_errors:
            msg = f"pi stderr indicates failure:\n{stderr_errors}"
            logger.error(msg)
            tracer.log_error(msg)
            tracer.log_stderr(stderr_errors)
            raise RuntimeError(msg)

        logger.info("Simulation completed")
