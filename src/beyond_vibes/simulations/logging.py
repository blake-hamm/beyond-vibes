"""MLflow logging integration for simulation runs."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

import mlflow

from beyond_vibes.settings import settings
from beyond_vibes.simulations.models import SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class TurnData:
    """Data for a single turn in the simulation."""

    turn_index: int
    timestamp: datetime
    response: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    tps: float | None = None
    ttft: float | None = None
    tokens_generated: int | None = None
    error: str | None = None


@dataclass
class SimulationSession:
    """Complete simulation session data."""

    sim_config: SimulationConfig
    started_at: datetime
    completed_at: datetime | None = None
    total_turns: int = 0
    total_time_seconds: float | None = None
    tps: float | None = None
    ttft: float | None = None
    total_tokens: int | None = None
    turns: list[TurnData] = field(default_factory=list)
    git_diff: str | None = None
    error: str | None = None


class SimulationLogger:
    """MLflow logger - captures raw session data for later DSPy eval."""

    def __init__(
        self,
        experiment_name: str = "beyond-vibes-simulations",
        tracking_uri: str | None = None,
    ) -> None:
        """Initialize simulation logger with MLflow tracking."""
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.run_id: str | None = None
        self.session: SimulationSession | None = None

    @contextmanager
    def log_simulation(
        self, sim_config: SimulationConfig
    ) -> Generator["SimulationLogger", Any, None]:
        """Context manager for logging a simulation run."""
        self.session = SimulationSession(
            sim_config=sim_config,
            started_at=datetime.now(),
        )

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        try:
            with mlflow.start_run(run_name=sim_config.name) as run:
                self.run_id = run.info.run_id
                logger.info(
                    "Started MLflow run %s for simulation: %s",
                    self.run_id,
                    sim_config.name,
                )

                try:
                    mlflow.log_param("task_name", sim_config.name)
                    mlflow.log_param("archetype", sim_config.archetype)
                    mlflow.log_param("repo_url", sim_config.repository.url)
                    mlflow.log_param("repo_branch", sim_config.repository.branch)
                except Exception as e:
                    logger.warning("Failed to log params: %s", e)

                yield self

                self._flush()

        except Exception as e:
            logger.error("Failed to start MLflow run: %s", e)
            raise

    # ruff: noqa: PLR0913
    def log_turn(
        self,
        turn_index: int,
        response: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        tps: float | None = None,
        ttft: float | None = None,
        tokens_generated: int | None = None,
        error: str | None = None,
    ) -> None:
        """Log a single turn in the simulation."""
        if self.session is None:
            logger.warning("No active session to log turn to")
            return

        turn = TurnData(
            turn_index=turn_index,
            timestamp=datetime.now(),
            response=response,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            tps=tps,
            ttft=ttft,
            tokens_generated=tokens_generated,
            error=error,
        )
        self.session.turns.append(turn)

    def log_git_diff(self, diff_content: str) -> None:
        """Log git diff as an artifact."""
        if self.session is None:
            logger.warning("No active session to log git diff to")
            return

        self.session.git_diff = diff_content

    def log_error(self, error_message: str) -> None:
        """Log an error that occurred during the simulation."""
        if self.session is None:
            logger.warning("No active session to log error to")
            return

        self.session.error = error_message

    # ruff: noqa: PLR0912
    def _flush(self) -> None:
        """Flush all accumulated data to MLflow."""
        if not self.run_id or self.session is None:
            return

        try:
            self.session.completed_at = datetime.now()
            if self.session.started_at and self.session.completed_at:
                self.session.total_time_seconds = (
                    self.session.completed_at - self.session.started_at
                ).total_seconds()

            self.session.total_turns = len(self.session.turns)

            total_tokens = 0
            total_tps: list[float] = []
            total_ttft: list[float] = []

            for turn in self.session.turns:
                if turn.tokens_generated:
                    total_tokens += turn.tokens_generated
                if turn.tps is not None:
                    total_tps.append(turn.tps)
                if turn.ttft is not None:
                    total_ttft.append(turn.ttft)

            self.session.total_tokens = total_tokens if total_tokens > 0 else None

            if total_tps:
                self.session.tps = sum(total_tps) / len(total_tps)
            if total_ttft:
                self.session.ttft = sum(total_ttft) / len(total_ttft)

            mlflow.log_metric("total_turns", self.session.total_turns)

            if self.session.total_time_seconds is not None:
                mlflow.log_metric("total_time_seconds", self.session.total_time_seconds)

            if self.session.tps is not None:
                mlflow.log_metric("avg_tps", self.session.tps)

            if self.session.ttft is not None:
                mlflow.log_metric("avg_ttft", self.session.ttft)

            if self.session.total_tokens is not None:
                mlflow.log_metric("total_tokens", self.session.total_tokens)

            if self.session.error:
                mlflow.log_metric("has_error", 1)

            if self.session.git_diff:
                mlflow.log_text(self.session.git_diff, "git_diff.patch")

            logger.info("Flushed session data to MLflow run %s", self.run_id)

        except Exception as e:
            logger.warning("Failed to flush session data to MLflow: %s", e)
