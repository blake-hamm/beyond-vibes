"""MLflow tracing integration for simulation runs."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

import mlflow

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.models import SimulationConfig

logger = logging.getLogger(__name__)


def generate_session_id(model_config: ModelConfig, quant_tag: str | None = None) -> str:
    """Generate a unique session ID from model config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant = quant_tag or "fp16"
    return f"{model_config.name}_{quant}_{timestamp}"


@dataclass
class MessageData:
    """Data for a single message in the simulation."""

    message_index: int
    timestamp: datetime
    raw_message: dict | None = None


@dataclass
class SimulationSession:
    """Complete simulation session data."""

    sim_config: SimulationConfig
    model_config: ModelConfig
    quant_tag: str | None = None
    session_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    total_messages: int = 0
    total_time_seconds: float | None = None
    messages: list[MessageData] = field(default_factory=list)
    git_diff: str | None = None
    error: str | None = None


class MlflowTracer:
    """MLflow tracer - captures simulation data as traces for later DSPy eval."""

    def __init__(
        self,
        experiment_name: str = "beyond-vibes",
        quant_tag: str | None = None,
        container_tag: str | None = None,
    ) -> None:
        """Initialize simulation logger with MLflow tracing."""
        self.experiment_name = experiment_name
        self.run_id: str | None = None
        self.session: SimulationSession | None = None
        self.quant_tag = quant_tag
        self.container_tag = container_tag

    @contextmanager
    def log_simulation(
        self,
        sim_config: SimulationConfig,
        model_config: ModelConfig,
    ) -> Generator["MlflowTracer", Any, None]:
        """Context manager for tracing a simulation run."""
        session_id = generate_session_id(model_config, self.quant_tag)

        self.session = SimulationSession(
            sim_config=sim_config,
            model_config=model_config,
            quant_tag=self.quant_tag,
            session_id=session_id,
            started_at=datetime.now(),
        )

        mlflow.set_experiment(self.experiment_name)

        try:
            with mlflow.start_run(run_name=session_id) as run:
                self.run_id = run.info.run_id
                logger.info(
                    "Started MLflow run %s for session: %s",
                    self.run_id,
                    session_id,
                )

                # Parameters (performance analysis)
                mlflow.log_param("model.name", model_config.name)
                mlflow.log_param("model.provider", model_config.provider)
                mlflow.log_param("model.model_id", model_config.get_model_id())
                if model_config.repo_id:
                    mlflow.log_param("model.repo_id", model_config.repo_id)
                if self.quant_tag:
                    mlflow.log_param("model.quant", self.quant_tag)
                if self.container_tag:
                    mlflow.log_param("runtime.container", self.container_tag)

                # Tags (filtering)
                mlflow.set_tag("task.name", sim_config.name)
                mlflow.set_tag("task.archetype", sim_config.archetype)
                mlflow.set_tag("repository.url", sim_config.repository.url)
                mlflow.set_tag("repository.branch", sim_config.repository.branch)

                yield self

                self._flush()

        except Exception as e:
            logger.error("Failed to start MLflow run: %s", e)
            raise

    def log_message(self, message: dict) -> None:
        """Log a raw message as a span in the trace with accurate timestamps."""
        if self.session is None:
            logger.warning("No active session to log message to")
            return

        message_index = len(self.session.messages)

        # Extract timestamps from message metadata (milliseconds -> nanoseconds)
        time_info = message.get("info", {}).get("time", {})
        created_ms = time_info.get("created")
        completed_ms = time_info.get("completed")

        # Convert milliseconds to nanoseconds for MLflow
        start_time_ns = int(created_ms * 1_000_000)
        end_time_ns = int(completed_ms * 1_000_000)

        # Create independent span with session_id in metadata
        # Each span is its own root trace, but shares the session_id
        span = mlflow.start_span_no_context(
            name=f"message_{message_index}",
            start_time_ns=start_time_ns,
            metadata={"mlflow.trace.session": self.session.session_id},
        )

        span.set_inputs(
            {
                "message_index": message_index,
                "message_id": message.get("info", {}).get("id", ""),
                "role": message.get("info", {}).get("role", ""),
            }
        )

        span.set_outputs({"raw_message": message})

        # End span with custom timestamp
        span.end(end_time_ns=end_time_ns)

        msg_data = MessageData(
            message_index=message_index,
            timestamp=datetime.now(),
            raw_message=message,
        )
        self.session.messages.append(msg_data)

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

    def _flush(self) -> None:
        """Flush all accumulated data to MLflow."""
        if not self.run_id or self.session is None:
            return

        self.session.completed_at = datetime.now()
        if self.session.started_at and self.session.completed_at:
            self.session.total_time_seconds = (
                self.session.completed_at - self.session.started_at
            ).total_seconds()

        self.session.total_messages = len(self.session.messages)

        mlflow.log_metric("total_messages", self.session.total_messages)

        if self.session.total_time_seconds is not None:
            mlflow.log_metric("total_time_seconds", self.session.total_time_seconds)

        if self.session.error:
            mlflow.log_metric("has_error", 1)

        if self.session.git_diff:
            mlflow.log_text(self.session.git_diff, "git_diff.patch")

        logger.info("Flushed session data to MLflow run %s", self.run_id)
