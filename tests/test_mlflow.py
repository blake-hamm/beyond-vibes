"""Tests for MLflow tracer."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.mlflow import (
    MlflowTracer,
    SimulationSession,
    generate_session_id,
)
from beyond_vibes.simulations.models import RepositoryConfig, SimulationConfig


@pytest.fixture
def mock_simulation_config() -> SimulationConfig:
    """Create a mock simulation config."""
    return SimulationConfig(
        name="test-task",
        description="Test task",
        archetype="test",
        repository=RepositoryConfig(url="https://github.com/test/repo", branch="main"),
        prompt="Test prompt",
        agent="build",
        max_turns=10,
    )


@pytest.fixture
def mock_model_config() -> ModelConfig:
    """Create a mock model config."""
    return ModelConfig(
        name="test-model",
        repo_id="test/repo",
        provider="local",
        quant_tags=["Q8_0"],
    )


class TestGenerateSessionId:
    """Tests for generate_session_id function."""

    def test_generate_with_quant_tag(self, mock_model_config: ModelConfig) -> None:
        """Test session ID generation with quant tag."""
        session_id = generate_session_id(mock_model_config, quant_tag="Q4_K_M")

        assert session_id.startswith("test-model_Q4_K_M_")
        # Should include timestamp
        assert len(session_id) > len("test-model_Q4_K_M_")

    def test_generate_without_quant_tag(self, mock_model_config: ModelConfig) -> None:
        """Test session ID generation without quant tag defaults to fp16."""
        session_id = generate_session_id(mock_model_config)

        assert "fp16" in session_id
        assert session_id.startswith("test-model_fp16_")


class TestSimulationSession:
    """Tests for SimulationSession dataclass."""

    def test_session_defaults(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test SimulationSession default values."""
        session = SimulationSession(
            sim_config=mock_simulation_config,
            model_config=mock_model_config,
        )

        assert session.sim_config == mock_simulation_config
        assert session.model_config == mock_model_config
        assert session.quant_tag is None
        assert session.session_id == ""
        assert isinstance(session.started_at, datetime)
        assert session.completed_at is None
        assert session.total_messages == 0
        assert session.total_time_seconds is None
        assert session.messages == []
        assert session.git_diff is None
        assert session.error is None
        assert session.total_cost == 0.0
        assert session.total_input_tokens == 0
        assert session.total_output_tokens == 0
        assert session.total_tokens == 0
        assert session.tool_call_counts == {}


class TestMlflowTracerInit:
    """Tests for MlflowTracer initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with default values."""
        tracer = MlflowTracer()

        assert tracer.experiment_name == "beyond-vibes"
        assert tracer.run_id is None
        assert tracer.session is None
        assert tracer.quant_tag is None
        assert tracer.container_tag is None

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        tracer = MlflowTracer(
            experiment_name="custom-experiment",
            quant_tag="Q4_K_M",
            container_tag="v1.0",
        )

        assert tracer.experiment_name == "custom-experiment"
        assert tracer.quant_tag == "Q4_K_M"
        assert tracer.container_tag == "v1.0"


class TestLogSimulation:
    """Tests for log_simulation context manager."""

    def test_log_simulation_success(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test successful simulation logging."""
        tracer = MlflowTracer()

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=mock_run
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            with tracer.log_simulation(
                mock_simulation_config, mock_model_config
            ) as ctx:
                assert ctx is tracer
                assert tracer.session is not None
                assert tracer.run_id == "run-123"

            # Verify MLflow calls
            mock_mlflow.set_experiment.assert_called_once_with("beyond-vibes")
            mock_mlflow.log_param.assert_any_call("model.name", "test-model")
            mock_mlflow.log_param.assert_any_call("model.provider", "local")
            mock_mlflow.set_tag.assert_any_call("task.name", "test-task")
            mock_mlflow.set_tag.assert_any_call("task.archetype", "test")

    def test_log_simulation_with_optional_params(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test logging with optional parameters."""
        tracer = MlflowTracer(quant_tag="Q4_K_M", container_tag="v1.0")
        mock_model_config.repo_id = "test/repo"

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(
                return_value=mock_run
            )
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            with tracer.log_simulation(mock_simulation_config, mock_model_config):
                pass

            # Verify optional params are logged
            mock_mlflow.log_param.assert_any_call("model.quant", "Q4_K_M")
            mock_mlflow.log_param.assert_any_call("runtime.container", "v1.0")
            mock_mlflow.log_param.assert_any_call("model.repo_id", "test/repo")

    def test_log_simulation_exception(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test exception handling in log_simulation."""
        tracer = MlflowTracer()

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_mlflow.set_experiment.side_effect = Exception("MLflow error")

            with pytest.raises(Exception, match="MLflow error"):
                with tracer.log_simulation(mock_simulation_config, mock_model_config):
                    pass


class TestLogMessage:
    """Tests for log_message method."""

    def test_log_message_no_session(self) -> None:
        """Test logging message without active session."""
        tracer = MlflowTracer()

        with patch("beyond_vibes.simulations.mlflow.logger") as mock_logger:
            tracer.log_message({"info": {"id": "msg-1"}})
            mock_logger.warning.assert_called_once()

    def test_log_message_with_text_part(self) -> None:
        """Test logging message with text part."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.model_config = MagicMock()
        tracer.session.model_config.get_model_id.return_value = "test-model"

        message = {
            "info": {
                "id": "msg-1",
                "role": "assistant",
                "time": {"created": 1000, "completed": 2000},
                "cost": 0.01,
                "tokens": {"input": 10, "output": 20},
            },
            "parts": [{"type": "text", "text": "Hello world"}],
        }

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span

            tracer.log_message(message)

            mock_mlflow.start_span_no_context.assert_called_once()
            mock_span.set_inputs.assert_called_once()
            mock_span.set_outputs.assert_called_once()
            mock_span.set_attributes.assert_called()
            mock_span.end.assert_called_once()

    def test_log_message_with_reasoning_part(self) -> None:
        """Test logging message with reasoning part."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.model_config = MagicMock()
        tracer.session.model_config.get_model_id.return_value = "test-model"

        message = {
            "info": {
                "id": "msg-1",
                "role": "assistant",
                "time": {"created": 1000, "completed": 2000},
                "cost": 0.01,
                "tokens": {"input": 10, "output": 20},
            },
            "parts": [{"type": "reasoning", "text": "Thinking..."}],
        }

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span

            tracer.log_message(message)

            # Check that reasoning content is in outputs
            call_args = mock_span.set_outputs.call_args
            assert any(
                part["type"] == "reasoning" for part in call_args[0][0]["content"]
            )

    def test_log_message_with_tool_call(self) -> None:
        """Test logging message with tool call."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.model_config = MagicMock()
        tracer.session.model_config.get_model_id.return_value = "test-model"

        message = {
            "info": {
                "id": "msg-1",
                "role": "assistant",
                "time": {"created": 1000, "completed": 2000},
                "cost": 0.01,
                "tokens": {"input": 10, "output": 20},
            },
            "parts": [
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "call-1",
                    "state": {
                        "input": {"command": "ls"},
                        "output": "file.txt",
                        "status": "success",
                        "time": {"start": 1000, "end": 1500},
                    },
                }
            ],
        }

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_parent_span = MagicMock()
            mock_child_span = MagicMock()
            mock_mlflow.start_span_no_context.side_effect = [
                mock_parent_span,
                mock_child_span,
            ]

            tracer.log_message(message)

            # Should create parent span and child span for tool
            expected_span_count = 2
            assert mock_mlflow.start_span_no_context.call_count == expected_span_count

    def test_log_message_with_tool_error(self) -> None:
        """Test logging message with tool error."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.model_config = MagicMock()
        tracer.session.model_config.get_model_id.return_value = "test-model"

        message = {
            "info": {
                "id": "msg-1",
                "role": "assistant",
                "time": {"created": 1000, "completed": 2000},
                "cost": 0.01,
                "tokens": {"input": 10, "output": 20},
            },
            "parts": [
                {
                    "type": "tool",
                    "tool": "bash",
                    "callID": "call-1",
                    "state": {
                        "input": {"command": "invalid"},
                        "output": "error message",
                        "status": "error",
                        "error": "Command failed",
                        "time": {"start": 1000, "end": 1500},
                    },
                }
            ],
        }

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_parent_span = MagicMock()
            mock_child_span = MagicMock()
            mock_mlflow.start_span_no_context.side_effect = [
                mock_parent_span,
                mock_child_span,
            ]

            tracer.log_message(message)

            # Parent span should have ERROR status
            mock_parent_span.set_status.assert_called_with("ERROR")


class TestExtractTimestamps:
    """Tests for _extract_timestamps_ns method."""

    def test_extract_with_valid_timestamps(self) -> None:
        """Test extraction with valid timestamps."""
        tracer = MlflowTracer()
        time_info = {"created": 1000, "completed": 2000}

        start_ns, end_ns = tracer._extract_timestamps_ns(
            time_info, "created", "completed"
        )

        assert start_ns == 1000 * 1_000_000  # Converted to nanoseconds
        assert end_ns == 2000 * 1_000_000

    def test_extract_with_missing_timestamps(self) -> None:
        """Test extraction with missing timestamps and fallbacks."""
        tracer = MlflowTracer()
        time_info = {}

        start_ns, end_ns = tracer._extract_timestamps_ns(
            time_info,
            "created",
            "completed",
            fallback_start_ns=100,
            fallback_end_ns=200,
        )

        fallback_start = 100
        fallback_end = 200
        assert start_ns == fallback_start
        assert end_ns == fallback_end

    def test_extract_with_none_timestamps(self) -> None:
        """Test extraction when timestamps are None."""
        tracer = MlflowTracer()
        time_info = {"created": None, "completed": None}

        start_ns, end_ns = tracer._extract_timestamps_ns(
            time_info,
            "created",
            "completed",
            fallback_start_ns=100,
            fallback_end_ns=200,
        )

        fallback_start = 100
        fallback_end = 200
        assert start_ns == fallback_start
        assert end_ns == fallback_end


class TestHandleToolErrors:
    """Tests for _handle_tool_errors method."""

    def test_explicit_error_status(self) -> None:
        """Test handling explicit error status."""
        tracer = MlflowTracer()
        mock_span = MagicMock()
        state = {"status": "error", "error": "Something went wrong"}

        tracer._handle_tool_errors(mock_span, state, "bash", "call-1", "output")

        mock_span.set_status.assert_called_with("ERROR")
        mock_span.add_event.assert_called_once()

    def test_nonzero_exit_code(self) -> None:
        """Test handling nonzero exit code."""
        tracer = MlflowTracer()
        mock_span = MagicMock()
        state = {"status": "success", "metadata": {"exit": 1}}

        tracer._handle_tool_errors(mock_span, state, "bash", "call-1", "error output")

        mock_span.set_status.assert_called_with("ERROR")
        mock_span.add_event.assert_called_once()

    def test_no_error(self) -> None:
        """Test when there's no error."""
        tracer = MlflowTracer()
        mock_span = MagicMock()
        state = {"status": "success", "metadata": {"exit": 0}}

        tracer._handle_tool_errors(mock_span, state, "bash", "call-1", "output")

        mock_span.set_status.assert_not_called()
        mock_span.add_event.assert_not_called()


class TestAccumulateToolCall:
    """Tests for _accumulate_tool_call method."""

    def test_accumulate_new_tool(self) -> None:
        """Test accumulating call for new tool."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.tool_call_counts = {}

        tracer._accumulate_tool_call("bash")

        assert tracer.session.tool_call_counts["bash"] == 1

    def test_accumulate_existing_tool(self) -> None:
        """Test accumulating call for existing tool."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.tool_call_counts = {"bash": 2}

        tracer._accumulate_tool_call("bash")

        expected_count = 3
        assert tracer.session.tool_call_counts["bash"] == expected_count

    def test_accumulate_no_session(self) -> None:
        """Test accumulating when no session exists."""
        tracer = MlflowTracer()
        tracer.session = None

        # Should not raise
        tracer._accumulate_tool_call("bash")


class TestLogGitDiff:
    """Tests for log_git_diff method."""

    def test_log_git_diff_success(self) -> None:
        """Test logging git diff."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()

        tracer.log_git_diff("diff content")

        assert tracer.session.git_diff == "diff content"

    def test_log_git_diff_no_session(self) -> None:
        """Test logging git diff without session."""
        tracer = MlflowTracer()
        tracer.session = None

        with patch("beyond_vibes.simulations.mlflow.logger") as mock_logger:
            tracer.log_git_diff("diff content")
            mock_logger.warning.assert_called_once()


class TestLogError:
    """Tests for log_error method."""

    def test_log_error_success(self) -> None:
        """Test logging error."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()

        tracer.log_error("Something went wrong")

        assert tracer.session.error == "Something went wrong"

    def test_log_error_no_session(self) -> None:
        """Test logging error without session."""
        tracer = MlflowTracer()
        tracer.session = None

        with patch("beyond_vibes.simulations.mlflow.logger") as mock_logger:
            tracer.log_error("Something went wrong")
            mock_logger.warning.assert_called_once()


class TestFlush:
    """Tests for _flush method."""

    def test_flush_no_run_id(self) -> None:
        """Test flush when no run_id exists."""
        tracer = MlflowTracer()
        tracer.run_id = None
        tracer.session = MagicMock()

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()
            mock_mlflow.log_metric.assert_not_called()

    def test_flush_no_session(self) -> None:
        """Test flush when no session exists."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = None

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()
            mock_mlflow.log_metric.assert_not_called()

    def test_flush_success(self) -> None:
        """Test successful flush."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = [MagicMock(), MagicMock()]
        tracer.session.total_cost = 0.05
        tracer.session.total_input_tokens = 100
        tracer.session.total_output_tokens = 200
        tracer.session.total_tokens = 300
        tracer.session.tool_call_counts = {"bash": 3, "read": 2}
        tracer.session.error = None
        tracer.session.git_diff = None

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            # Verify metrics are logged
            mock_mlflow.log_metric.assert_any_call("total_messages", 2)
            mock_mlflow.log_metric.assert_any_call("total_cost", 0.05)
            mock_mlflow.log_metric.assert_any_call("total_input_tokens", 100)
            mock_mlflow.log_metric.assert_any_call("total_output_tokens", 200)
            mock_mlflow.log_metric.assert_any_call("total_tokens", 300)
            mock_mlflow.log_metric.assert_any_call("tool_calls.bash", 3)
            mock_mlflow.log_metric.assert_any_call("tool_calls.read", 2)
            mock_mlflow.log_metric.assert_any_call("total_tool_calls", 5)

    def test_flush_with_error(self) -> None:
        """Test flush when session has error."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = []
        tracer.session.tool_call_counts = {}
        tracer.session.error = "Something went wrong"
        tracer.session.git_diff = None

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            mock_mlflow.log_metric.assert_any_call("has_error", 1)

    def test_flush_with_git_diff(self) -> None:
        """Test flush when session has git diff."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = []
        tracer.session.tool_call_counts = {}
        tracer.session.error = None
        tracer.session.git_diff = "diff content"

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            mock_mlflow.log_text.assert_called_once_with(
                "diff content", "git_diff.patch"
            )
