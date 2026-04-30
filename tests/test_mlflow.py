"""Tests for MLflow tracer."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.mlflow import (
    MessagePerformanceMetrics,
    MlflowTracer,
    SimulationSession,
    generate_session_id,
)
from beyond_vibes.simulations.models import RepositoryConfig, SimulationConfig
from beyond_vibes.simulations.pi_dev import TurnData


@pytest.fixture
def mock_simulation_config() -> SimulationConfig:
    """Create a mock simulation config."""
    return SimulationConfig(
        name="test-task",
        description="Test task",
        archetype="test",
        repository=RepositoryConfig(url="https://github.com/test/repo", branch="main"),
        prompt="Test prompt",
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
            llm_config=mock_model_config,
        )

        assert session.sim_config == mock_simulation_config
        assert session.llm_config == mock_model_config
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


class TestLogTurn:
    """Tests for log_turn method."""

    @pytest.fixture
    def mock_turn(self) -> TurnData:
        """Create a mock TurnData with text content."""
        return TurnData(
            turn_index=0,
            content=[{"type": "text", "text": "Hello world"}],
            usage={
                "input": 10,
                "output": 5,
                "totalTokens": 15,
                "cost": {"total": 0.001},
            },
            stop_reason="stop",
            raw_message={"role": "assistant", "responseId": "msg_abc"},
        )

    def test_log_turn_no_session(self) -> None:
        """Test logging turn without active session."""
        tracer = MlflowTracer()

        with patch("beyond_vibes.simulations.mlflow.logger") as mock_logger:
            tracer.log_turn(TurnData(turn_index=0))
            mock_logger.warning.assert_called_once()

    def test_log_turn_with_text_content(self, mock_turn: TurnData) -> None:
        """Test logging turn with text content."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.run_id = "run-123"

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span

            tracer.log_turn(mock_turn)

            mock_mlflow.start_span_no_context.assert_called_once()
            content = mock_span.set_outputs.call_args[0][0]["content"][0]
            assert content["type"] == "text"
            assert content["content"] == "Hello world"
            mock_span.end.assert_called_once()

    def test_log_turn_with_thinking_content(self) -> None:
        """Test logging turn with thinking content."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.run_id = "run-123"

        turn = TurnData(
            turn_index=0,
            content=[{"type": "thinking", "thinking": "Hmm..."}],
            usage={},
            stop_reason="stop",
        )

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span

            tracer.log_turn(turn)

            content = mock_span.set_outputs.call_args[0][0]["content"][0]
            assert content["type"] == "thinking"
            assert content["content"] == "Hmm..."

    def test_log_turn_with_usage(self) -> None:
        """Test logging turn captures usage attributes."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.run_id = "run-123"

        turn = TurnData(
            turn_index=0,
            content=[{"type": "text", "text": "Hi"}],
            usage={
                "input": 100,
                "output": 50,
                "totalTokens": 150,
                "cacheRead": 10,
                "cacheWrite": 20,
                "cost": {"total": 0.005},
            },
            stop_reason="toolUse",
            raw_message={"responseId": "msg_xyz"},
        )

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span

            tracer.log_turn(turn)

            all_attrs = {}
            for call in mock_span.set_attributes.call_args_list:
                all_attrs.update(call[0][0])
            assert all_attrs["llm.token_usage.input_tokens"] == 100  # noqa: PLR2004
            assert all_attrs["llm.token_usage.output_tokens"] == 50  # noqa: PLR2004
            assert all_attrs["llm.token_usage.total_tokens"] == 150  # noqa: PLR2004
            assert all_attrs["llm.token_usage.cache_read_tokens"] == 10  # noqa: PLR2004
            assert all_attrs["llm.token_usage.cache_write_tokens"] == 20  # noqa: PLR2004
            assert all_attrs["cost"] == 0.005  # noqa: PLR2004
            assert all_attrs["stop_reason"] == "toolUse"
            assert all_attrs["response_id"] == "msg_xyz"

    def test_log_turn_with_tool_calls(self) -> None:
        """Test logging turn creates child spans for tools."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.session.tool_call_counts = {}
        tracer.session.tool_last_name = None
        tracer.session.tool_consecutive_calls = 0
        tracer.session.tool_max_consecutive_calls = 0
        tracer.run_id = "run-123"

        turn = TurnData(
            turn_index=0,
            content=[{"type": "text", "text": "Result"}],
            usage={},
            tool_calls=[
                {"toolCallId": "tool_1", "toolName": "bash", "args": {"command": "ls"}}
            ],
            tool_results=[
                {
                    "toolCallId": "tool_1",
                    "toolName": "bash",
                    "result": {"stdout": "file.txt"},
                    "isError": False,
                }
            ],
        )

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_parent = MagicMock()
            mock_child = MagicMock()
            mock_mlflow.start_span_no_context.side_effect = [mock_parent, mock_child]

            tracer.log_turn(turn)

            assert mock_mlflow.start_span_no_context.call_count == 2  # noqa: PLR2004
            child_call = mock_mlflow.start_span_no_context.call_args_list[1]
            assert child_call[1]["span_type"] == "TOOL"
            assert child_call[1]["name"] == "tool:bash:tool_1"
            mock_child.set_outputs.assert_called_once_with(
                {"output": {"stdout": "file.txt"}}
            )

    def test_log_turn_with_tool_error(self) -> None:
        """Test logging turn marks tool span as ERROR."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.session.tool_error_count = 0
        tracer.session.error_message_indices = []
        tracer.session.tool_call_counts = {}
        tracer.session.tool_last_name = None
        tracer.session.tool_consecutive_calls = 0
        tracer.session.tool_max_consecutive_calls = 0
        tracer.run_id = "run-123"

        turn = TurnData(
            turn_index=0,
            content=[{"type": "text", "text": "Error"}],
            usage={},
            tool_calls=[
                {"toolCallId": "tool_1", "toolName": "bash", "args": {"command": "bad"}}
            ],
            tool_results=[
                {
                    "toolCallId": "tool_1",
                    "toolName": "bash",
                    "result": "Command not found",
                    "isError": True,
                }
            ],
        )

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_parent = MagicMock()
            mock_child = MagicMock()
            mock_mlflow.start_span_no_context.side_effect = [mock_parent, mock_child]

            tracer.log_turn(turn)

            mock_child.set_status.assert_called_once_with("ERROR")
            mock_child.add_event.assert_called_once()
            assert tracer.session.tool_error_count == 1

    def test_log_turn_accumulates_session_totals(self) -> None:
        """Test that log_turn accumulates cost and token totals."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.session.total_cost = 0.0
        tracer.session.total_input_tokens = 0
        tracer.session.total_output_tokens = 0
        tracer.session.total_tokens = 0
        tracer.run_id = "run-123"

        turn = TurnData(
            turn_index=0,
            content=[{"type": "text", "text": "Hi"}],
            usage={
                "input": 10,
                "output": 5,
                "totalTokens": 15,
                "cost": {"total": 0.001},
            },
        )

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span
            tracer.log_turn(turn)

        assert tracer.session.total_cost == 0.001  # noqa: PLR2004
        assert tracer.session.total_input_tokens == 10  # noqa: PLR2004
        assert tracer.session.total_output_tokens == 5  # noqa: PLR2004
        assert tracer.session.total_tokens == 15  # noqa: PLR2004

    def test_log_turn_appends_message_data(self) -> None:
        """Test that log_turn appends MessageData to session."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.run_id = "run-123"

        turn = TurnData(
            turn_index=3,
            content=[{"type": "text", "text": "Hi"}],
            usage={},
            raw_message={"role": "assistant"},
        )

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span
            tracer.log_turn(turn)

        assert len(tracer.session.messages) == 1  # noqa: PLR2004
        assert tracer.session.messages[0].message_index == 3  # noqa: PLR2004
        assert tracer.session.messages[0].raw_message == {"role": "assistant"}

    def test_log_turn_sets_perf_attributes(self, mock_turn: TurnData) -> None:
        """Test that log_turn sets performance span attributes."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.session.message_metrics = []
        tracer.run_id = "run-123"

        mock_turn.ttft_ms = 100.0
        mock_turn.generation_time_ms = 200.0
        mock_turn.generation_tps = 50.0
        mock_turn.prompt_tps = 200.0
        mock_turn.prompt_processing_ms = 50.0
        mock_turn.tool_calls = []

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span
            tracer.log_turn(mock_turn)

        perf_call = None
        for call in mock_span.set_attributes.call_args_list:
            attrs = call[0][0]
            if "perf.ttft_ms" in attrs:
                perf_call = attrs
                break

        assert perf_call is not None
        assert perf_call["perf.ttft_ms"] == 100.0  # noqa: PLR2004
        assert perf_call["perf.generation_time_ms"] == 200.0  # noqa: PLR2004
        assert perf_call["perf.generation_tps"] == 50.0  # noqa: PLR2004
        assert perf_call["perf.prompt_tps"] == 200.0  # noqa: PLR2004
        assert perf_call["perf.has_tool_calls"] is False

        assert len(tracer.session.message_metrics) == 1  # noqa: PLR2004
        assert tracer.session.message_metrics[0].ttft_ms == 100.0  # noqa: PLR2004

    def test_log_turn_with_tool_calls_sets_has_tool_calls(self) -> None:
        """Test perf metric reflects tool calls."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.session.message_metrics = []
        tracer.run_id = "run-123"

        turn = TurnData(
            turn_index=0,
            content=[{"type": "text", "text": "Result"}],
            usage={"input": 10, "output": 5},
            tool_calls=[{"toolCallId": "tool_1", "toolName": "bash", "args": {}}],
            tool_results=[],
        )
        turn.ttft_ms = 50.0
        turn.generation_tps = 10.0

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span
            tracer.log_turn(turn)

        assert tracer.session.message_metrics[0].has_tool_calls is True

    def test_log_turn_without_latency_metrics(self, mock_turn: TurnData) -> None:
        """Test log_turn handles TurnData without latency metrics."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.session_id = "session-123"
        tracer.session.messages = []
        tracer.session.llm_config.get_model_id.return_value = "test-model"
        tracer.session.llm_config.provider = "local"
        tracer.session.message_metrics = []
        tracer.run_id = "run-123"

        mock_turn.ttft_ms = None
        mock_turn.generation_time_ms = None
        mock_turn.generation_tps = None
        mock_turn.prompt_tps = None
        mock_turn.prompt_processing_ms = None
        mock_turn.tool_calls = []

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            mock_span = MagicMock()
            mock_mlflow.start_span_no_context.return_value = mock_span
            tracer.log_turn(mock_turn)

        assert len(tracer.session.message_metrics) == 1  # noqa: PLR2004
        assert tracer.session.message_metrics[0].ttft_ms is None
        assert tracer.session.message_metrics[0].generation_tps is None
        assert tracer.session.message_metrics[0].prompt_tps is None


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

        tracer._handle_tool_errors(mock_span, state, "bash", "call-1", "output", 0)

        mock_span.set_status.assert_called_with("ERROR")
        mock_span.add_event.assert_called_once()

    def test_nonzero_exit_code(self) -> None:
        """Test handling nonzero exit code."""
        tracer = MlflowTracer()
        mock_span = MagicMock()
        state = {"status": "success", "metadata": {"exit": 1}}

        tracer._handle_tool_errors(
            mock_span, state, "bash", "call-1", "error output", 0
        )

        mock_span.set_status.assert_called_with("ERROR")
        mock_span.add_event.assert_called_once()

    def test_no_error(self) -> None:
        """Test when there's no error."""
        tracer = MlflowTracer()
        mock_span = MagicMock()
        state = {"status": "success", "metadata": {"exit": 0}}

        tracer._handle_tool_errors(mock_span, state, "bash", "call-1", "output", 0)

        mock_span.set_status.assert_not_called()
        mock_span.add_event.assert_not_called()


class TestAccumulateToolCall:
    """Tests for _accumulate_tool_call method."""

    def test_accumulate_new_tool(self) -> None:
        """Test accumulating call for new tool."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.tool_call_counts = {}
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.tool_last_name = None
        tracer.session.tool_consecutive_calls = 0
        tracer.session.tool_max_consecutive_calls = 0

        tracer._accumulate_tool_call("bash", 0)

        assert tracer.session.tool_call_counts["bash"] == 1

    def test_accumulate_existing_tool(self) -> None:
        """Test accumulating call for existing tool."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()
        tracer.session.tool_call_counts = {"bash": 2}
        tracer.session.tool_last_name = None
        tracer.session.tool_consecutive_calls = 0
        tracer.session.tool_max_consecutive_calls = 0

        tracer._accumulate_tool_call("bash", 0)

        expected_count = 3
        assert tracer.session.tool_call_counts["bash"] == expected_count

    def test_accumulate_no_session(self) -> None:
        """Test accumulating when no session exists."""
        tracer = MlflowTracer()
        tracer.session = None

        # Should not raise
        tracer._accumulate_tool_call("bash", 0)


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


class TestLogSystemPrompt:
    """Tests for log_system_prompt method."""

    def test_log_system_prompt_success(self) -> None:
        """Test logging system prompt."""
        tracer = MlflowTracer()
        tracer.session = MagicMock()

        tracer.log_system_prompt("System prompt content")

        assert tracer.session.system_prompt == "System prompt content"

    def test_log_system_prompt_no_session(self) -> None:
        """Test logging system prompt without session."""
        tracer = MlflowTracer()
        tracer.session = None

        with patch("beyond_vibes.simulations.mlflow.logger") as mock_logger:
            tracer.log_system_prompt("System prompt content")
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
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 1
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.error = None
        tracer.session.git_diff = None
        tracer.session.system_prompt = None

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
            mock_mlflow.log_metric.assert_any_call("tool_total_calls", 5)

    def test_flush_with_error(self) -> None:
        """Test flush when session has error."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = []
        tracer.session.tool_call_counts = {}
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.error = "Something went wrong"
        tracer.session.git_diff = None
        tracer.session.system_prompt = None

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
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.error = None
        tracer.session.git_diff = "diff content"
        tracer.session.system_prompt = None

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            mock_mlflow.log_text.assert_called_once_with(
                "diff content", "git_diff.patch"
            )

    def test_flush_with_system_prompt(self) -> None:
        """Test flush when session has system prompt."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = []
        tracer.session.tool_call_counts = {}
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.error = None
        tracer.session.git_diff = None
        tracer.session.system_prompt = "System prompt content"

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            mock_mlflow.log_text.assert_called_once_with(
                "System prompt content", "system_prompt.txt"
            )

    def test_flush_with_git_diff_and_system_prompt(self) -> None:
        """Test flush when session has both git diff and system prompt."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = []
        tracer.session.tool_call_counts = {}
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.error = None
        tracer.session.git_diff = "diff content"
        tracer.session.system_prompt = "System prompt content"

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            mock_mlflow.log_text.assert_any_call("diff content", "git_diff.patch")
            mock_mlflow.log_text.assert_any_call(
                "System prompt content", "system_prompt.txt"
            )
            assert mock_mlflow.log_text.call_count == 2  # noqa: PLR2004

    def test_flush_with_latency_metrics(self) -> None:
        """Test flush aggregates and logs latency metrics."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = [MagicMock(), MagicMock()]
        tracer.session.total_cost = 0.0
        tracer.session.total_input_tokens = 0
        tracer.session.total_output_tokens = 0
        tracer.session.total_tokens = 0
        tracer.session.tool_call_counts = {}
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.error = None
        tracer.session.git_diff = None
        tracer.session.system_prompt = None
        tracer.session.message_metrics = [
            MessagePerformanceMetrics(
                ttft_ms=100.0,
                generation_time_ms=200.0,
                generation_tps=50.0,
                prompt_tps=200.0,
                prompt_processing_ms=50.0,
                output_tokens=10,
                has_tool_calls=False,
            ),
            MessagePerformanceMetrics(
                ttft_ms=200.0,
                generation_time_ms=400.0,
                generation_tps=25.0,
                prompt_tps=100.0,
                prompt_processing_ms=100.0,
                output_tokens=10,
                has_tool_calls=False,
            ),
        ]

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            mock_mlflow.log_metric.assert_any_call("avg_ttft_ms", 150.0)
            mock_mlflow.log_metric.assert_any_call("avg_prompt_tps", 150.0)
            mock_mlflow.log_metric.assert_any_call("avg_generation_tps", 37.5)
            mock_mlflow.log_metric.assert_any_call("total_generation_time_ms", 600.0)
            mock_mlflow.log_metric.assert_any_call("total_prompt_processing_ms", 150.0)

    def test_flush_without_latency_metrics(self) -> None:
        """Test flush does not log perf metrics when none exist."""
        tracer = MlflowTracer()
        tracer.run_id = "run-123"
        tracer.session = MagicMock()
        tracer.session.messages = []
        tracer.session.total_cost = 0.0
        tracer.session.total_input_tokens = 0
        tracer.session.total_output_tokens = 0
        tracer.session.total_tokens = 0
        tracer.session.tool_call_counts = {}
        tracer.session.tool_error_count = 0
        tracer.session.tool_loop_threshold = 3
        tracer.session.tool_max_consecutive_calls = 0
        tracer.session.error_message_indices = []
        tracer.session.started_at = datetime.now()
        tracer.session.error = None
        tracer.session.git_diff = None
        tracer.session.system_prompt = None
        tracer.session.message_metrics = []
        # Prevent MagicMock auto-creation from making these truthy
        tracer.session.avg_ttft_ms = None
        tracer.session.avg_prompt_tps = None
        tracer.session.avg_generation_tps = None
        tracer.session.total_generation_time_ms = None
        tracer.session.total_prompt_processing_ms = None

        with patch("beyond_vibes.simulations.mlflow.mlflow") as mock_mlflow:
            tracer._flush()

            for call in mock_mlflow.log_metric.call_args_list:
                assert call[0][0] not in {
                    "avg_ttft_ms",
                    "avg_prompt_tps",
                    "avg_generation_tps",
                    "total_generation_time_ms",
                    "total_prompt_processing_ms",
                }
