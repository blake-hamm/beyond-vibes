"""Tests for simulation orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beyond_vibes.model_config import ModelConfig
from beyond_vibes.simulations.models import RepositoryConfig, SimulationConfig
from beyond_vibes.simulations.orchestration import (
    SimulationOrchestrator,
    run_simulation,
)


@pytest.fixture
def mock_simulation_config() -> SimulationConfig:
    """Create a mock simulation config."""
    return SimulationConfig(
        name="test-task",
        description="Test task",
        archetype="test",
        repository=RepositoryConfig(url="https://github.com/test/repo", branch="main"),
        prompt="Test prompt",
        agent="orchestrator",
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


class TestSimulationOrchestratorInit:
    """Tests for SimulationOrchestrator initialization."""

    def test_init(self) -> None:
        """Test orchestrator initialization."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        assert orchestrator.opencode == mock_opencode
        assert orchestrator.tracer == mock_tracer
        assert orchestrator.sandbox == mock_sandbox
        assert orchestrator._seen_message_ids == set()
        assert orchestrator._assistant_message_count == 0
        assert orchestrator._session_id is None


class TestSimulationOrchestratorRun:
    """Tests for SimulationOrchestrator.run method."""

    def test_run_success(self) -> None:
        """Test successful simulation run."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        # Mock sandbox context manager
        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        # Mock opencode methods
        mock_opencode.create_session.return_value = "session-123"
        mock_opencode.send_prompt.return_value = None

        # Mock messages - one complete assistant message with stop signal
        mock_messages = [
            {
                "info": {
                    "id": "msg-1",
                    "role": "assistant",
                    "time": {"completed": 1234567890},
                    "finish": "stop",
                }
            }
        ]
        mock_opencode.get_messages.return_value = mock_messages
        mock_opencode.abort_session.return_value = True

        # Collect yielded messages
        messages = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                model_id="test-model",
                provider="local",
                agent="orchestrator",
                max_turns=5,
            )
        )

        assert len(messages) == 1
        assert messages[0]["info"]["id"] == "msg-1"
        mock_opencode.create_session.assert_called_once_with(Path("/tmp/test"))
        mock_opencode.abort_session.assert_called_once_with("session-123")

    def test_run_sandbox_failure(self) -> None:
        """Test run when sandbox creation fails."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        # Mock sandbox returning None
        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(return_value=None)
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(RuntimeError, match="Failed to create sandbox"):
            list(
                orchestrator.run(
                    repo_url="https://github.com/test/repo",
                    branch="main",
                    prompt="Test prompt",
                    model_id="test-model",
                    provider="local",
                    agent="orchestrator",
                )
            )

    def test_run_message_deduplication(self) -> None:
        """Test that messages are deduplicated."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_opencode.create_session.return_value = "session-123"

        # Same message returned twice
        mock_message = {
            "info": {
                "id": "msg-1",
                "role": "assistant",
                "time": {"completed": 1234567890},
                "finish": "stop",
            }
        }
        mock_opencode.get_messages.side_effect = [[mock_message], [mock_message]]
        mock_opencode.abort_session.return_value = True

        messages = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                model_id="test-model",
                provider="local",
                agent="orchestrator",
                max_turns=5,
            )
        )

        # Should only yield the message once
        assert len(messages) == 1

    def test_run_max_turns_reached(self) -> None:
        """Test that run stops when max_turns is reached."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_opencode.create_session.return_value = "session-123"

        # Create messages up to max_turns (with content so they count)
        mock_messages = [
            {
                "info": {
                    "id": f"msg-{i}",
                    "role": "assistant",
                    "time": {"completed": 1234567890},
                },
                "parts": [{"type": "text", "text": f"Response {i}"}],
            }
            for i in range(5)
        ]
        mock_opencode.get_messages.side_effect = [
            mock_messages[: i + 1] for i in range(5)
        ]
        mock_opencode.abort_session.return_value = True

        messages = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                model_id="test-model",
                provider="local",
                agent="orchestrator",
                max_turns=3,
            )
        )

        # Should stop at max_turns (3 assistant messages)
        expected_message_count = 3
        assert len(messages) == expected_message_count
        mock_opencode.abort_session.assert_called_once()

    def test_run_incomplete_messages_filtered(self) -> None:
        """Test that incomplete messages (no completed timestamp) are filtered."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_opencode.create_session.return_value = "session-123"

        # Mix of complete and incomplete messages
        mock_messages = [
            {
                "info": {
                    "id": "msg-1",
                    "role": "assistant",
                    "time": {"completed": 1234567890},
                    "finish": "stop",
                }
            },
            {
                "info": {
                    "id": "msg-2",
                    "role": "assistant",
                    "time": {},  # No completed timestamp
                }
            },
        ]
        mock_opencode.get_messages.return_value = mock_messages
        mock_opencode.abort_session.return_value = True

        messages = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                model_id="test-model",
                provider="local",
                agent="orchestrator",
            )
        )

        # Should only yield the complete message
        assert len(messages) == 1
        assert messages[0]["info"]["id"] == "msg-1"

    def test_run_exception_abort_session(self) -> None:
        """Test that session is aborted when exception occurs."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_opencode.create_session.return_value = "session-123"
        mock_opencode.get_messages.side_effect = Exception("Connection error")
        mock_opencode.abort_session.return_value = True

        with pytest.raises(Exception, match="Connection error"):
            list(
                orchestrator.run(
                    repo_url="https://github.com/test/repo",
                    branch="main",
                    prompt="Test prompt",
                    model_id="test-model",
                    provider="local",
                    agent="orchestrator",
                )
            )

        mock_opencode.abort_session.assert_called_once_with("session-123")

    def test_run_stop_with_content_continues(self) -> None:
        """Test that finish=stop with content does not abort the session."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_opencode.create_session.return_value = "session-123"

        # Assistant message with finish=stop but has text content
        content_message = {
            "info": {
                "id": "msg-1",
                "role": "assistant",
                "time": {"completed": 1234567890},
                "finish": "stop",
            },
            "parts": [{"type": "text", "text": "I will now use a tool"}],
        }
        # Followed by an empty stop message (orchestrator loop)
        empty_message = {
            "info": {
                "id": "msg-2",
                "role": "assistant",
                "time": {"completed": 1234567891},
                "finish": "stop",
            },
        }
        mock_opencode.get_messages.side_effect = [
            [content_message],
            [content_message, empty_message],
        ]
        mock_opencode.abort_session.return_value = True

        messages = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                model_id="test-model",
                provider="local",
                agent="orchestrator",
                max_turns=5,
            )
        )

        # Should yield both messages and abort on the empty one
        expected_messages = 2
        assert len(messages) == expected_messages
        mock_opencode.abort_session.assert_called_once_with("session-123")

    def test_run_no_stop_signal(self) -> None:
        """Test run without stop signal continues until max_turns."""
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_opencode, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_opencode.create_session.return_value = "session-123"

        # Messages without stop signal (with content so they count toward max_turns)
        mock_messages = [
            {
                "info": {
                    "id": "msg-1",
                    "role": "assistant",
                    "time": {"completed": 1234567890},
                },
                "parts": [{"type": "text", "text": "Working on it"}],
            }
        ]
        mock_opencode.get_messages.side_effect = [mock_messages, mock_messages]
        mock_opencode.abort_session.return_value = True

        # Use max_turns=1 to limit the test
        messages = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                model_id="test-model",
                provider="local",
                agent="orchestrator",
                max_turns=1,
            )
        )

        assert len(messages) == 1
        mock_opencode.abort_session.assert_called_once()


class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_run_simulation_success(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test successful simulation execution."""
        mock_sandbox = MagicMock()
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()

        # Mock tracer context manager
        mock_tracer.log_simulation.return_value.__enter__ = MagicMock(
            return_value=mock_tracer
        )
        mock_tracer.log_simulation.return_value.__exit__ = MagicMock(return_value=False)

        # Mock orchestrator
        with patch(
            "beyond_vibes.simulations.orchestration.SimulationOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.run.return_value = iter([])

            result = run_simulation(
                sim_config=mock_simulation_config,
                model_config=mock_model_config,
                sandbox=mock_sandbox,
                opencode_client=mock_opencode,
                tracer=mock_tracer,
                prompt="Test prompt",
            )

        assert result is False  # No error occurred
        mock_tracer.log_simulation.assert_called_once_with(
            mock_simulation_config, mock_model_config
        )

    def test_run_simulation_with_messages(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test simulation that yields messages."""
        mock_sandbox = MagicMock()
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()

        mock_tracer.log_simulation.return_value.__enter__ = MagicMock(
            return_value=mock_tracer
        )
        mock_tracer.log_simulation.return_value.__exit__ = MagicMock(return_value=False)

        test_messages = [
            {"info": {"id": "msg-1"}},
            {"info": {"id": "msg-2"}},
        ]

        with patch(
            "beyond_vibes.simulations.orchestration.SimulationOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.run.return_value = iter(test_messages)

            result = run_simulation(
                sim_config=mock_simulation_config,
                model_config=mock_model_config,
                sandbox=mock_sandbox,
                opencode_client=mock_opencode,
                tracer=mock_tracer,
                prompt="Test prompt",
            )

        assert result is False
        # Verify messages were logged
        expected_logged_messages = 2
        assert mock_tracer.log_message.call_count == expected_logged_messages

    def test_run_simulation_error(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test simulation that encounters an error."""
        mock_sandbox = MagicMock()
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.session = MagicMock()

        mock_tracer.log_simulation.return_value.__enter__ = MagicMock(
            return_value=mock_tracer
        )
        mock_tracer.log_simulation.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "beyond_vibes.simulations.orchestration.SimulationOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.run.side_effect = Exception("Simulation failed")

            result = run_simulation(
                sim_config=mock_simulation_config,
                model_config=mock_model_config,
                sandbox=mock_sandbox,
                opencode_client=mock_opencode,
                tracer=mock_tracer,
                prompt="Test prompt",
            )

        assert result is True  # Error occurred
        mock_tracer.log_error.assert_called_once_with("Simulation failed")

    def test_run_simulation_error_no_session(
        self, mock_simulation_config: SimulationConfig, mock_model_config: ModelConfig
    ) -> None:
        """Test simulation error when tracer has no session."""
        mock_sandbox = MagicMock()
        mock_opencode = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.session = None

        mock_tracer.log_simulation.return_value.__enter__ = MagicMock(
            return_value=mock_tracer
        )
        mock_tracer.log_simulation.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "beyond_vibes.simulations.orchestration.SimulationOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.run.side_effect = Exception("Simulation failed")

            result = run_simulation(
                sim_config=mock_simulation_config,
                model_config=mock_model_config,
                sandbox=mock_sandbox,
                opencode_client=mock_opencode,
                tracer=mock_tracer,
                prompt="Test prompt",
            )

        assert result is True
        # log_error should not be called when session is None
        mock_tracer.log_error.assert_not_called()
