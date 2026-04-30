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


class TestSimulationOrchestratorInit:
    """Tests for SimulationOrchestrator initialization."""

    def test_init(self: "TestSimulationOrchestratorInit") -> None:
        """Test orchestrator initialization."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)

        assert orchestrator.pi == mock_pi
        assert orchestrator.tracer == mock_tracer
        assert orchestrator.sandbox == mock_sandbox
        assert orchestrator.completion_status is None


class TestSimulationOrchestratorRun:
    """Tests for SimulationOrchestrator.run method."""

    def test_run_success(self: "TestSimulationOrchestratorRun") -> None:
        """Test successful simulation run yielding TurnData."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_turn = TurnData(turn_index=0, content=[{"type": "text", "text": "Hi"}])
        mock_pi.run.return_value = iter([mock_turn])
        mock_pi.max_turns_reached = False

        turns = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                max_turns=5,
            )
        )

        assert len(turns) == 1
        assert turns[0].turn_index == 0
        assert orchestrator.completion_status == "completed"
        mock_pi.run.assert_called_once()

    def test_run_sandbox_failure(self: "TestSimulationOrchestratorRun") -> None:
        """Test run when sandbox creation fails."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(return_value=None)
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(RuntimeError, match="Failed to create sandbox"):
            list(
                orchestrator.run(
                    repo_url="https://github.com/test/repo",
                    branch="main",
                    prompt="Test prompt",
                )
            )

    def test_run_max_turns_reached(self: "TestSimulationOrchestratorRun") -> None:
        """Test that completion_status is max_turns when pi hits limit."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_turns = [
            TurnData(turn_index=i, content=[{"type": "text", "text": f"T{i}"}])
            for i in range(3)
        ]
        mock_pi.run.return_value = iter(mock_turns)
        mock_pi.max_turns_reached = True

        turns = list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                max_turns=3,
            )
        )

        expected_turn_count = 3
        assert len(turns) == expected_turn_count
        assert orchestrator.completion_status == "max_turns"

    def test_run_exception(self: "TestSimulationOrchestratorRun") -> None:
        """Test that exception sets completion_status to error."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_pi.run.side_effect = Exception("Connection error")

        with pytest.raises(Exception, match="Connection error"):
            list(
                orchestrator.run(
                    repo_url="https://github.com/test/repo",
                    branch="main",
                    prompt="Test prompt",
                )
            )

        assert orchestrator.completion_status == "error"

    def test_run_git_diff_capture(self: "TestSimulationOrchestratorRun") -> None:
        """Test that git diff is captured in finally block."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)
        mock_sandbox.get_git_diff.return_value = "diff --git a/file.txt"

        mock_pi.run.return_value = iter([])
        mock_pi.max_turns_reached = False

        list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                capture_git_diff=True,
            )
        )

        mock_tracer.log_git_diff.assert_called_once_with("diff --git a/file.txt")

    def test_run_system_prompt_passed(self: "TestSimulationOrchestratorRun") -> None:
        """Test that system_prompt is passed to pi client."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)

        mock_sandbox.sandbox.return_value.__enter__ = MagicMock(
            return_value=Path("/tmp/test")
        )
        mock_sandbox.sandbox.return_value.__exit__ = MagicMock(return_value=False)

        mock_pi.run.return_value = iter([])
        mock_pi.max_turns_reached = False

        list(
            orchestrator.run(
                repo_url="https://github.com/test/repo",
                branch="main",
                prompt="Test prompt",
                system_prompt="You are a test",
            )
        )

        call_kwargs = mock_pi.run.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a test"

    def test_check_turn_errors_found(self: "TestSimulationOrchestratorRun") -> None:
        """Test check_turn_errors returns message when a turn errored."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)
        error_turn = TurnData(
            turn_index=0,
            stop_reason="error",
            error_message="API key invalid",
        )
        mock_pi._turns = [error_turn]

        result = orchestrator.check_turn_errors()
        assert result == "API key invalid"

    def test_check_turn_errors_none(self: "TestSimulationOrchestratorRun") -> None:
        """Test check_turn_errors returns None when no error turns."""
        mock_pi = MagicMock()
        mock_tracer = MagicMock()
        mock_sandbox = MagicMock()

        orchestrator = SimulationOrchestrator(mock_pi, mock_tracer, mock_sandbox)
        mock_pi._turns = [
            TurnData(turn_index=0, stop_reason="stop"),
            TurnData(turn_index=1, stop_reason="toolUse"),
        ]

        assert orchestrator.check_turn_errors() is None


class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_run_simulation_success(
        self: "TestRunSimulation",
        mock_simulation_config: SimulationConfig,
        mock_model_config: ModelConfig,
    ) -> None:
        """Test successful simulation execution."""
        mock_sandbox = MagicMock()
        mock_pi = MagicMock()
        mock_tracer = MagicMock()

        mock_tracer.log_simulation.return_value.__enter__ = MagicMock(
            return_value=mock_tracer
        )
        mock_tracer.log_simulation.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "beyond_vibes.simulations.orchestration.SimulationOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.run.return_value = iter([])
            mock_orchestrator.check_turn_errors.return_value = None

            run_simulation(
                sim_config=mock_simulation_config,
                model_config=mock_model_config,
                sandbox=mock_sandbox,
                pi_client=mock_pi,
                tracer=mock_tracer,
                prompt="Test prompt",
            )

        mock_tracer.log_simulation.assert_called_once_with(
            mock_simulation_config, mock_model_config
        )

    def test_run_simulation_with_turns(
        self: "TestRunSimulation",
        mock_simulation_config: SimulationConfig,
        mock_model_config: ModelConfig,
    ) -> None:
        """Test simulation that yields turns."""
        mock_sandbox = MagicMock()
        mock_pi = MagicMock()
        mock_tracer = MagicMock()

        mock_tracer.log_simulation.return_value.__enter__ = MagicMock(
            return_value=mock_tracer
        )
        mock_tracer.log_simulation.return_value.__exit__ = MagicMock(return_value=False)

        test_turns = [
            TurnData(turn_index=0, content=[{"type": "text", "text": "A"}]),
            TurnData(turn_index=1, content=[{"type": "text", "text": "B"}]),
        ]

        with patch(
            "beyond_vibes.simulations.orchestration.SimulationOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.run.return_value = iter(test_turns)
            mock_orchestrator.check_turn_errors.return_value = None

            run_simulation(
                sim_config=mock_simulation_config,
                model_config=mock_model_config,
                sandbox=mock_sandbox,
                pi_client=mock_pi,
                tracer=mock_tracer,
                prompt="Test prompt",
            )

        expected_logged_turns = 2
        assert mock_tracer.log_turn.call_count == expected_logged_turns

    def test_run_simulation_error(
        self: "TestRunSimulation",
        mock_simulation_config: SimulationConfig,
        mock_model_config: ModelConfig,
    ) -> None:
        """Test simulation that encounters an error propagates the exception."""
        mock_sandbox = MagicMock()
        mock_pi = MagicMock()
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

            with pytest.raises(Exception, match="Simulation failed"):
                run_simulation(
                    sim_config=mock_simulation_config,
                    model_config=mock_model_config,
                    sandbox=mock_sandbox,
                    pi_client=mock_pi,
                    tracer=mock_tracer,
                    prompt="Test prompt",
                )

        mock_tracer.log_error.assert_called_once()
        assert "Simulation failed" in mock_tracer.log_error.call_args[0][0]

    def test_run_simulation_error_no_session(
        self: "TestRunSimulation",
        mock_simulation_config: SimulationConfig,
        mock_model_config: ModelConfig,
    ) -> None:
        """Test simulation error when tracer has no session propagates exception."""
        mock_sandbox = MagicMock()
        mock_pi = MagicMock()
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

            with pytest.raises(Exception, match="Simulation failed"):
                run_simulation(
                    sim_config=mock_simulation_config,
                    model_config=mock_model_config,
                    sandbox=mock_sandbox,
                    pi_client=mock_pi,
                    tracer=mock_tracer,
                    prompt="Test prompt",
                )

        mock_tracer.log_error.assert_called_once()
        assert "Simulation failed" in mock_tracer.log_error.call_args[0][0]
