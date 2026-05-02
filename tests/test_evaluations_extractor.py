"""Tests for extractor module."""

from unittest.mock import Mock, patch

import pytest

from beyond_vibes.evaluations.extractor import (
    _extract_final_message,
    _load_artifact,
    _load_trace_session,
    extract_run_data,
    query_simulation_runs,
)


class TestExtractFinalMessage:
    """Tests for _extract_final_message function."""

    def test_with_text_parts(self) -> None:
        """Test extraction with text and thinking blocks."""
        trace = {
            "turns": [
                {
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "thinking", "thinking": "Thinking..."},
                    ]
                }
            ]
        }

        result = _extract_final_message(trace)
        assert "Hello" in result
        assert "Thinking..." in result

    def test_empty_trace(self) -> None:
        """Test empty trace returns empty string."""
        assert _extract_final_message({"turns": []}) == ""

    def test_no_messages_key(self) -> None:
        """Test missing messages key returns empty string."""
        assert _extract_final_message({}) == ""

    def test_legacy_raw_message_format(self) -> None:
        """Test backward compatibility with old OpenCode raw_message format."""
        trace = {
            "turns": [
                {
                    "raw_message": {
                        "content": [
                            {"type": "text", "text": "Legacy"},
                        ]
                    }
                }
            ]
        }

        result = _extract_final_message(trace)
        assert result == "Legacy"


class TestLoadArtifact:
    """Tests for _load_artifact function."""

    @patch("mlflow.artifacts.load_text")
    def test_success(self, mock_load: Mock) -> None:
        """Test successful artifact load."""
        mock_load.return_value = "content"

        result = _load_artifact("run-123", "file.txt")

        assert result == "content"
        mock_load.assert_called_once_with("runs:/run-123/file.txt")

    @patch("mlflow.artifacts.load_text")
    def test_not_found_returns_none(self, mock_load: Mock) -> None:
        """Test missing artifact returns None."""
        mock_load.side_effect = Exception("Not found")

        assert _load_artifact("run-123", "missing.txt") is None


class TestLoadTraceSession:
    """Tests for _load_trace_session function."""

    @patch("mlflow.artifacts.load_dict")
    def test_success(self, mock_load: Mock) -> None:
        """Test successful trace load."""
        mock_trace = {"total_messages": 10}
        mock_load.return_value = mock_trace

        result = _load_trace_session("run-123")

        assert result == mock_trace

    @patch("mlflow.artifacts.load_dict")
    def test_missing_returns_fallback(self, mock_load: Mock) -> None:
        """Test missing trace returns fallback."""
        mock_load.side_effect = Exception("Not found")

        result = _load_trace_session("run-123")

        assert result["total_messages"] == 0
        assert "turns" in result


class TestExtractRunData:
    """Tests for extract_run_data function."""

    @patch("mlflow.get_run")
    @patch("beyond_vibes.evaluations.extractor._load_artifact")
    @patch("beyond_vibes.evaluations.extractor._load_trace_session")
    def test_full_extraction(
        self,
        mock_trace: Mock,
        mock_artifact: Mock,
        mock_get_run: Mock,
    ) -> None:
        """Test full data extraction pipeline."""
        mock_run = Mock()
        mock_run.data.tags = {"task.name": "test", "task.archetype": "test"}
        mock_run.data.params = {"task.prompt": "Do something"}
        mock_get_run.return_value = mock_run

        mock_artifact.return_value = "system prompt"
        mock_trace.return_value = {
            "turns": [{"content": [{"type": "text", "text": "Done"}]}],
        }

        result = extract_run_data("run-123")

        assert result.run_id == "run-123"
        assert result.task_name == "test"
        assert result.system_prompt == "system prompt"

    @patch("mlflow.get_run")
    def test_missing_run_raises(self, mock_get_run: Mock) -> None:
        """Test missing run raises exception."""
        mock_get_run.side_effect = Exception("Run not found")

        with pytest.raises(Exception, match="Run not found"):
            extract_run_data("missing")


class TestQuerySimulationRuns:
    """Tests for query_simulation_runs function."""

    @patch("mlflow.get_run")
    def test_by_run_ids(self, mock_get_run: Mock) -> None:
        """Test querying by specific run IDs."""
        mock_get_run.return_value = Mock()

        results = query_simulation_runs(run_ids=["run-1", "run-2"])

        assert len(results) == 2  # noqa: PLR2004

    @patch("mlflow.get_run")
    def test_missing_run_skipped(self, mock_get_run: Mock) -> None:
        """Test missing runs are skipped."""
        mock_get_run.side_effect = [Mock(), Exception("Not found")]

        results = query_simulation_runs(run_ids=["run-1", "run-2"])

        assert len(results) == 1

    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.search_runs")
    def test_by_experiment(
        self,
        mock_search: Mock,
        mock_get_exp: Mock,
    ) -> None:
        """Test querying by experiment."""
        mock_exp = Mock()
        mock_exp.experiment_id = "exp-123"
        mock_get_exp.return_value = mock_exp
        mock_search.return_value = [Mock()]

        results = query_simulation_runs(experiment="test")

        assert len(results) == 1

    @patch("mlflow.get_experiment_by_name")
    def test_missing_experiment_returns_empty(self, mock_get_exp: Mock) -> None:
        """Test missing experiment returns empty list."""
        mock_get_exp.return_value = None

        assert query_simulation_runs(experiment="missing") == []
