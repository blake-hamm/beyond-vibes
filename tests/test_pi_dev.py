"""Tests for pi.dev client."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beyond_vibes.simulations.pi_dev import (
    PiDevClient,
    PiDevError,
    TurnData,
)


@pytest.fixture
def fixture_lines() -> list[str]:
    """Load the fixture JSONL lines."""
    fixture_path = Path(__file__).parent / "fixtures" / "pi_dev_output.jsonl"
    return fixture_path.read_text().strip().split("\n")


class TestPiDevClientInit:
    """Tests for PiDevClient initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with default values."""
        client = PiDevClient()
        assert client.provider == "kimi-coding"
        assert client.model == "kimi-for-coding"
        expected_timeout = 2400.0
        assert client.timeout == expected_timeout
        assert client.stderr_log is None

    def test_init_custom(self) -> None:
        """Test initialization with custom values."""
        client = PiDevClient(
            provider="openai",
            model="gpt-4o",
            timeout=60.0,
            stderr_log="/tmp/pi_err.log",
        )
        assert client.provider == "openai"
        assert client.model == "gpt-4o"
        expected_timeout = 60.0
        assert client.timeout == expected_timeout
        assert client.stderr_log == Path("/tmp/pi_err.log")


class TestPiDevClientRun:
    """Tests for run method."""

    def test_run_success(self, fixture_lines: list[str]) -> None:
        """Test successful run yielding TurnData."""
        client = PiDevClient()

        mock_proc = MagicMock()
        mock_proc.stdout = iter(fixture_lines)
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                turns = list(client.run("Say hello"))

        assert len(turns) == 1
        turn = turns[0]
        assert turn.turn_index == 0
        assert turn.stop_reason == "stop"
        assert turn.error_message is None
        assert turn.usage is not None
        expected_input_tokens = 4634
        expected_output_tokens = 66
        expected_content_blocks = 2
        assert turn.usage["input"] == expected_input_tokens
        assert turn.usage["output"] == expected_output_tokens
        assert len(turn.content) == expected_content_blocks
        assert turn.content[0]["type"] == "thinking"
        assert turn.content[1]["type"] == "text"
        assert turn.content[1]["text"] == "Hello"
        assert turn.response_id == "msg_OTKGtprx2JutpeeRM9Snknvs"

    def test_run_command_construction(self) -> None:
        """Test that pi command is constructed correctly."""
        client = PiDevClient(provider="test-provider", model="test-model")

        mock_proc = MagicMock()
        mock_proc.stdout = iter(
            [
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hi"}],
                            "usage": {"input": 10, "output": 5, "totalTokens": 15},
                            "stopReason": "stop",
                        },
                    }
                )
            ]
        )
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            with patch("os.getpgid", return_value=1234):
                list(client.run("Test prompt", system_prompt="You are a test"))

        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert Path(cmd[0]).name == "pi"
        assert "--mode" in cmd
        assert "json" in cmd
        assert "--no-session" in cmd
        assert "--provider" in cmd
        assert "test-provider" in cmd
        assert "--model" in cmd
        assert "test-model" in cmd
        assert "--print" in cmd
        assert "--system-prompt" in cmd
        assert "You are a test" in cmd
        assert "Test prompt" in cmd
        assert call_args[1]["cwd"] is None

    def test_run_working_dir(self) -> None:
        """Test that working directory is passed to subprocess."""
        client = PiDevClient()
        mock_proc = MagicMock()
        mock_proc.stdout = iter(
            [
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hi"}],
                            "usage": {"input": 10, "output": 5, "totalTokens": 15},
                            "stopReason": "stop",
                        },
                    }
                )
            ]
        )
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            with patch("os.getpgid", return_value=1234):
                list(client.run("Test prompt", working_dir=Path("/tmp/test")))

        call_args = mock_popen.call_args
        assert call_args[1]["cwd"] == Path("/tmp/test")

    def test_run_premature_eof(self) -> None:
        """Test premature EOF raises PiDevError."""
        client = PiDevClient()
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                with pytest.raises(PiDevError, match="Premature EOF"):
                    list(client.run("Test prompt"))

    def test_run_captures_error_message(self) -> None:
        """Test that errorMessage is captured when stopReason is error."""
        client = PiDevClient()
        lines = [
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [],
                        "usage": {"input": 0, "output": 0, "totalTokens": 0},
                        "stopReason": "error",
                        "errorMessage": "401 Incorrect API key",
                    },
                }
            ),
        ]

        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                turns = list(client.run("Test prompt"))

        assert len(turns) == 1
        assert turns[0].stop_reason == "error"
        assert turns[0].error_message == "401 Incorrect API key"

    def test_run_max_turns(self, fixture_lines: list[str]) -> None:
        """Test max_turns stops after N assistant turns."""
        client = PiDevClient()
        # Duplicate fixture to simulate 2 turns
        doubled = fixture_lines + fixture_lines

        mock_proc = MagicMock()
        mock_proc.stdout = iter(doubled)
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                with patch.object(client, "_max_turns_reached", False):
                    with patch("os.killpg") as mock_killpg:
                        turns = list(client.run("Say hello", max_turns=1))

        assert len(turns) == 1
        mock_killpg.assert_called()

    def test_run_invalid_json_raises(self) -> None:
        """Test that invalid JSON lines raise PiDevError."""
        client = PiDevClient()
        lines = [
            "not json",
        ]

        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                with pytest.raises(PiDevError, match="Invalid JSON line"):
                    list(client.run("Test"))


@pytest.fixture
def tool_fixture_lines() -> list[str]:
    """Load the tool fixture JSONL lines."""
    fixture_path = Path(__file__).parent / "fixtures" / "pi_dev_tool_output.jsonl"
    return fixture_path.read_text().strip().split("\n")


class TestPiDevClientToolRun:
    """Tests for run method with tool calls."""

    def test_run_with_tool_calls(self, tool_fixture_lines: list[str]) -> None:
        """Test that tool calls and results are captured in TurnData."""
        client = PiDevClient()

        mock_proc = MagicMock()
        mock_proc.stdout = iter(tool_fixture_lines)
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                turns = list(client.run("Read file"))

        expected_turn_count = 2
        assert len(turns) == expected_turn_count

        # First turn has tool call
        turn0 = turns[0]
        assert turn0.turn_index == 0
        assert turn0.stop_reason == "toolUse"
        assert len(turn0.tool_calls) == 1
        assert turn0.tool_calls[0]["toolName"] == "read"
        assert turn0.tool_calls[0]["args"] == {"path": "/tmp/test.txt"}
        assert len(turn0.tool_results) == 1
        assert turn0.tool_results[0]["isError"] is False
        assert turn0.tool_results[0]["result"]["content"][0]["text"] == "Hello world"

        # Second turn has no tools
        turn1 = turns[1]
        assert turn1.turn_index == 1
        assert turn1.stop_reason == "stop"
        assert len(turn1.tool_calls) == 0
        assert len(turn1.tool_results) == 0

    def test_tool_call_without_message_start(self) -> None:
        """Test tool events are captured even without preceding message_start."""
        client = PiDevClient()
        lines = [
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hi"}],
                        "usage": {"input": 10, "output": 5, "totalTokens": 15},
                        "stopReason": "stop",
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_start",
                    "toolCallId": "tool_1",
                    "toolName": "bash",
                    "args": {"command": "ls"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_execution_end",
                    "toolCallId": "tool_1",
                    "toolName": "bash",
                    "result": {"stdout": "file.txt"},
                    "isError": False,
                }
            ),
        ]

        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                turns = list(client.run("Test"))

        assert len(turns) == 1
        assert len(turns[0].tool_calls) == 1
        assert turns[0].tool_calls[0]["toolName"] == "bash"
        assert len(turns[0].tool_results) == 1


class TestPiDevClientLatency:
    """Tests for latency capture in _read_turns."""

    def test_run_captures_timestamps(self, fixture_lines: list[str]) -> None:
        """Test that run captures wall-clock timestamps."""
        client = PiDevClient()
        mock_proc = MagicMock()
        mock_proc.stdout = iter(fixture_lines)
        mock_proc.pid = 1234
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("os.getpgid", return_value=1234):
                turns = list(client.run("Say hello"))

        assert len(turns) == 1
        turn = turns[0]
        assert turn.user_message_end is not None
        assert turn.assistant_message_end is not None
        assert turn.prompt_processing_time_seconds is not None
        assert turn.generation_time_seconds is not None
        assert turn.time_to_first_token_seconds is not None


class TestTurnData:
    """Tests for TurnData dataclass."""

    def test_defaults(self) -> None:
        """Test TurnData default values."""
        turn = TurnData(turn_index=0)
        assert turn.turn_index == 0
        assert turn.content == []
        assert turn.usage is None
        assert turn.stop_reason is None
        assert turn.tool_calls == []
        assert turn.tool_results == []
        assert turn.response_id is None
        assert turn.error_message is None
        assert turn.user_message_end is None
        assert turn.assistant_message_start is None
        assert turn.assistant_message_end is None
        assert turn.prompt_processing_time_seconds is None
        assert turn.time_to_first_token_seconds is None
        assert turn.generation_time_seconds is None
        assert turn.prompt_tokens_per_second is None
        assert turn.generation_tokens_per_second is None


class TestPiDevError:
    """Tests for PiDevError exception."""

    def test_error_with_stderr(self) -> None:
        """Test PiDevError carries stderr."""
        exc = PiDevError("boom", stderr="some stderr")
        assert exc.stderr == "some stderr"

    def test_error_without_stderr(self) -> None:
        """Test PiDevError without stderr."""
        exc = PiDevError("boom")
        assert exc.stderr is None
