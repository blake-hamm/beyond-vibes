"""Tests for OpenCode client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from beyond_vibes.simulations.opencode import OpenCodeClient


class TestOpenCodeClientInit:
    """Tests for OpenCodeClient initialization."""

    def test_init_with_default_url(self) -> None:
        """Test initialization with default URL from settings."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://default:4096"
            client = OpenCodeClient()

        assert client.base_url == "http://default:4096"
        assert client._working_dir is None

    def test_init_with_custom_url(self) -> None:
        """Test initialization with custom base URL."""
        with patch("beyond_vibes.simulations.opencode.settings"):
            client = OpenCodeClient(base_url="http://custom:8080")

        assert client.base_url == "http://custom:8080"

    def test_init_creates_http_client(self) -> None:
        """Test that initialization creates httpx client."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        assert isinstance(client._client, httpx.Client)


class TestCreateSession:
    """Tests for create_session method."""

    def test_create_session_success(self) -> None:
        """Test successful session creation."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "session-123"}
        mock_response.raise_for_status.return_value = None
        client._client.post = MagicMock(return_value=mock_response)

        working_dir = Path("/tmp/test")
        session_id = client.create_session(working_dir)

        assert session_id == "session-123"
        assert client._working_dir == str(working_dir)
        client._client.post.assert_called_once_with(
            "/session", params={"directory": str(working_dir)}
        )

    def test_create_session_http_error(self) -> None:
        """Test HTTP error handling during session creation."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError(
            "Connection failed"
        )
        client._client.post = MagicMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPError, match="Connection failed"):
            client.create_session(Path("/tmp/test"))


class TestSendPrompt:
    """Tests for send_prompt method."""

    def test_send_prompt_with_all_params(self) -> None:
        """Test sending prompt with all parameters."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            client._working_dir = "/tmp/test"

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        client._client.post = MagicMock(return_value=mock_response)

        client.send_prompt(
            session_id="session-123",
            prompt="Test prompt",
            model_id="gpt-4",
            provider="openai",
            agent="code",
        )

        expected_call = client._client.post.call_args
        assert expected_call[0][0] == "/session/session-123/prompt_async"
        assert expected_call[1]["params"] == {"directory": "/tmp/test"}
        assert expected_call[1]["json"]["model"]["modelID"] == "gpt-4"
        assert expected_call[1]["json"]["model"]["providerID"] == "openai"
        assert expected_call[1]["json"]["agent"] == "code"
        assert expected_call[1]["json"]["parts"][0]["text"] == "Test prompt"

    def test_send_prompt_with_defaults(self) -> None:
        """Test sending prompt with default parameters."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            client._working_dir = "/tmp/test"

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        client._client.post = MagicMock(return_value=mock_response)

        client.send_prompt(session_id="session-456", prompt="Hello")

        expected_call = client._client.post.call_args
        assert expected_call[1]["json"]["model"]["providerID"] == "kimi-for-coding"
        assert expected_call[1]["json"]["agent"] == "orchestrator"
        assert expected_call[1]["json"]["model"]["modelID"] is None

    def test_send_prompt_without_working_dir(self) -> None:
        """Test sending prompt when working_dir is not set."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            # Don't set working_dir

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        client._client.post = MagicMock(return_value=mock_response)

        client.send_prompt(session_id="session-789", prompt="Test")

        expected_call = client._client.post.call_args
        assert expected_call[1]["params"] is None

    def test_send_prompt_http_error(self) -> None:
        """Test HTTP error handling when sending prompt."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError("Bad request")
        client._client.post = MagicMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPError):
            client.send_prompt("session-123", "prompt")


class TestGetMessages:
    """Tests for get_messages method."""

    def test_get_messages_success(self) -> None:
        """Test successful message retrieval."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            client._working_dir = "/tmp/test"

        expected_messages = [{"id": "msg-1"}, {"id": "msg-2"}]
        mock_response = MagicMock()
        mock_response.json.return_value = expected_messages
        mock_response.raise_for_status.return_value = None
        client._client.get = MagicMock(return_value=mock_response)

        messages = client.get_messages("session-123")

        assert messages == expected_messages
        client._client.get.assert_called_once_with(
            "/session/session-123/message", params={"directory": "/tmp/test"}
        )

    def test_get_messages_without_working_dir(self) -> None:
        """Test getting messages without working_dir."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        client._client.get = MagicMock(return_value=mock_response)

        client.get_messages("session-456")

        expected_call = client._client.get.call_args
        assert expected_call[1]["params"] is None

    def test_get_messages_http_error(self) -> None:
        """Test HTTP error handling when getting messages."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError("Not found")
        client._client.get = MagicMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPError):
            client.get_messages("session-123")


class TestAbortSession:
    """Tests for abort_session method."""

    def test_abort_session_success(self) -> None:
        """Test successful session abort."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            client._working_dir = "/tmp/test"

        mock_response = MagicMock()
        mock_response.json.return_value = True
        mock_response.raise_for_status.return_value = None
        client._client.post = MagicMock(return_value=mock_response)

        result = client.abort_session("session-123")

        assert result is True
        client._client.post.assert_called_once_with(
            "/session/session-123/abort", params={"directory": "/tmp/test"}
        )

    def test_abort_session_returns_false(self) -> None:
        """Test abort session returning False."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        mock_response = MagicMock()
        mock_response.json.return_value = False
        mock_response.raise_for_status.return_value = None
        client._client.post = MagicMock(return_value=mock_response)

        result = client.abort_session("session-123")

        assert result is False

    def test_abort_session_http_error(self) -> None:
        """Test HTTP error handling when aborting session."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPError("Server error")
        client._client.post = MagicMock(return_value=mock_response)

        with pytest.raises(httpx.HTTPError):
            client.abort_session("session-123")


class TestClose:
    """Tests for close method."""

    def test_close_client(self) -> None:
        """Test closing HTTP client."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            client._client.close = MagicMock()

        client.close()

        client._client.close.assert_called_once()


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter(self) -> None:
        """Test entering context manager."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()

        with client as ctx:
            assert ctx is client

    def test_context_manager_exit(self) -> None:
        """Test exiting context manager closes client."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            client._client.close = MagicMock()

        with client:
            pass

        client._client.close.assert_called_once()

    def test_context_manager_exit_with_exception(self) -> None:
        """Test context manager handles exceptions properly."""
        with patch("beyond_vibes.simulations.opencode.settings") as mock_settings:
            mock_settings.opencode_url = "http://test:4096"
            client = OpenCodeClient()
            client._client.close = MagicMock()

        with pytest.raises(ValueError, match="Test error"):
            with client:
                raise ValueError("Test error")

        client._client.close.assert_called_once()
