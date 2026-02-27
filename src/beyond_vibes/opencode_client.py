"""OpenCode client wrapper for running simulations."""

import logging
from pathlib import Path

import httpx

from beyond_vibes.settings import settings

logger = logging.getLogger(__name__)


class OpenCodeClient:
    """Wrapper around OpenCode API for simulation runs."""

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the OpenCode client.

        Args:
            base_url: Optional base URL for the OpenCode API.
                      Defaults to settings.opencode_url.

        """
        self.base_url = base_url or settings.opencode_url
        self._client = httpx.Client(base_url=self.base_url, timeout=300.0)
        self._working_dir: str | None = None

    def create_session(self, working_dir: Path) -> str:
        """Initialize a new session with the given working directory."""
        self._working_dir = str(working_dir)
        response = self._client.post(
            "/session", params={"directory": self._working_dir}
        )
        response.raise_for_status()
        session_data = response.json()
        session_id = session_data["id"]

        logger.info("Created session %s with working dir: %s", session_id, working_dir)
        return session_id

    def send_prompt(
        self,
        session_id: str,
        prompt: str,
        model_id: str | None = None,
        agent: str = "build",
    ) -> None:
        """Send a prompt to an existing session asynchronously."""
        provider_id = settings.opencode_provider
        message_id = f"msg_{session_id[:8]}"

        params = {"directory": self._working_dir} if self._working_dir else None
        response = self._client.post(
            f"/session/{session_id}/prompt_async",
            params=params,
            json={
                "model": {
                    "modelID": model_id,
                    "providerID": provider_id,
                },
                "agent": agent,
                "parts": [{"type": "text", "text": prompt}],
                "messageID": message_id,
            },
        )
        response.raise_for_status()

        logger.debug("Sent prompt to session %s", session_id)

    def get_messages(self, session_id: str) -> list[dict]:
        """Get all messages from a session."""
        params = {"directory": self._working_dir} if self._working_dir else None
        response = self._client.get(f"/session/{session_id}/message", params=params)
        response.raise_for_status()
        return response.json()

    def abort_session(self, session_id: str) -> bool:
        """Abort a running session."""
        params = {"directory": self._working_dir} if self._working_dir else None
        response = self._client.post(f"/session/{session_id}/abort", params=params)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
