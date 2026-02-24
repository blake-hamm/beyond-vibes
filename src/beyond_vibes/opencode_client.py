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
        self._session_id: str | None = None

    def create_session(self, working_dir: Path) -> str:
        """Initialize a new session with the given working directory."""
        response = self._client.post("/session", json={})
        response.raise_for_status()
        session_data = response.json()
        session_id = session_data["id"]
        self._session_id = session_id

        self._init_session(
            session_id=session_id,
            model_id=settings.opencode_provider,
            provider_id=settings.opencode_provider,
        )

        logger.info("Created session %s with working dir: %s", session_id, working_dir)
        return session_id

    def _init_session(
        self,
        session_id: str,
        model_id: str,
        provider_id: str,
    ) -> dict:
        """Initialize a session (analyze the app and create AGENTS.md)."""
        response = self._client.post(
            f"/session/{session_id}/init",
            json={
                "messageID": "msg_init",
                "modelID": model_id,
                "providerID": provider_id,
            },
        )
        response.raise_for_status()
        return response.json()

    def run_prompt(
        self,
        session_id: str,
        prompt: str,
        model_id: str | None = None,
        provider_id: str | None = None,
    ) -> dict:
        """Execute a prompt in the given session."""
        model_id = model_id or settings.opencode_provider
        provider_id = provider_id or settings.opencode_provider
        message_id = f"msg_{session_id[:8]}"

        response = self._client.post(
            f"/session/{session_id}/message",
            json={
                "modelID": model_id,
                "providerID": provider_id,
                "parts": [{"type": "text", "text": prompt}],
                "messageID": message_id,
            },
        )
        response.raise_for_status()
        response_data = response.json()

        logger.debug(
            "Ran prompt in session %s, got response: %s",
            session_id,
            response_data.get("info", {}).get("id"),
        )
        return response_data

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
