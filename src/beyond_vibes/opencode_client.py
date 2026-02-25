"""OpenCode client wrapper for running simulations."""

import logging
import time
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

        logger.info("Created session %s with working dir: %s", session_id, working_dir)
        return session_id

    def run_prompt(
        self,
        session_id: str,
        prompt: str,
        model_id: str | None = None,
        provider_id: str | None = None,
        agent: str = "build",
    ) -> dict:
        """Execute a prompt in the given session."""
        model_id = model_id or settings.opencode_provider
        provider_id = provider_id or settings.opencode_provider
        message_id = f"msg_{session_id[:8]}"

        response = self._client.post(
            f"/session/{session_id}/prompt_async",
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

        logger.debug(
            "Waiting for message to complete in session %s",
            session_id,
        )

        last_message_count = 0
        max_attempts = 600
        attempt = 0
        while attempt < max_attempts:
            messages_response = self._client.get(f"/session/{session_id}/message")
            messages_response.raise_for_status()
            messages = messages_response.json()

            assistant_messages = [
                m for m in messages if m.get("info", {}).get("role") == "assistant"
            ]

            if assistant_messages:
                latest = assistant_messages[-1]
                msg_info = latest.get("info", {})
                if msg_info.get("time", {}).get("completed"):
                    logger.debug(
                        "Session %s completed with %d messages",
                        session_id,
                        len(messages),
                    )
                    return latest

            if len(messages) > last_message_count:
                logger.debug(
                    "Session %s has %d messages (waiting for completion)",
                    session_id,
                    len(messages),
                )
                last_message_count = len(messages)

            attempt += 1
            time.sleep(5)

        logger.warning(
            "Timed out waiting for session %s to complete",
            session_id,
        )
        return {
            "info": {
                "error": {
                    "name": "Timeout",
                    "message": f"Timed out waiting for session {session_id}",
                }
            },
            "parts": [],
        }

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
