"""OpenCode client wrapper for running simulations."""

import logging
from pathlib import Path
from typing import Any

from opencode_ai import APIConnectionError, APIError, Client

from beyond_vibes.settings import settings

# ruff: noqa: ANN401
logger = logging.getLogger(__name__)


class OpenCodeClient:
    """Wrapper around opencode-ai SDK for simulation runs."""

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the OpenCode client.

        Args:
            base_url: Optional base URL for the OpenCode API.
                      Defaults to settings.opencode_url.

        """
        self.base_url = base_url or settings.opencode_url
        self.client = Client(base_url=self.base_url)
        self._session_id: str | None = None

    def create_session(self, working_dir: Path) -> str:
        """Initialize a new session with the given working directory."""
        try:
            session = self.client.session.create()
            self._session_id = session.id

            self.client.session.init(
                id=session.id,
                message_id="init",
                model_id=settings.opencode_provider,
                provider_id=settings.opencode_provider,
            )

            logger.info(
                "Created session %s with working dir: %s", session.id, working_dir
            )
            return session.id
        except APIConnectionError as e:
            logger.error("Failed to connect to OpenCode server: %s", e)
            raise
        except APIError as e:
            logger.error("Failed to create session: %s", e)
            raise

    def run_prompt(
        self,
        session_id: str,
        prompt: str,
        model_id: str | None = None,
        provider_id: str | None = None,
    ) -> Any:
        """Execute a prompt in the given session."""
        model_id = model_id or settings.opencode_provider
        provider_id = provider_id or settings.opencode_provider

        try:
            text_part: Any = {"text": prompt}
            parts: list[Any] = [text_part]

            response = self.client.session.chat(
                id=session_id,
                model_id=model_id,
                parts=parts,
                provider_id=provider_id,
            )

            logger.debug(
                "Ran prompt in session %s, got response: %s", session_id, response.id
            )
            return response
        except APIConnectionError as e:
            logger.error("Connection error running prompt: %s", e)
            raise
        except APIError as e:
            logger.error("Error running prompt: %s", e)
            raise
