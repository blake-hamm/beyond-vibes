"""pi.dev CLI client for running simulations."""

import json
import logging
import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Self

logger = logging.getLogger(__name__)


class PiDevError(Exception):
    """Base exception for pi.dev client errors."""


class PiDevTimeoutError(PiDevError):
    """Raised when pi.dev subprocess times out."""


@dataclass
class TurnData:
    """Native pi turn structure from assistant message_end event."""

    turn_index: int
    content: list[dict] = field(default_factory=list)
    usage: dict | None = None
    stop_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    raw_message: dict | None = None


class PiDevClient:
    """Wrapper around pi.dev CLI for simulation runs."""

    def __init__(
        self,
        provider: str = "kimi-coding",
        model: str = "kimi-for-coding",
        timeout: float = 300.0,
        stderr_log: Path | str | None = None,
    ) -> None:
        """Initialize the pi.dev client."""
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.stderr_log = Path(stderr_log) if stderr_log else Path("pi_dev_stderr.log")
        self._proc: subprocess.Popen | None = None
        self._timer: threading.Timer | None = None

    def run(
        self,
        prompt: str,
        working_dir: Path | None = None,
        max_turns: int = 75,
        system_prompt: str | None = None,
    ) -> Iterator[TurnData]:
        """Run pi with prompt and yield TurnData for each assistant turn."""
        cmd = [
            "pi",
            "--mode",
            "json",
            "--no-session",
            "--provider",
            self.provider,
            "--model",
            self.model,
            "--print",
        ]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        cmd.append(prompt)

        stderr_fd = self.stderr_log.open("w")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_fd,
                text=True,
                cwd=working_dir,
                preexec_fn=os.setsid,  # noqa: S603,PLW1509 — new process group for killpg
            )

            if self.timeout > 0:
                self._timer = threading.Timer(self.timeout, self._kill)
                self._timer.start()

            yield from self._read_turns(max_turns)

        finally:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._proc:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self._proc.wait(timeout=1)
                self._proc = None
            stderr_fd.close()

    def abort(self) -> None:
        """Abort the running pi process."""
        self._kill()

    def _kill(self) -> None:
        if self._proc:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _read_turns(  # noqa: PLR0912,PLR0915
        self, max_turns: int
    ) -> Iterator[TurnData]:
        if not self._proc or not self._proc.stdout:
            raise PiDevError("Process not started")

        assistant_count = 0
        current_turn: TurnData | None = None
        seen_assistant_end = False

        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON line: %s", e)
                continue

            event_type = event.get("type")

            if event_type == "message_start":
                msg = event.get("message", {})
                if msg.get("role") == "assistant":
                    current_turn = TurnData(turn_index=assistant_count)

            elif event_type == "message_update":
                if current_turn is not None:
                    ame = event.get("assistantMessageEvent", {})
                    delta_type = ame.get("type")
                    if delta_type == "text_delta":
                        delta_text = ame.get("delta", "")
                        if (
                            current_turn.content
                            and current_turn.content[-1].get("type") == "text"
                        ):
                            current_turn.content[-1]["text"] += delta_text
                        else:
                            current_turn.content.append(
                                {"type": "text", "text": delta_text}
                            )
                    elif delta_type == "thinking_delta":
                        delta_text = ame.get("delta", "")
                        if (
                            current_turn.content
                            and current_turn.content[-1].get("type") == "thinking"
                        ):
                            current_turn.content[-1]["text"] += delta_text
                        else:
                            current_turn.content.append(
                                {"type": "thinking", "text": delta_text}
                            )

            elif event_type == "message_end":
                msg = event.get("message", {})
                if msg.get("role") == "assistant":
                    seen_assistant_end = True
                    if current_turn is None:
                        current_turn = TurnData(turn_index=assistant_count)
                    # message_end is source of truth for content
                    current_turn.content = msg.get("content", [])
                    current_turn.usage = msg.get("usage")
                    current_turn.stop_reason = msg.get("stopReason")
                    current_turn.raw_message = msg

                    yield current_turn
                    assistant_count += 1
                    current_turn = None

                    if assistant_count >= max_turns:
                        logger.info("Max turns (%d) reached, terminating pi", max_turns)
                        self._kill()
                        return

            elif event_type == "tool_execution_start":
                if current_turn is not None:
                    current_turn.tool_calls.append(
                        {
                            "toolCallId": event.get("toolCallId"),
                            "toolName": event.get("toolName"),
                            "args": event.get("args"),
                        }
                    )

            elif event_type == "tool_execution_end":
                if current_turn is not None:
                    current_turn.tool_results.append(
                        {
                            "toolCallId": event.get("toolCallId"),
                            "toolName": event.get("toolName"),
                            "result": event.get("result"),
                            "isError": event.get("isError", False),
                        }
                    )

        if not seen_assistant_end:
            raise PiDevError("Premature EOF: no assistant message_end events received")

        logger.info("pi stream ended after %d turns", assistant_count)

    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, ensuring cleanup."""
        self.abort()
