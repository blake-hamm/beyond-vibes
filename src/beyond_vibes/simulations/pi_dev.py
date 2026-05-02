"""pi.dev CLI client for running simulations."""

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class PiDevError(Exception):
    """Base exception for pi.dev client errors."""

    def __init__(self, message: str, stderr: str | None = None) -> None:
        """Initialize with message and optional captured stderr."""
        super().__init__(message)
        self.stderr = stderr


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
    response_id: str | None = None
    error_message: str | None = None

    # Raw wall-clock timestamps captured during JSONL streaming.
    # Perf-counter values (float seconds) are used for latency math.
    # Nanosecond values (int) are epoch timestamps for MLflow spans.
    user_message_end: float | None = None
    assistant_message_start: float | None = None
    assistant_message_end: float | None = None
    assistant_message_start_ns: int | None = None
    assistant_message_end_ns: int | None = None

    # Derived latency metrics (computed before yielding a turn)
    prompt_processing_time_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    generation_time_seconds: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None

    # Simulation-level error for this turn (set by MlflowTracer.log_error)
    simulation_error: str | None = None


class PiDevClient:
    """Wrapper around pi.dev CLI for simulation runs."""

    def __init__(
        self,
        provider: str = "kimi-coding",
        model: str = "kimi-for-coding",
        timeout: float = 2400.0,
        stderr_log: Path | str | None = None,
    ) -> None:
        """Initialize the pi.dev client."""
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.stderr_log = Path(stderr_log) if stderr_log else None
        self._max_turns_reached = False

    @property
    def max_turns_reached(self) -> bool:
        """Return True if the last run stopped due to max_turns."""
        return self._max_turns_reached

    def run(  # noqa: PLR0912,PLR0915
        self,
        prompt: str,
        working_dir: Path | None = None,
        max_turns: int = 75,
        system_prompt: str | None = None,
    ) -> Iterator[TurnData]:
        """Run pi with prompt and yield TurnData for each assistant turn."""
        pi_bin = shutil.which("pi")
        if pi_bin is None:
            raise PiDevError("pi CLI not found in PATH")

        self._max_turns_reached = False

        stderr_path = self.stderr_log or Path(tempfile.mktemp(suffix="_pi_stderr.log"))
        stderr_fd = stderr_path.open("w")

        proc: subprocess.Popen | None = None
        timer: threading.Timer | None = None
        timed_out = False
        terminated_by_us = False

        cmd = [
            pi_bin,
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

        def _kill() -> None:
            nonlocal terminated_by_us
            terminated_by_us = True
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        def _timeout() -> None:
            nonlocal timed_out
            timed_out = True
            _kill()

        def _read_stderr() -> str:
            if not stderr_path.exists():
                return ""
            try:
                return stderr_path.read_text()
            except Exception:
                return ""

        def _cleanup_proc() -> None:
            if proc is None:
                return
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

        def _read_turns() -> Iterator[TurnData]:  # noqa: PLR0912,PLR0915
            nonlocal timed_out, terminated_by_us
            if proc is None or not proc.stdout:
                raise PiDevError("Process not started")

            assistant_count = 0
            current_turn: TurnData | None = None
            pending_turn: TurnData | None = None
            seen_assistant_end = False
            pending_user_message_end: float | None = None

            for raw_line in proc.stdout:
                t_recv = time.perf_counter()
                t_recv_ns = time.time_ns()
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PiDevError(f"Invalid JSON line: {exc}") from exc

                event_type = event.get("type")

                if event_type == "message_start":
                    msg = event.get("message", {})
                    if msg.get("role") == "assistant":
                        logger.debug(
                            "message_start (assistant) turn=%d", assistant_count
                        )
                        if pending_turn is not None:
                            logger.debug(
                                "Yielding pending turn %d",
                                pending_turn.turn_index,
                            )
                            yield pending_turn
                            pending_turn = None
                        current_turn = TurnData(turn_index=assistant_count)
                        current_turn.user_message_end = pending_user_message_end
                        current_turn.assistant_message_start = t_recv
                        current_turn.assistant_message_start_ns = t_recv_ns
                        pending_user_message_end = None

                elif event_type == "message_update":
                    # Deltas are discarded; message_end is the source of truth.
                    pass

                elif event_type == "message_end":
                    msg = event.get("message", {})
                    role = msg.get("role")
                    if role == "user":
                        pending_user_message_end = t_recv
                        continue
                    if role == "assistant":
                        seen_assistant_end = True
                        stop_reason = msg.get("stopReason")
                        error_message = msg.get("errorMessage")
                        logger.debug(
                            "message_end (assistant) stop_reason=%s error=%s",
                            stop_reason,
                            error_message,
                        )
                        if current_turn is None:
                            current_turn = TurnData(turn_index=assistant_count)
                            current_turn.user_message_end = pending_user_message_end
                            pending_user_message_end = None

                        current_turn.content = msg.get("content", [])
                        current_turn.usage = msg.get("usage")
                        current_turn.stop_reason = stop_reason
                        current_turn.error_message = error_message
                        current_turn.response_id = msg.get("responseId")
                        current_turn.assistant_message_end = t_recv
                        current_turn.assistant_message_end_ns = t_recv_ns

                        # Compute latency metrics inline
                        usage = current_turn.usage or {}
                        input_tokens = usage.get("input", 0)
                        output_tokens = usage.get("output", 0)

                        if (
                            current_turn.assistant_message_start is not None
                            and current_turn.user_message_end is not None
                        ):
                            current_turn.prompt_processing_time_seconds = (
                                current_turn.assistant_message_start
                                - current_turn.user_message_end
                            )
                            current_turn.time_to_first_token_seconds = (
                                current_turn.prompt_processing_time_seconds
                            )

                        if (
                            current_turn.assistant_message_end is not None
                            and current_turn.assistant_message_start is not None
                        ):
                            current_turn.generation_time_seconds = (
                                current_turn.assistant_message_end
                                - current_turn.assistant_message_start
                            )

                        if (
                            current_turn.prompt_processing_time_seconds is not None
                            and input_tokens > 0
                        ):
                            current_turn.prompt_tokens_per_second = (
                                input_tokens
                                / current_turn.prompt_processing_time_seconds
                            )

                        if (
                            current_turn.generation_time_seconds is not None
                            and output_tokens > 0
                        ):
                            current_turn.generation_tokens_per_second = (
                                output_tokens / current_turn.generation_time_seconds
                            )

                        pending_turn = current_turn
                        current_turn = None
                        assistant_count += 1

                        logger.debug(
                            "Turn %d complete: stop_reason=%s error_message=%s",
                            pending_turn.turn_index,
                            pending_turn.stop_reason,
                            pending_turn.error_message,
                        )
                        if assistant_count >= max_turns:
                            logger.info(
                                "Max turns (%d) reached, terminating pi", max_turns
                            )
                            self._max_turns_reached = True
                            _kill()
                            if pending_turn is not None:
                                yield pending_turn
                            return

                elif event_type == "tool_execution_start":
                    target = current_turn if current_turn is not None else pending_turn
                    if target is not None:
                        target.tool_calls.append(
                            {
                                "toolCallId": event.get("toolCallId"),
                                "toolName": event.get("toolName"),
                                "args": event.get("args"),
                            }
                        )

                elif event_type == "tool_execution_end":
                    target = current_turn if current_turn is not None else pending_turn
                    if target is not None:
                        target.tool_results.append(
                            {
                                "toolCallId": event.get("toolCallId"),
                                "toolName": event.get("toolName"),
                                "result": event.get("result"),
                                "isError": event.get("isError", False),
                            }
                        )

            logger.debug(
                "pi stdout exhausted: seen_assistant_end=%s returncode=%s turns=%d",
                seen_assistant_end,
                proc.returncode if proc else None,
                assistant_count,
            )

            if timed_out and not self._max_turns_reached:
                raise PiDevTimeoutError(
                    f"pi timed out after {self.timeout}s ({assistant_count} turns)"
                )

            if not seen_assistant_end:
                msg = "Premature EOF: no assistant message_end events received"
                rc = proc.returncode if proc else None
                if rc is not None and rc != 0:
                    msg += f" (pi exited with code {rc})"
                raise PiDevError(msg)

            if pending_turn is not None:
                logger.debug(
                    "Yielding final pending turn %d: stop_reason=%s",
                    pending_turn.turn_index,
                    pending_turn.stop_reason,
                )
                yield pending_turn

            if proc and proc.poll() is not None:
                rc = proc.returncode
                if rc != 0 and not terminated_by_us:
                    raise PiDevError(f"pi exited with code {rc}")

            logger.info("pi stream ended after %d turns", assistant_count)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_fd,
                text=True,
                cwd=working_dir,
                preexec_fn=os.setsid,  # noqa: S603,PLW1509 — new process group for killpg
            )

            if self.timeout > 0:
                timer = threading.Timer(self.timeout, _timeout)
                timer.start()

            yield from _read_turns()

        except PiDevError as exc:
            stderr_content = _read_stderr()
            if stderr_content:
                logger.error("pi stderr: %s", stderr_content.strip())
                if exc.stderr is None:
                    exc.stderr = stderr_content
            raise

        finally:
            if timer is not None:
                timer.cancel()
            _cleanup_proc()
            stderr_fd.close()
            if self.stderr_log is None:
                try:
                    stderr_path.unlink(missing_ok=True)
                except Exception:
                    pass
