"""Analyze pi.dev JSONL output for derivable timing/latency metrics."""

# ruff: noqa: T201

import json
import subprocess
import time
from pathlib import Path

_MIN_ASSISTANT_MESSAGES = 2


def _show_event_sequence(events: list[dict]) -> None:
    """Print all event types in order."""
    print("\nEvent sequence:")
    for idx, ev in enumerate(events):
        ev_type = ev.get("type", "unknown")
        ts = ev.get("timestamp")
        msg = ev.get("message", {})
        msg_ts = msg.get("timestamp")
        print(f"  [{idx:2d}] {ev_type:25s}  event.ts={ts!s:30s}  msg.ts={msg_ts!s:15s}")


def _show_message_timestamps(events: list[dict]) -> None:
    """Extract and print message-level timestamps."""
    print("\nMessage-level timestamps (from message objects):")
    msg_events = [
        (idx, ev)
        for idx, ev in enumerate(events)
        if ev.get("type") in ("message_start", "message_end")
    ]
    for idx, ev in msg_events:
        msg = ev.get("message", {})
        role = msg.get("role", "?")
        ts = msg.get("timestamp")
        ev_type = ev.get("type")
        print(f"  [{idx:2d}] {ev_type:12s} role={role:10s} ts={ts}")


def _show_assistant_deltas(events: list[dict]) -> None:
    """Compute and print deltas between consecutive assistant messages."""
    assistant_ts = []
    for idx, ev in enumerate(events):
        if ev.get("type") == "message_end":
            msg = ev.get("message", {})
            if msg.get("role") == "assistant":
                assistant_ts.append((idx, msg.get("timestamp"), ev.get("usage", {})))

    print("\nAssistant message_end timestamps:")
    for idx, ts, usage in assistant_ts:
        print(
            f"  [{idx:2d}] ts={ts}"
            f"  input={usage.get('input')}"
            f"  output={usage.get('output')}"
        )

    if len(assistant_ts) >= _MIN_ASSISTANT_MESSAGES:
        print("\nDelta between consecutive assistant messages:")
        for j in range(1, len(assistant_ts)):
            prev_idx, prev_ts, _ = assistant_ts[j - 1]
            curr_idx, curr_ts, _ = assistant_ts[j]
            delta = curr_ts - prev_ts
            print(f"  [{prev_idx}] -> [{curr_idx}]  {delta} ms")


def _show_usage_structure(events: list[dict]) -> None:
    """Show usage block structure from the first assistant message_end."""
    print("\nUsage block structure (from first assistant message_end):")
    for ev in events:
        if ev.get("type") == "message_end":
            msg = ev.get("message", {})
            if msg.get("role") == "assistant" and "usage" in msg:
                print(json.dumps(msg.get("usage"), indent=2))
                break


def analyze_fixture(path: Path) -> None:
    """Read a JSONL fixture and print timing-related fields per event."""
    print(f"\n{'=' * 60}")
    print(f"Fixture: {path.name}")
    print(f"{'=' * 60}")

    events = []
    with path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                events.append(json.loads(stripped))

    _show_event_sequence(events)
    _show_message_timestamps(events)
    _show_assistant_deltas(events)
    _show_usage_structure(events)


def capture_live_output(prompt: str = "Say hello") -> list[dict]:
    """Run pi --mode json and capture events with wall-clock timing."""
    print(f"\n{'=' * 60}")
    print(f"Live capture: '{prompt}'")
    print(f"{'=' * 60}")

    cmd = [
        "pi",
        "--mode",
        "json",
        "--no-session",
        "--provider",
        "kimi-coding",
        "--model",
        "kimi-for-coding",
        "--print",
        prompt,
    ]

    events = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    t0 = time.perf_counter()
    for raw_line in proc.stdout:  # type: ignore[union-attr]
        t_recv = time.perf_counter() - t0
        stripped = raw_line.strip()
        if stripped:
            try:
                ev = json.loads(stripped)
                ev["_wall_clock_s"] = round(t_recv, 4)
                events.append(ev)
            except json.JSONDecodeError:
                pass

    proc.wait()
    return events


def _print_event_table(events: list[dict]) -> None:
    """Pretty-print live-captured events with wall-clock timing."""
    print("\nLive event stream with wall-clock arrival times:")
    print(
        f"{'idx':>4}  {'event_type':25s}"
        f"  {'wall_clock_s':>12s}"
        f"  {'msg.ts':>15s}"
        f"  {'delta_ms':>10s}"
    )
    print("-" * 80)

    prev_ts = None
    for idx, ev in enumerate(events):
        ev_type = ev.get("type", "unknown")
        wall = ev.get("_wall_clock_s", 0.0)
        msg = ev.get("message", {})
        msg_ts = msg.get("timestamp")

        delta_str = ""
        if msg_ts is not None and prev_ts is not None:
            delta = msg_ts - prev_ts
            delta_str = f"{delta}"
            prev_ts = msg_ts
        elif msg_ts is not None:
            prev_ts = msg_ts

        print(
            f"{idx:4d}  {ev_type:25s}"
            f"  {wall:12.4f}"
            f"  {str(msg_ts):>15s}"
            f"  {delta_str:>10s}"
        )


def _print_inter_event_latencies(events: list[dict]) -> None:
    """Compute wall-clock deltas between events."""
    print("\nWall-clock inter-event latencies:")
    for idx in range(1, len(events)):
        prev = events[idx - 1]["_wall_clock_s"]
        curr = events[idx]["_wall_clock_s"]
        delta_ms = (curr - prev) * 1000
        ev_type = events[idx].get("type", "unknown")
        prev_type = events[idx - 1].get("type", "?")
        print(f"  {prev_type:25s} -> {ev_type:25s}  {delta_ms:8.2f} ms")


def _print_generation_timing(events: list[dict]) -> None:
    """Identify assistant message boundaries and generation timing."""
    print("\nAssistant generation timing (wall-clock):")
    start_idx = None
    for idx, ev in enumerate(events):
        if ev.get("type") == "message_start":
            msg = ev.get("message", {})
            if msg.get("role") == "assistant":
                start_idx = idx
        elif ev.get("type") == "message_end" and start_idx is not None:
            msg = ev.get("message", {})
            if msg.get("role") == "assistant":
                end_t = ev["_wall_clock_s"]
                start_t = events[start_idx]["_wall_clock_s"]
                duration_ms = (end_t - start_t) * 1000
                usage = msg.get("usage", {})
                out_tok = usage.get("output", 0)
                tps = (out_tok / duration_ms * 1000) if duration_ms > 0 else 0
                print(
                    f"  turn msg_start[{start_idx}]"
                    f" -> msg_end[{idx}]:"
                    f" {duration_ms:.2f} ms"
                )
                print(f"    output_tokens={out_tok}  =>  TPS={tps:.2f}")
                start_idx = None


def print_live_events(events: list[dict]) -> None:
    """Pretty-print live-captured events with wall-clock timing."""
    _print_event_table(events)
    _print_inter_event_latencies(events)
    _print_generation_timing(events)


def main() -> None:
    """Run analysis against local fixtures."""
    fixtures = [
        Path("tests/fixtures/pi_dev_output.jsonl"),
        Path("tests/fixtures/pi_dev_tool_output.jsonl"),
    ]

    for f in fixtures:
        if f.exists():
            analyze_fixture(f)
        else:
            print(f"Fixture not found: {f}")

    # Uncomment to do a live capture (requires API key / pi auth)
    # live_events = capture_live_output("Say hello in exactly one word")
    # print_live_events(live_events)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
