"""Manual test script for Phase 1: PiDevClient.

Run this to confirm pi.dev JSONL events are parsed into TurnData correctly.

Usage:
    uv run python scripts/test_phase1.py
"""

import logging

from beyond_vibes.simulations.pi_dev import PiDevClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRUNCATE_LEN = 100


def main() -> None:
    """Run a manual test of PiDevClient with a simple prompt."""
    client = PiDevClient()
    prompt = "Say hello in one word"

    logger.info("Spawning pi with prompt: %r", prompt)
    logger.info("-" * 50)

    for turn in client.run(prompt, max_turns=1):
        logger.info("Turn %d:", turn.turn_index)
        logger.info("  stop_reason: %s", turn.stop_reason)
        logger.info("  usage: %s", turn.usage)
        logger.info("  content blocks: %d", len(turn.content))
        for block in turn.content:
            block_type = block.get("type", "unknown")
            text = block.get("text", "")
            if len(text) > TRUNCATE_LEN:
                text = text[:TRUNCATE_LEN] + "..."
            logger.info("    [%s] %s", block_type, text)
        logger.info("-" * 50)

    logger.info("Done. Subprocess reaped cleanly.")


if __name__ == "__main__":
    main()
