"""Sandbox management for simulations - temp directories and git clone."""

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from git import GitCommandError, Repo

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages temporary sandbox directories for simulation runs."""

    def __init__(self) -> None:
        """Initialize sandbox with no workspace."""
        self.workspace: Path | None = None

    def create(self) -> Path:
        """Create a temporary workspace directory."""
        self.workspace = Path(tempfile.mkdtemp(prefix="beyond_vibes_"))
        logger.debug(f"Created sandbox workspace: {self.workspace}")
        return self.workspace

    def clone_repo(self, url: str, branch: str = "main") -> Path:
        """Clone a git repository into the sandbox workspace."""
        if self.workspace is None:
            raise RuntimeError("Workspace not created. Call create() first.")

        try:
            Repo.clone_from(
                url,
                self.workspace,
                branch=branch,
                depth=1,
            )
            logger.info(f"Cloned {url} (branch: {branch}) into {self.workspace}")
            return self.workspace
        except GitCommandError as e:
            logger.error(f"Git clone failed: {e}")
            raise RuntimeError(f"Failed to clone {url}: {e}") from e

    def cleanup(self) -> None:
        """Remove the sandbox workspace if it exists."""
        if self.workspace is not None and self.workspace.exists():
            try:
                shutil.rmtree(self.workspace)
                logger.debug(f"Cleaned up sandbox: {self.workspace}")
            except OSError as e:
                logger.warning(f"Failed to cleanup sandbox {self.workspace}: {e}")
        self.workspace = None

    @contextmanager
    def sandbox(
        self, url: str | None = None, branch: str = "main"
    ) -> Generator[Path | None, None, None]:
        """Context manager that creates, optionally clones, and cleans up workspace."""
        self.create()
        try:
            if url:
                self.clone_repo(url, branch)
            yield self.workspace
        finally:
            self.cleanup()

    @property
    def path(self) -> Path | None:
        """Get the current workspace path."""
        return self.workspace
