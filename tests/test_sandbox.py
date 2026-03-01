"""Tests for sandbox management."""

import shutil
from unittest.mock import MagicMock, patch

import pytest
from git import GitCommandError

from beyond_vibes.simulations.sandbox import SandboxManager


class TestSandboxManagerInit:
    """Tests for SandboxManager initialization."""

    def test_init_no_workspace(self) -> None:
        """Test that initialization creates manager with no workspace."""
        manager = SandboxManager()
        assert manager.workspace is None


class TestCreate:
    """Tests for create method."""

    def test_create_workspace(self) -> None:
        """Test creating a temporary workspace."""
        manager = SandboxManager()
        workspace = manager.create()

        assert workspace is not None
        assert workspace.exists()
        assert workspace.name.startswith("beyond_vibes_")
        assert manager.workspace == workspace

    def test_create_multiple_calls(self) -> None:
        """Test that multiple create calls create different workspaces."""
        manager = SandboxManager()
        workspace1 = manager.create()
        workspace2 = manager.create()

        assert workspace1 != workspace2
        assert workspace1.exists()
        assert workspace2.exists()
        assert manager.workspace == workspace2


class TestCloneRepo:
    """Tests for clone_repo method."""

    def test_clone_repo_success(self) -> None:
        """Test successful repository cloning."""
        manager = SandboxManager()
        manager.create()

        with patch("beyond_vibes.simulations.sandbox.Repo.clone_from") as mock_clone:
            mock_clone.return_value = MagicMock()
            result = manager.clone_repo("https://github.com/test/repo", branch="main")

        assert result == manager.workspace
        mock_clone.assert_called_once_with(
            "https://github.com/test/repo",
            manager.workspace,
            branch="main",
            depth=1,
        )

    def test_clone_repo_custom_branch(self) -> None:
        """Test cloning with custom branch."""
        manager = SandboxManager()
        manager.create()

        with patch("beyond_vibes.simulations.sandbox.Repo.clone_from") as mock_clone:
            manager.clone_repo("https://github.com/test/repo", branch="develop")

        mock_clone.assert_called_once_with(
            "https://github.com/test/repo",
            manager.workspace,
            branch="develop",
            depth=1,
        )

    def test_clone_repo_without_create(self) -> None:
        """Test that cloning without creating workspace raises error."""
        manager = SandboxManager()

        with pytest.raises(RuntimeError, match="Workspace not created"):
            manager.clone_repo("https://github.com/test/repo")

    def test_clone_repo_git_error(self) -> None:
        """Test handling of GitCommandError."""
        manager = SandboxManager()
        manager.create()

        with patch("beyond_vibes.simulations.sandbox.Repo.clone_from") as mock_clone:
            mock_clone.side_effect = GitCommandError("clone", "Connection failed")

            with pytest.raises(RuntimeError, match="Failed to clone"):
                manager.clone_repo("https://github.com/test/repo")


class TestCleanup:
    """Tests for cleanup method."""

    def test_cleanup_removes_workspace(self) -> None:
        """Test that cleanup removes the workspace."""
        manager = SandboxManager()
        workspace = manager.create()
        assert workspace.exists()

        manager.cleanup()

        assert not workspace.exists()
        assert manager.workspace is None

    def test_cleanup_no_workspace(self) -> None:
        """Test cleanup when workspace is None."""
        manager = SandboxManager()
        # Should not raise
        manager.cleanup()
        assert manager.workspace is None

    def test_cleanup_already_deleted(self) -> None:
        """Test cleanup when workspace is already deleted."""
        manager = SandboxManager()
        workspace = manager.create()
        # Delete manually
        shutil.rmtree(workspace)

        # Should not raise, just log warning
        manager.cleanup()
        assert manager.workspace is None

    def test_cleanup_oserror(self) -> None:
        """Test cleanup when OSError occurs."""
        manager = SandboxManager()
        manager.create()

        with patch("beyond_vibes.simulations.sandbox.shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = OSError("Permission denied")
            # Should not raise, just log warning
            manager.cleanup()

        assert manager.workspace is None


class TestSandboxContextManager:
    """Tests for sandbox context manager."""

    def test_sandbox_without_clone(self) -> None:
        """Test sandbox context manager without cloning."""
        manager = SandboxManager()

        with manager.sandbox() as workspace:
            assert workspace is not None
            assert workspace.exists()
            assert manager.workspace == workspace

        # After exiting context, workspace should be cleaned up
        assert not workspace.exists()
        assert manager.workspace is None

    def test_sandbox_with_clone(self) -> None:
        """Test sandbox context manager with cloning."""
        manager = SandboxManager()

        with patch("beyond_vibes.simulations.sandbox.Repo.clone_from") as mock_clone:
            mock_clone.return_value = MagicMock()

            with manager.sandbox(
                url="https://github.com/test/repo", branch="main"
            ) as workspace:
                assert workspace is not None
                mock_clone.assert_called_once()

        # After exiting context, workspace should be cleaned up
        assert manager.workspace is None

    def test_sandbox_cleanup_on_exception(self) -> None:
        """Test that sandbox is cleaned up even when exception occurs."""
        manager = SandboxManager()
        workspace_path = None

        def raise_test_error() -> None:
            nonlocal workspace_path
            with manager.sandbox() as workspace:
                workspace_path = workspace
                assert workspace is not None
                assert workspace.exists()
                raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            raise_test_error()

        # Workspace should still be cleaned up
        assert workspace_path is not None
        assert not workspace_path.exists()
        assert manager.workspace is None

    def test_sandbox_default_branch(self) -> None:
        """Test that default branch is 'main'."""
        manager = SandboxManager()
        captured_workspace = None

        with patch("beyond_vibes.simulations.sandbox.Repo.clone_from") as mock_clone:
            with manager.sandbox(url="https://github.com/test/repo") as workspace:
                captured_workspace = workspace

        mock_clone.assert_called_once_with(
            "https://github.com/test/repo",
            captured_workspace,
            branch="main",
            depth=1,
        )


class TestPathProperty:
    """Tests for path property."""

    def test_path_returns_workspace(self) -> None:
        """Test that path property returns current workspace."""
        manager = SandboxManager()
        assert manager.path is None

        workspace = manager.create()
        assert manager.path == workspace

        manager.cleanup()
        assert manager.path is None
