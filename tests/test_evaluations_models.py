"""Tests for evaluations models."""

from beyond_vibes.evaluations.models import EvalResult, JudgeInput


class TestJudgeInput:
    """Tests for JudgeInput dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Test JudgeInput creation with all fields."""
        input_data = JudgeInput(
            run_id="test-run-123",
            task_name="unit_tests",
            archetype="repo_maintenance",
            system_prompt="You are an expert...",
            task_prompt="Create unit tests",
            final_message="Tests created",
            git_diff="diff --git a/test.py...",
            trace={"total_messages": 5},
        )

        assert input_data.run_id == "test-run-123"
        assert input_data.git_diff is not None

    def test_optional_git_diff_none(self) -> None:
        """Test git_diff can be None."""
        input_data = JudgeInput(
            run_id="test-run-123",
            task_name="unit_tests",
            archetype="repo_maintenance",
            system_prompt="...",
            task_prompt="Test",
            final_message="Done",
            git_diff=None,
            trace={},
        )

        assert input_data.git_diff is None


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_create_basic(self) -> None:
        """Test EvalResult creation."""
        result = EvalResult(name="guidelines", score=0.85)

        assert result.name == "guidelines"
        assert result.score == 0.85  # noqa: PLR2004
        assert result.rationale is None

    def test_create_with_rationale(self) -> None:
        """Test EvalResult with rationale."""
        result = EvalResult(
            name="guidelines",
            score=0.9,
            rationale="Agent followed all guidelines.",
        )

        assert result.rationale == "Agent followed all guidelines."
