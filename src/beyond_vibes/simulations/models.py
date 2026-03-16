"""Configuration models for simulations."""

from pydantic import BaseModel, Field, field_validator


class RepositoryConfig(BaseModel):
    """Configuration for a repository to simulate."""

    url: str
    branch: str = "main"


class JudgeMapping(BaseModel):
    """Maps a judge to a specific input artifact."""

    name: str
    input: str = "git_diff"

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        """Validate that input is one of the allowed values."""
        allowed = {"git_diff", "final_message", "trace"}
        if v not in allowed:
            raise ValueError(f"input must be one of {allowed}, got {v}")
        return v


class SimulationConfig(BaseModel):
    """Configuration for a simulation run."""

    name: str
    description: str
    archetype: str
    repository: RepositoryConfig
    prompt: str
    agent: str = "build"
    system_prompt: str | None = None
    max_turns: int = 50
    capture_git_diff: bool = False
    guidelines: list[str] = Field(default_factory=list)
    judges: list[JudgeMapping] = Field(default_factory=list)
