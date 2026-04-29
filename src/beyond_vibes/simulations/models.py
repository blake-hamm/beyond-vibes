"""Configuration models for simulations."""

from pydantic import BaseModel, Field


class RepositoryConfig(BaseModel):
    """Configuration for a repository to simulate."""

    url: str
    branch: str = "main"


class SimulationConfig(BaseModel):
    """Configuration for a simulation run."""

    name: str
    description: str
    archetype: str
    repository: RepositoryConfig
    prompt: str
    system_prompt: str | None = None
    max_turns: int = 50
    capture_git_diff: bool = False
    guidelines: dict[str, str] = Field(default_factory=dict)
