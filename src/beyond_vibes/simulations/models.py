"""Configuration models for simulations."""

from pydantic import BaseModel


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
