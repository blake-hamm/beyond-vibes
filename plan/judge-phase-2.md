# Judge Phase 2: Core Data Models and Settings

## Overview

This phase establishes the foundational data models that the entire evaluation system depends on. We need to define:
1. Judge mapping configuration in task configs
2. Judge registry schema
3. Settings for judge model configuration

These are **blocking dependencies** for all subsequent phases.

## Prerequisites

- ✅ Phase 1: Trace artifact (`trace_session.json`) is logged by simulation
- Existing `SimulationConfig` model in `src/beyond_vibes/simulations/models.py`
- Existing `Settings` class in `src/beyond_vibes/settings.py`

## Design Principles

- Follow existing conventions in the codebase
- POC stage: breaking changes are acceptable
- Use the loader pattern from `simulations/prompts/loader.py` for YAML handling

## Changes Required

### 1. Update SimulationConfig with JudgeMapping

**File**: `src/beyond_vibes/simulations/models.py`

**Upstream Dependencies**:
- Task YAML loader must support new nested structure

**Downstream Impact**:
- `judge_factory.py` (Phase 5) reads `task_config.judges` list
- `runner.py` (Phase 6) iterates over judge mappings
- Task YAML files (Phase 7) use this schema

**Implementation**:

```python
class JudgeMapping(BaseModel):
    """Maps a judge to a specific input artifact.
    
    Used in SimulationConfig to declare which judges run on which artifacts.
    
    Attributes:
        name: Judge name from registry (e.g., "guidelines", "tool_efficiency")
        input: Which artifact to evaluate ("git_diff" or "final_message")
    """
    name: str
    input: str = "git_diff"  # git_diff or final_message

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str) -> str:
        allowed = {"git_diff", "final_message"}
        if v not in allowed:
            raise ValueError(f"input must be one of {allowed}, got {v}")
        return v


class SimulationConfig(BaseModel):
    """Task simulation configuration with evaluation support."""
    name: str
    description: str
    archetype: str
    repository: RepositoryConfig
    prompt: str
    agent: str = "build"
    system_prompt: str | None = None
    max_turns: int = 50
    capture_git_diff: bool = False
    
    # Evaluation fields
    guidelines: list[str] = Field(default_factory=list)
    judges: list[JudgeMapping] = Field(default_factory=list)
```

**Testing**:
```python
def test_judge_mapping_validation():
    """Test that invalid input values are rejected."""
    with pytest.raises(ValueError):
        JudgeMapping(name="guidelines", input="invalid")
    
    # Valid inputs
    JudgeMapping(name="guidelines", input="git_diff")
    JudgeMapping(name="tool_efficiency", input="final_message")

def test_simulation_config_with_judges():
    """Test that SimulationConfig accepts judge mappings."""
    config = SimulationConfig(
        name="test",
        description="Test task",
        archetype="repo_maintenance",
        repository=RepositoryConfig(url="http://example.com"),
        prompt="Test prompt",
        judges=[JudgeMapping(name="guidelines", input="git_diff")],
        guidelines=["Do X", "Don't Y"],
    )
    assert len(config.judges) == 1
    assert config.judges[0].name == "guidelines"
    assert config.guidelines == ["Do X", "Don't Y"]
```

---

### 2. Create Judges Registry

**File**: `src/beyond_vibes/evaluations/judges.yaml`

**Upstream Dependencies**:
- None (this is a new standalone file)

**Downstream Impact**:
- `judge_factory.py` (Phase 5) loads and parses this registry
- Must be valid YAML with specific schema

**Implementation**:

```yaml
# Judges Registry
# Defines all available judges that can be referenced in task YAML files
# 
# Schema:
#   judge_name:
#     type: mlflow.builtin | mlflow.third_party | custom
#     class: ClassName                    # Required for builtin/third_party
#     module: module.path                 # Required for third_party
#     factory: dotted.path.to.factory     # Required for custom
#     description: "Human readable description"

judges:
  # Built-in MLflow judges
  guidelines:
    type: mlflow.builtin
    class: Guidelines
    description: "Evaluates if response meets task-specific criteria"
    
  tool_efficiency:
    type: mlflow.builtin
    class: ToolCallEfficiency
    description: "Detects redundant tool calls and loops"
    
  # Third-party DeepEval judges (POC - using git_diff as context)
  faithfulness:
    type: mlflow.third_party
    module: mlflow.metrics.deepeval
    class: Faithfulness
    description: "POC: Measures if git diff is faithful to task (using diff as context)"
    
  answer_relevancy:
    type: mlflow.third_party
    module: mlflow.metrics.deepeval
    class: AnswerRelevancy
    description: "POC: Measures if git diff is relevant to task"

# Future: Custom judges can be added here
# custom_judge_name:
#   type: custom
#   factory: beyond_vibes.evaluations.custom:create_custom_judge
#   description: "Custom evaluation logic"
```

**Validation**:
- All `type` values must be one of: `mlflow.builtin`, `mlflow.third_party`, `custom`
- `mlflow.builtin` requires `class` field
- `mlflow.third_party` requires `module` and `class` fields
- `custom` requires `factory` field

**Note on Testing**:
Following the pattern in `simulations/prompts/loader.py`, we'll create a Pydantic model for the registry and test the model validation instead of testing YAML directly. The loader utility will be implemented in Phase 5.

---

### 3. Update Settings with Judge Configuration

**File**: `src/beyond_vibes/settings.py`

**Upstream Dependencies**:
- Existing Settings class with pydantic-settings
- Follows existing naming convention (no prefix)

**Downstream Impact**:
- `judge_factory.py` (Phase 5) uses `settings.judge_model` as default
- `runner.py` (Phase 6) uses all judge settings
- CLI (Phase 7) can override via `--judge-model` flag

**Implementation**:

```python
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    # ... existing settings ...
    
    # Judge configuration
    judge_model: str = Field(
        default="openai:/gpt-4o-mini",
        description="Judge LLM model (OpenAI format, supports litellm)",
    )
    judge_api_key: str | None = Field(
        default=None,
        description="API key for judge model (OpenRouter, etc.)",
    )
    judge_base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible endpoint (litellm, vLLM)",
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
```

**Environment Variable Mapping**:

| Setting | Env Var | Default |
|---------|---------|---------|
| `judge_model` | `JUDGE_MODEL` | `openai:/gpt-4o-mini` |
| `judge_api_key` | `JUDGE_API_KEY` | `None` |
| `judge_base_url` | `JUDGE_BASE_URL` | `None` |

**Testing**:
```python
def test_judge_settings_defaults():
    """Test default judge settings."""
    settings = Settings()
    assert settings.judge_model == "openai:/gpt-4o-mini"
    assert settings.judge_api_key is None
    assert settings.judge_base_url is None

def test_judge_settings_from_env(monkeypatch):
    """Test loading judge settings from environment."""
    monkeypatch.setenv("JUDGE_MODEL", "openai:/gpt-4o")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:8000/v1")
    
    settings = Settings()
    assert settings.judge_model == "openai:/gpt-4o"
    assert settings.judge_base_url == "http://localhost:8000/v1"
```

**Configuration Examples**:

```bash
# OpenAI (default)
export JUDGE_MODEL="openai:/gpt-4o-mini"

# OpenRouter
export JUDGE_MODEL="openai:/gpt-4o"
export JUDGE_BASE_URL="https://openrouter.ai/api/v1"
export JUDGE_API_KEY="sk-or-..."

# Local via litellm
export JUDGE_MODEL="openai:/mistral-7b"
export JUDGE_BASE_URL="http://localhost:8000/v1"
export JUDGE_API_KEY="dummy-key"
```

---

## Dependencies

### Required Dependencies (add to pyproject.toml)

```toml
dependencies = [
    # ... existing ...
    "pydantic>=2.0.0",  # Already have, but ensure for validators
]
```

### Phase Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| Phase 3 | Phase 2 | Models and settings must exist before creating evaluators module |
| Phase 4 | Phase 2 | Extractor imports JudgeMapping and uses settings |
| Phase 5 | Phase 2 | Judge factory reads judges.yaml registry |
| Phase 6 | Phase 2 | Runner uses settings for default model |
| Phase 7 | Phase 2 | Task YAMLs use JudgeMapping schema |

---

## Success Criteria

- [ ] `JudgeMapping` model validates input field (only allows git_diff/final_message)
- [ ] `SimulationConfig` has `guidelines` and `judges` fields with empty defaults
- [ ] `judges.yaml` registry file exists with valid schema
- [ ] Settings class has `judge_model`, `judge_api_key`, `judge_base_url` fields
- [ ] Environment variables `JUDGE_*` load correctly into settings
- [ ] All tests pass for models and settings

---

## Next Phase

**Phase 3**: Create Evaluations Module Structure
- Depends on: Phase 2 (this phase)
- Creates: `src/beyond_vibes/evaluations/` directory and `__init__.py`
