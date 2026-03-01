"""Tests for prompts loader."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from beyond_vibes.simulations.models import RepositoryConfig, SimulationConfig
from beyond_vibes.simulations.prompts.loader import (
    build_prompt,
    list_prompts,
    load_prompt,
    load_task_config,
    render_template,
)


class TestRenderTemplate:
    """Tests for render_template function."""

    def test_simple_variable_replacement(self) -> None:
        """Test basic variable replacement."""
        template = "Hello {{name}}!"
        result = render_template(template, {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_variables(self) -> None:
        """Test replacement of multiple variables."""
        template = "{{greeting}} {{name}}, you are {{age}} years old"
        result = render_template(
            template, {"greeting": "Hi", "name": "Alice", "age": "30"}
        )
        assert result == "Hi Alice, you are 30 years old"

    def test_missing_variable_keeps_placeholder(self) -> None:
        """Test that missing variables keep the placeholder."""
        template = "Hello {{name}}, welcome to {{place}}"
        result = render_template(template, {"name": "Alice"})
        assert result == "Hello Alice, welcome to {{place}}"

    def test_empty_variables_dict(self) -> None:
        """Test with empty variables dict."""
        template = "Hello {{name}}"
        result = render_template(template, {})
        assert result == "Hello {{name}}"

    def test_non_string_variable_values(self) -> None:
        """Test that non-string values are converted to strings."""
        template = "Count: {{count}}"
        result = render_template(template, {"count": 42})
        assert result == "Count: 42"

    def test_no_variables_in_template(self) -> None:
        """Test template without variables."""
        template = "Hello World"
        result = render_template(template, {"name": "Alice"})
        assert result == "Hello World"


class TestLoadPrompt:
    """Tests for load_prompt function."""

    def test_load_valid_prompt(self, tmp_path: Path) -> None:
        """Test loading a valid prompt file."""
        prompt_file = tmp_path / "test_task.yaml"
        data = {
            "name": "test-task",
            "description": "A test task",
            "archetype": "test",
            "repository": {"url": "https://github.com/test/repo", "branch": "main"},
            "prompt": "Test prompt with {{variable}}",
            "agent": "build",
            "max_turns": 50,
        }
        prompt_file.write_text(yaml.dump(data))

        config = load_prompt(prompt_file, {"variable": "value"})

        assert isinstance(config, SimulationConfig)
        assert config.name == "test-task"
        assert config.prompt == "Test prompt with value"

    def test_load_prompt_without_variables(self, tmp_path: Path) -> None:
        """Test loading prompt without template variables."""
        prompt_file = tmp_path / "test_task.yaml"
        data = {
            "name": "simple-task",
            "description": "Simple task",
            "archetype": "simple",
            "repository": {"url": "https://github.com/test/repo"},
            "prompt": "Simple prompt",
        }
        prompt_file.write_text(yaml.dump(data))

        config = load_prompt(prompt_file)

        assert config.name == "simple-task"
        assert config.prompt == "Simple prompt"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing file."""
        missing_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            load_prompt(missing_file)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Test ValueError for invalid YAML."""
        prompt_file = tmp_path / "invalid.yaml"
        prompt_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_prompt(prompt_file)

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test ValueError for empty file."""
        prompt_file = tmp_path / "empty.yaml"
        prompt_file.write_text("")

        with pytest.raises(ValueError, match="Empty prompt file"):
            load_prompt(prompt_file)

    def test_invalid_config(self, tmp_path: Path) -> None:
        """Test ValueError for invalid SimulationConfig."""
        prompt_file = tmp_path / "invalid_config.yaml"
        # Missing required fields
        data = {"name": "invalid"}
        prompt_file.write_text(yaml.dump(data))

        with pytest.raises(ValueError, match="Invalid prompt config"):
            load_prompt(prompt_file)


class TestLoadTaskConfig:
    """Tests for load_task_config function."""

    def test_load_task_with_default_dir(self, tmp_path: Path) -> None:
        """Test loading task from custom tasks directory."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_file = tasks_dir / "my_task.yaml"
        data = {
            "name": "my-task",
            "description": "My task",
            "archetype": "custom",
            "repository": {"url": "https://github.com/test/repo"},
            "prompt": "Do {{action}}",
        }
        task_file.write_text(yaml.dump(data))

        config = load_task_config("my_task", '{"action": "something"}', tasks_dir)

        assert config.name == "my-task"
        assert config.prompt == "Do something"

    def test_load_task_invalid_json_vars(self, tmp_path: Path) -> None:
        """Test ValueError for invalid JSON in prompt_vars."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_file = tasks_dir / "test.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "description": "test",
                    "archetype": "test",
                    "repository": {"url": "test"},
                    "prompt": "test",
                }
            )
        )

        with pytest.raises(json.JSONDecodeError):
            load_task_config("test", "invalid json", tasks_dir)

    def test_load_task_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError when task file doesn't exist."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            load_task_config("missing_task", "{}", tasks_dir)

    def test_load_task_empty_json_vars(self, tmp_path: Path) -> None:
        """Test loading with empty JSON vars string."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        task_file = tasks_dir / "empty_vars.yaml"
        data = {
            "name": "empty-vars-task",
            "description": "Task with empty vars",
            "archetype": "test",
            "repository": {"url": "https://github.com/test/repo"},
            "prompt": "Simple prompt",
        }
        task_file.write_text(yaml.dump(data))

        config = load_task_config("empty_vars", "{}", tasks_dir)

        assert config.name == "empty-vars-task"


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_no_system_prompts(self) -> None:
        """Test with no system prompts configured."""
        sim_config = SimulationConfig(
            name="test",
            description="test",
            archetype="test",
            repository=RepositoryConfig(url="https://github.com/test/repo"),
            prompt="Task prompt",
        )

        with patch("beyond_vibes.simulations.prompts.loader.settings") as mock_settings:
            mock_settings.system_prompt = None
            result = build_prompt(sim_config)

        assert result == "Task prompt"

    def test_only_settings_system_prompt(self) -> None:
        """Test with only settings system prompt."""
        sim_config = SimulationConfig(
            name="test",
            description="test",
            archetype="test",
            repository=RepositoryConfig(url="https://github.com/test/repo"),
            prompt="Task prompt",
        )

        with patch("beyond_vibes.simulations.prompts.loader.settings") as mock_settings:
            mock_settings.system_prompt = "Settings system prompt"
            result = build_prompt(sim_config)

        assert result == "Settings system prompt\n\n---\n\nTask prompt"

    def test_only_config_system_prompt(self) -> None:
        """Test with only sim_config system prompt."""
        sim_config = SimulationConfig(
            name="test",
            description="test",
            archetype="test",
            repository=RepositoryConfig(url="https://github.com/test/repo"),
            prompt="Task prompt",
            system_prompt="Config system prompt",
        )

        with patch("beyond_vibes.simulations.prompts.loader.settings") as mock_settings:
            mock_settings.system_prompt = None
            result = build_prompt(sim_config)

        assert result == "Config system prompt\n\n---\n\nTask prompt"

    def test_both_system_prompts(self) -> None:
        """Test with both system prompts - config first, then settings."""
        sim_config = SimulationConfig(
            name="test",
            description="test",
            archetype="test",
            repository=RepositoryConfig(url="https://github.com/test/repo"),
            prompt="Task prompt",
            system_prompt="Config system prompt",
        )

        with patch("beyond_vibes.simulations.prompts.loader.settings") as mock_settings:
            mock_settings.system_prompt = "Settings system prompt"
            result = build_prompt(sim_config)

        expected = (
            "Config system prompt\n\n---\n\n"
            "Settings system prompt\n\n---\n\n"
            "Task prompt"
        )
        assert result == expected


class TestListPrompts:
    """Tests for list_prompts function."""

    def test_list_yaml_files(self, tmp_path: Path) -> None:
        """Test listing YAML files recursively."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "task1.yaml").write_text("name: task1")
        (subdir / "task2.yaml").write_text("name: task2")
        (tmp_path / "not_yaml.txt").write_text("not yaml")

        result = list_prompts(tmp_path)

        expected_file_count = 2
        assert len(result) == expected_file_count
        assert tmp_path / "task1.yaml" in result
        assert subdir / "task2.yaml" in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test listing empty directory."""
        result = list_prompts(tmp_path)
        assert result == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test listing non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"

        result = list_prompts(nonexistent)

        assert result == []

    def test_no_yaml_files(self, tmp_path: Path) -> None:
        """Test directory with no YAML files."""
        (tmp_path / "readme.md").write_text("# README")
        (tmp_path / "script.py").write_text("print('hello')")

        result = list_prompts(tmp_path)

        assert result == []
