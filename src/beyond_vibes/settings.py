"""Application settings loaded from environment variables."""

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Main application settings."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    s3_bucket: str
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str
    hf_token: str | None = None
    system_prompt: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None
    mlflow_enable_system_metrics_logging: bool = True
    judge_model: str = Field(
        default="openai:/openai/gpt-4o",
        description=(
            "Judge LLM model (OpenAI format for OpenRouter, "
            "e.g., 'openai:/openai/gpt-4o')"
        ),
    )
    judge_api_key: str | None = Field(
        default=None,
        description="API key for judge model (OpenRouter, etc.)",
    )
    judge_base_url: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible endpoint (litellm, vLLM)",
    )

    # Evaluation truncation settings
    eval_trace_keep_first: int = Field(
        default=3,
        description="Number of trace messages to keep from start",
    )
    eval_trace_keep_last: int = Field(
        default=3,
        description="Number of trace messages to keep from end",
    )
    eval_git_diff_max_chars: int = Field(
        default=30000,
        description="Maximum characters to retain from git diff",
    )
    eval_context_diff_max_chars: int = Field(
        default=15000,
        description="Maximum characters for git diff when used as context",
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Accept both uppercase and lowercase log levels."""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("judge_api_key", "judge_base_url", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str | None) -> str | None:
        """Convert empty strings to None for optional fields."""
        if v == "":
            return None
        return v

    def model_post_init(self, __context: object, /) -> None:
        """Automatically set OpenAI environment variables from judge settings.

        MLflow judges look for OPENAI_API_KEY and OPENAI_BASE_URL directly.
        This ensures they're set from our JUDGE_* settings for OpenRouter.
        """
        # Set OPENAI_API_KEY from JUDGE_API_KEY if provided
        if self.judge_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.judge_api_key

        # Set OPENAI_BASE_URL from JUDGE_BASE_URL if provided
        if self.judge_base_url and not os.environ.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = self.judge_base_url

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
