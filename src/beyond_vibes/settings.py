"""Application settings loaded from environment variables."""

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
    opencode_url: str = "http://127.0.0.1:4096"
    system_prompt: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: str | None = None
    mlflow_enable_system_metrics_logging: bool = True
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

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        """Accept both uppercase and lowercase log levels."""
        if isinstance(v, str):
            return v.upper()
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
