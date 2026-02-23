"""Application settings loaded from environment variables."""

from typing import Literal

from dotenv import load_dotenv
from pydantic import field_validator
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
