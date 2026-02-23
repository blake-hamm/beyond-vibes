"""S3 settings loaded from environment variables."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class S3Settings(BaseSettings):
    """S3 configuration settings."""

    bucket: str
    endpoint: str
    access_key: str
    secret_key: str

    model_config = SettingsConfigDict(env_prefix="S3_")
