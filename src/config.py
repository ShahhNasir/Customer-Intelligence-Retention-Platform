"""
Centralized, typed application configuration. Reads from .env once;
every other module imports `settings` from here instead of touching
os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    llm_provider: str = "groq"
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
