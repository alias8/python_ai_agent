from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime config, populated from the environment (or a .env file).

    Unlike a plain os.environ.get() call, BaseSettings actually validates
    these at startup — a missing anthropic_api_key fails immediately with a
    clear error rather than surfacing later as a confusing 401 from Claude.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str
    claude_model: str = "claude-sonnet-5"
    # Matches docker-compose.yml's local Postgres (port 5433, not 5432).
    database_url: str = "postgresql+psycopg://agent:agent@localhost:5433/agent_service"


@lru_cache
def get_settings() -> Settings:
    # Cached so Settings() — which reads the environment and .env file — only
    # runs once per process, not on every request that needs config.
    return Settings()
