from functools import lru_cache

from pydantic import Field
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
    # Shared secret for verifying the customer JWTs the storefront issues
    # (HS256). Required, no default: a missing secret must fail at startup,
    # not fall back to something guessable. 32+ characters, the minimum
    # RFC 7518 recommends for an HS256 key — a short one fails here too,
    # rather than quietly weakening every token.
    jwt_secret: str = Field(min_length=32)
    # The `aud` claim a token must carry — so a token the storefront issued
    # for some other service can't be replayed against this one.
    jwt_audience: str = "agent-service"


@lru_cache
def get_settings() -> Settings:
    # Cached so Settings() — which reads the environment and .env file — only
    # runs once per process, not on every request that needs config.
    return Settings()
