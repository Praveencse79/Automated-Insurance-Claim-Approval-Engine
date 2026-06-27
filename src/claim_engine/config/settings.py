"""Centralised, strongly-typed application configuration.

All runtime configuration is loaded from environment variables (optionally via a
local ``.env`` file). Using :class:`pydantic_settings.BaseSettings` gives us
validation, type-coercion and a single, documented schema for every tunable in
the system.

Design rationale
----------------
* **12-factor config** - configuration lives in the environment, never in code.
* **Fail-fast** - invalid configuration raises at process start-up, not deep
  inside a request.
* **Mock-mode aware** - when cloud credentials are absent the engine flips to a
  deterministic in-memory mode so the whole pipeline is runnable in CI and on a
  laptop without secrets.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvironmentName = Literal["local", "dev", "staging", "prod"]


class Settings(BaseSettings):
    """Typed view over every environment variable the engine understands.

    Instances are cached process-wide via :func:`get_settings`; treat the
    object as immutable at runtime.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CLAIM_ENGINE_",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ runtime
    mock_mode: bool = Field(
        default=True,
        description="Replace all external integrations with in-memory fakes.",
    )
    environment: EnvironmentName = Field(default="local")
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------- claude / llm
    # NOTE: the secret key uses its conventional vendor name, so it is read
    # without the CLAIM_ENGINE_ prefix via an explicit validation alias.
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-3-5-sonnet-20241022")
    claude_max_tokens: int = Field(default=1024, ge=256, le=8192)
    claude_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # ----------------------------------------------------------------- pinecone
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index: str = Field(default="insurance-policy-kb")
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")
    embedding_dimension: int = Field(default=1536, ge=8)

    # ---------------------------------------------------------------- snowflake
    snowflake_account: Optional[str] = Field(default=None, alias="SNOWFLAKE_ACCOUNT")
    snowflake_user: Optional[str] = Field(default=None, alias="SNOWFLAKE_USER")
    snowflake_password: Optional[str] = Field(default=None, alias="SNOWFLAKE_PASSWORD")
    snowflake_warehouse: str = Field(default="CLAIMS_WH", alias="SNOWFLAKE_WAREHOUSE")
    snowflake_database: str = Field(default="INSURANCE", alias="SNOWFLAKE_DATABASE")
    snowflake_schema: str = Field(default="CLAIMS", alias="SNOWFLAKE_SCHEMA")

    # -------------------------------------------------------------- decisioning
    auto_approve_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    manual_review_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    high_value_threshold: float = Field(default=500_000.0, ge=0.0)

    # ----------------------------------------------------------------- helpers
    @property
    def claude_credentials_present(self) -> bool:
        """True when a real Anthropic key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def pinecone_credentials_present(self) -> bool:
        """True when a real Pinecone key is configured."""
        return bool(self.pinecone_api_key)

    @property
    def snowflake_credentials_present(self) -> bool:
        """True when the minimum Snowflake credentials are configured."""
        return bool(self.snowflake_account and self.snowflake_user and self.snowflake_password)

    def resolve_use_mock(self, credentials_present: bool) -> bool:
        """Decide whether a given integration should use its mock implementation.

        An integration is mocked when ``mock_mode`` is explicitly enabled *or*
        when the credentials required to reach the real service are missing.
        This lets a developer run a partially-configured environment (for
        example, real Pinecone but mocked Snowflake) without code changes.
        """
        return self.mock_mode or not credentials_present


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings` singleton.

    The cache guarantees the ``.env`` file is parsed exactly once and that every
    component shares an identical configuration view.
    """
    return Settings()  # type: ignore[call-arg]  # values sourced from env
