"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass

DEFAULT_ENVIRONMENT = "development"
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings."""

    environment: str = DEFAULT_ENVIRONMENT
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from process environment variables."""
        return cls(
            environment=os.getenv("APP_ENV", DEFAULT_ENVIRONMENT),
            log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )
