"""Configuration loading for Orion Capital.

All secrets come from environment variables (optionally via a local .env file).
Nothing here ever hard-codes credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional at runtime
    pass


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_CIO_MODEL = "claude-opus-4-8"


@dataclass
class Settings:
    kite_api_key: str | None
    kite_api_secret: str | None
    kite_access_token: str | None
    anthropic_api_key: str | None
    model: str
    cio_model: str

    @property
    def has_kite(self) -> bool:
        return bool(self.kite_api_key and self.kite_access_token)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)


def load_settings() -> Settings:
    return Settings(
        kite_api_key=os.getenv("KITE_API_KEY") or None,
        kite_api_secret=os.getenv("KITE_API_SECRET") or None,
        kite_access_token=os.getenv("KITE_ACCESS_TOKEN") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("ORION_MODEL", DEFAULT_MODEL),
        cio_model=os.getenv("ORION_CIO_MODEL", DEFAULT_CIO_MODEL),
    )
