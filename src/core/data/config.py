"""
Configuration. All values are loaded from environment variables.
See example.env for keys and default values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class WikipediaConfig:
    base_url: str
    un_member_states_url: str
    request_timeout: int
    max_retries: int
    backoff_factor: float
    user_agent: str

    @classmethod
    def from_env(cls) -> "WikipediaConfig":
        return cls(
            base_url=_env(
                "WIKIPEDIA_BASE_URL",
                "https://en.wikipedia.org",
            ),
            un_member_states_url=_env(
                "WIKIPEDIA_UN_MEMBER_STATES_URL",
                "https://en.wikipedia.org/wiki/Member_states_of_the_United_Nations",
            ),
            request_timeout=_env_int("WIKIPEDIA_REQUEST_TIMEOUT", 15),
            max_retries=_env_int("WIKIPEDIA_MAX_RETRIES", 3),
            backoff_factor=_env_float("WIKIPEDIA_BACKOFF_FACTOR", 0.5),
            user_agent=_env(
                "WIKIPEDIA_USER_AGENT",
                "WikipediaInfoboxCollector/1.0 (student academic project)",
            ),
        )


@dataclass(frozen=True)
class MongoConfig:
    """MongoDB connection settings. MONGODB_URI is required for storage."""

    uri: str
    database: str
    collection: str

    @classmethod
    def from_env(cls) -> "MongoConfig":
        return cls(
            uri=_env("MONGODB_URI", ""),
            database=_env("MONGODB_DATABASE", "wikinfobox"),
            collection=_env("MONGODB_COLLECTION", "countries"),
        )


WIKIPEDIA = WikipediaConfig.from_env()
MONGO = MongoConfig.from_env()
