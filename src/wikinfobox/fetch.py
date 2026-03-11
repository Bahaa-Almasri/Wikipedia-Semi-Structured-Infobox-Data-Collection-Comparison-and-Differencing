from __future__ import annotations

import time
from typing import Optional

import requests

from .config import WIKIPEDIA


class HttpError(RuntimeError):
    pass


def get(url: str, *, timeout: Optional[int] = None) -> str:
    """HTTP GET with basic retry logic, returns response text."""
    timeout = timeout or WIKIPEDIA.request_timeout

    session = requests.Session()
    session.headers.update({"User-Agent": WIKIPEDIA.user_agent})

    last_exc: Exception | None = None
    for attempt in range(1, WIKIPEDIA.max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code >= 400:
                raise HttpError(f"GET {url} failed with status {resp.status_code}")
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == WIKIPEDIA.max_retries:
                break
            sleep_for = WIKIPEDIA.backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_for)

    raise HttpError(f"Failed to fetch {url}") from last_exc

