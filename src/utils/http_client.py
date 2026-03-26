from __future__ import annotations

import time
from typing import Dict, Optional

import requests


class HttpError(RuntimeError):
    pass


def get(
    url: str,
    *,
    timeout: Optional[int] = 15,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    backoff_factor: float = 0.5,
) -> str:
    """HTTP GET with basic retry logic, returns response text."""
    session = requests.Session()
    if headers:
        session.headers.update(headers)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout or 15)
            if resp.status_code >= 400:
                raise HttpError(f"GET {url} failed with status {resp.status_code}")
            return resp.text
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_for = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_for)

    raise HttpError(f"Failed to fetch {url}") from last_exc
