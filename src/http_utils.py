"""Small shared HTTP helper used by every collector and API client.

Centralizes timeout, User-Agent header, and a tiny retry-with-backoff policy
so individual modules stay short and consistent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 1.5
DEFAULT_USER_AGENT = "nxt-lvl-scanner/1.0 (pcctradinginc@gmail.com)"


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> requests.Response:
    """Perform an HTTP request with a default User-Agent, timeout, and retries.

    Raises the last encountered exception if all attempts fail — callers
    (collectors, API clients) are expected to catch broadly and degrade
    gracefully.
    """
    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)

    last_exc: Exception | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=merged_headers,
                params=params,
                json=json_body,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - deliberately broad, caller decides
            last_exc = exc
            if attempt < attempts:
                logger.warning(
                    "Request %s %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    method,
                    url,
                    attempt,
                    attempts,
                    exc,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
            else:
                logger.warning(
                    "Request %s %s failed after %d attempts: %s", method, url, attempts, exc
                )
    assert last_exc is not None
    raise last_exc


def get_json(url: str, **kwargs: Any) -> Any:
    """Convenience wrapper: GET a URL and parse the response body as JSON."""
    response = request_json("GET", url, **kwargs)
    return response.json()


def get_text(url: str, **kwargs: Any) -> str:
    """Convenience wrapper: GET a URL and return the raw response text."""
    response = request_json("GET", url, **kwargs)
    return response.text
