"""GitHub REST collector: developer momentum around AI topics and stages.

Uses the public search API. Works without a token (lower rate limit) and
picks up GITHUB_TOKEN automatically when available (e.g. GitHub Actions
provides one for free via ${{ github.token }}).

Rate limiting: the search endpoint allows ~10 req/min unauthenticated, so we
cap ourselves to roughly 10 search requests per run with a 3s pause between
calls.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from src.http_utils import get_json

logger = logging.getLogger(__name__)

SEARCH_REPOS_URL = "https://api.github.com/search/repositories"
MAX_SEARCH_REQUESTS = 10
REQUEST_PAUSE_SECONDS = 3.0
LOOKBACK_DAYS = 90
TOP_REPOS_PER_QUERY = 10


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _since_date() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")


def collect(stage_keywords: dict[int, list[str]] | None = None) -> dict[str, Any]:
    """Collect trending new AI repos and per-stage "developer heat" scores.

    Returns a compact dict:
      {
        "source": "github_trends",
        "top_new_repos": [{"name", "stars", "description", "topics"}, ...],
        "stage_heat": {stage_id: total_stars_of_top10, ...},
      }
    """
    result: dict[str, Any] = {"source": "github_trends", "top_new_repos": [], "stage_heat": {}}

    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("github_trends: NXT_OFFLINE=1, skipping network calls")
        return result

    since = _since_date()
    requests_made = 0

    # a) trending new AI repos
    try:
        query = f"topic:ai created:>{since}"
        data = get_json(
            SEARCH_REPOS_URL,
            headers=_headers(),
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 20},
        )
        requests_made += 1
        items = data.get("items", []) if isinstance(data, dict) else []
        result["top_new_repos"] = [
            {
                "name": item.get("full_name"),
                "stars": item.get("stargazers_count", 0),
                "description": (item.get("description") or "")[:200],
                "topics": item.get("topics", [])[:8],
            }
            for item in items[:20]
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("github_trends: trending repo search failed: %s", exc)

    time.sleep(REQUEST_PAUSE_SECONDS)
    requests_made += 1  # account for the sleep-gated slot above conceptually

    # b) per-stage developer heat
    stage_keywords = stage_keywords or {}
    for stage_id, keywords in stage_keywords.items():
        if requests_made >= MAX_SEARCH_REQUESTS:
            logger.info("github_trends: reached max search request budget, stopping early")
            break
        if not keywords:
            continue
        keyword = keywords[0]
        try:
            query = f"{keyword} created:>{since} in:name,description"
            data = get_json(
                SEARCH_REPOS_URL,
                headers=_headers(),
                params={"q": query, "sort": "stars", "order": "desc", "per_page": TOP_REPOS_PER_QUERY},
            )
            requests_made += 1
            items = data.get("items", []) if isinstance(data, dict) else []
            total_stars = sum(item.get("stargazers_count", 0) for item in items[:TOP_REPOS_PER_QUERY])
            result["stage_heat"][stage_id] = total_stars
        except Exception as exc:  # noqa: BLE001
            logger.warning("github_trends: stage %s heat search failed: %s", stage_id, exc)
            result["stage_heat"][stage_id] = 0
        time.sleep(REQUEST_PAUSE_SECONDS)

    logger.info(
        "github_trends: %d top repos, heat for %d stages",
        len(result["top_new_repos"]),
        len(result["stage_heat"]),
    )
    return result
