"""Hacker News story-buzz collector via the free Algolia API.

For each stage, queries the top 1-2 keywords for stories created in the last
30 days and aggregates story count + total points as a sentiment/attention
proxy. Kept deliberately small (max 1-2 keywords/stage) to conserve request
budget.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from src.http_utils import get_json

logger = logging.getLogger(__name__)

SEARCH_URL = "https://hn.algolia.com/api/v1/search"
LOOKBACK_DAYS = 30
KEYWORDS_PER_STAGE = 2
REQUEST_PAUSE_SECONDS = 1.0


def _since_timestamp() -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp())


def collect(stage_keywords: dict[int, list[str]] | None = None) -> dict[str, Any]:
    """Collect per-stage HN story buzz (story count + total points).

    Returns:
      {
        "source": "hn_buzz",
        "stage_buzz": {stage_id: {"stories": int, "points": int}, ...},
      }
    """
    result: dict[str, Any] = {"source": "hn_buzz", "stage_buzz": {}}

    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("hn_buzz: NXT_OFFLINE=1, skipping network calls")
        return result

    stage_keywords = stage_keywords or {}
    since_ts = _since_timestamp()

    for stage_id, keywords in stage_keywords.items():
        stories = 0
        points = 0
        for keyword in keywords[:KEYWORDS_PER_STAGE]:
            try:
                data = get_json(
                    SEARCH_URL,
                    params={
                        "query": keyword,
                        "tags": "story",
                        "numericFilters": f"created_at_i>{since_ts}",
                        "hitsPerPage": 50,
                    },
                )
                hits = data.get("hits", [])
                stories += len(hits)
                points += sum(hit.get("points") or 0 for hit in hits)
            except Exception as exc:  # noqa: BLE001
                logger.warning("hn_buzz: search failed for stage %s keyword '%s': %s", stage_id, keyword, exc)
            time.sleep(REQUEST_PAUSE_SECONDS)

        result["stage_buzz"][stage_id] = {"stories": stories, "points": points}

    logger.info("hn_buzz: collected buzz for %d stages", len(result["stage_buzz"]))
    return result
