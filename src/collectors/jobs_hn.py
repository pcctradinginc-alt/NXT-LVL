"""Hacker News "Who is hiring?" collector via the free Algolia HN API.

Counts stage keyword mentions across the comments of the most recent thread
and compares against the second-most-recent (previous month) thread to
derive a month-over-month change per stage — a proxy for what companies are
actively hiring for.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.http_utils import get_json

logger = logging.getLogger(__name__)

SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"
MAX_COMMENTS = 2000
HITS_PER_PAGE = 1000


def _find_hiring_threads(limit: int = 2) -> list[dict[str, Any]]:
    data = get_json(
        SEARCH_BY_DATE_URL,
        params={
            "query": '"who is hiring"',
            "tags": "story,author_whoishiring",
            "hitsPerPage": limit,
        },
    )
    return data.get("hits", [])[:limit]


def _fetch_comments(story_id: str, max_comments: int = MAX_COMMENTS) -> list[str]:
    texts: list[str] = []
    page = 0
    while len(texts) < max_comments:
        data = get_json(
            SEARCH_BY_DATE_URL,
            params={
                "tags": f"comment,story_{story_id}",
                "hitsPerPage": HITS_PER_PAGE,
                "page": page,
            },
        )
        hits = data.get("hits", [])
        if not hits:
            break
        for hit in hits:
            text = hit.get("comment_text") or ""
            if text:
                texts.append(text)
        if len(hits) < HITS_PER_PAGE:
            break
        page += 1
        if page > (max_comments // HITS_PER_PAGE) + 1:
            break
    return texts[:max_comments]


def _count_keywords(comments: list[str], stage_keywords: dict[int, list[str]]) -> dict[int, int]:
    counts: dict[int, int] = {stage_id: 0 for stage_id in stage_keywords}
    lowered = [c.lower() for c in comments]
    for stage_id, keywords in stage_keywords.items():
        total = 0
        for keyword in keywords:
            kw = keyword.lower()
            total += sum(1 for c in lowered if kw in c)
        counts[stage_id] = total
    return counts


def collect(stage_keywords: dict[int, list[str]] | None = None) -> dict[str, Any]:
    """Collect per-stage job-posting keyword counts plus month-over-month change.

    Returns:
      {
        "source": "jobs_hn",
        "stage_job_counts": {stage_id: count, ...},
        "stage_job_mom_change": {stage_id: delta, ...},
        "total_comments": int,
      }
    """
    result: dict[str, Any] = {
        "source": "jobs_hn",
        "stage_job_counts": {},
        "stage_job_mom_change": {},
        "total_comments": 0,
    }

    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("jobs_hn: NXT_OFFLINE=1, skipping network calls")
        return result

    stage_keywords = stage_keywords or {}

    try:
        threads = _find_hiring_threads(limit=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_hn: failed to find hiring threads: %s", exc)
        return result

    if not threads:
        logger.warning("jobs_hn: no hiring threads found")
        return result

    latest = threads[0]
    try:
        latest_comments = _fetch_comments(str(latest.get("objectID")))
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_hn: failed to fetch latest thread comments: %s", exc)
        latest_comments = []

    result["total_comments"] = len(latest_comments)
    result["stage_job_counts"] = _count_keywords(latest_comments, stage_keywords)

    if len(threads) >= 2:
        previous = threads[1]
        try:
            previous_comments = _fetch_comments(str(previous.get("objectID")))
            previous_counts = _count_keywords(previous_comments, stage_keywords)
            result["stage_job_mom_change"] = {
                stage_id: result["stage_job_counts"].get(stage_id, 0) - previous_counts.get(stage_id, 0)
                for stage_id in stage_keywords
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("jobs_hn: failed to fetch previous thread comments: %s", exc)

    logger.info(
        "jobs_hn: analyzed %d comments across %d thread(s)", result["total_comments"], len(threads)
    )
    return result
