"""SEC EDGAR Full-Text Search collector: theme filing frequency (keyless).

Queries the free SEC EDGAR Full-Text Search index for one leading keyword per
theme and records the total hit count for recent filings (10-K/10-Q/8-K by
default). This approximates "how often is this theme showing up in SEC
filings / earnings commentary right now" — a proxy for real-world corporate
attention that complements GitHub/HN/arXiv developer & researcher buzz.

No API key required, but SEC requires a descriptive User-Agent with a
contact address — handled automatically by http_utils.DEFAULT_USER_AGENT.

Rate-limit friendly: at most one keyword per theme, a short pause between
requests, and an overall cap on total requests per run.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Any

from src.http_utils import get_json

logger = logging.getLogger(__name__)

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
MAX_REQUESTS = 12
REQUEST_PAUSE_SECONDS = 0.3


def _pick_keyword(theme: dict[str, Any]) -> str | None:
    """Pick the single most specific keyword/phrase for a theme.

    Heuristic: the longest keyword phrase tends to be the most specific
    (least likely to produce noisy/irrelevant hits), so prefer that.
    """
    keywords = theme.get("keywords") or []
    if not keywords:
        return None
    return max(keywords, key=len)


def _extract_total_hits(payload: Any) -> int:
    """Defensively extract hits.total.value from the FTS response shape."""
    if not isinstance(payload, dict):
        return 0
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        return 0
    total = hits.get("total")
    if isinstance(total, dict):
        value = total.get("value")
    else:
        value = total
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def collect(
    themes: list[dict[str, Any]] | None = None,
    lookback_days: int = 90,
    forms: str = "10-K,10-Q,8-K",
) -> dict[str, Any]:
    """Collect per-theme SEC EDGAR full-text search hit counts.

    Returns:
      {"source": "edgar_fts", "theme_counts": {theme_id: hit_count, ...}}

    Never raises — any per-theme failure is logged and yields a 0 count for
    that theme; NXT_OFFLINE=1 short-circuits to an empty result immediately.
    """
    result: dict[str, Any] = {"source": "edgar_fts", "theme_counts": {}}

    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("edgar_fts: NXT_OFFLINE=1, skipping network calls")
        return result

    themes = themes or []
    if not themes:
        return result

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    startdt = start_date.strftime("%Y-%m-%d")
    enddt = end_date.strftime("%Y-%m-%d")

    requests_made = 0
    for theme in themes:
        if requests_made >= MAX_REQUESTS:
            logger.info("edgar_fts: reached max request budget, stopping early")
            break

        theme_id = theme.get("id")
        if not theme_id:
            continue

        keyword = _pick_keyword(theme)
        if not keyword:
            result["theme_counts"][theme_id] = 0
            continue

        try:
            payload = get_json(
                FTS_URL,
                params={
                    "q": f'"{keyword}"',
                    "forms": forms,
                    "startdt": startdt,
                    "enddt": enddt,
                },
            )
            requests_made += 1
            hit_count = _extract_total_hits(payload)
            result["theme_counts"][theme_id] = hit_count
        except Exception as exc:  # noqa: BLE001
            logger.warning("edgar_fts: search failed for theme %s ('%s'): %s", theme_id, keyword, exc)
            result["theme_counts"][theme_id] = 0
            requests_made += 1

        time.sleep(REQUEST_PAUSE_SECONDS)

    logger.info("edgar_fts: collected counts for %d themes", len(result["theme_counts"]))
    return result
