"""Free, keyless earnings-calendar fallback (Yahoo Finance quoteSummary).

Tradier's fundamentals calendar endpoint (`TradierClient.get_next_earnings_date`)
is unavailable on many accounts, which leaves the earnings-trap gate (#16)
permanently dormant. This module adds a second, free, keyless source so the
gate can actually fire: Yahoo Finance's unofficial `quoteSummary` endpoint.

Yahoo has no official public API, may change its response shape, block
requests, or require a browser-like User-Agent without notice. Every failure
mode here (401, timeout, unexpected shape, parse error) degrades to None
rather than raising — the earnings-trap gate must simply stay dormant when
this source is unavailable, never crash the pipeline.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from src.http_utils import get_json

logger = logging.getLogger(__name__)

QUERY1_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
QUERY2_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"

# Yahoo 401s on requests that don't look like they came from a browser.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

TIMEOUT_SECONDS = 10


def _fetch(url_template: str, symbol: str) -> dict[str, Any] | None:
    try:
        return get_json(
            url_template.format(symbol=symbol),
            headers={"User-Agent": USER_AGENT},
            params={"modules": "calendarEvents"},
            timeout=TIMEOUT_SECONDS,
            retries=1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("earnings._fetch(%s) via %s failed: %s", symbol, url_template, exc)
        return None


def next_earnings_date(symbol: str) -> str | None:
    """Best-effort next earnings date via Yahoo Finance's quoteSummary endpoint.

    Tries the query1 host first, then query2 as a fallback (Yahoo sometimes
    rate-limits/blocks one edge hostname independently of the other). Returns
    the earliest `earningsDate` entry that falls today-or-later as
    "YYYY-MM-DD", or None on any failure: network error, auth failure,
    unexpected response shape, or no upcoming date found. This is an
    unofficial, keyless endpoint that Yahoo may change or remove without
    notice — every failure mode is treated as "no data available", never as
    a hard error.
    """
    if not symbol or not str(symbol).strip():
        return None
    symbol = str(symbol).strip().upper()

    data = _fetch(QUERY1_URL, symbol) or _fetch(QUERY2_URL, symbol)
    if not data:
        return None

    try:
        results = (data.get("quoteSummary") or {}).get("result") or []
        if not results:
            return None
        calendar_events = results[0].get("calendarEvents") or {}
        earnings = calendar_events.get("earnings") or {}
        earnings_dates = earnings.get("earningsDate") or []
        raw_timestamps = [
            entry.get("raw")
            for entry in earnings_dates
            if isinstance(entry, dict) and entry.get("raw") is not None
        ]
    except (AttributeError, TypeError, IndexError) as exc:
        logger.debug("earnings.next_earnings_date(%s): unexpected response shape: %s", symbol, exc)
        return None

    if not raw_timestamps:
        return None

    today = date.today()
    upcoming: list[date] = []
    for ts in raw_timestamps:
        try:
            parsed = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        except (TypeError, ValueError, OSError, OverflowError):
            continue
        if parsed >= today:
            upcoming.append(parsed)

    if not upcoming:
        return None

    return min(upcoming).isoformat()
