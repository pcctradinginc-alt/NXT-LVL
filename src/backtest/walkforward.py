"""Walk-forward out-of-sample backtest for the NXT LVL scoring logic.

Honest scope: src/backtest/price_backtest.py and calibrate.py both note that a
true retroactive backtest of the collector-driven signal logic is impossible
because most free data sources are point-in-time snapshots that were never
archived. This module closes part of that gap: several of the sources ARE
independently re-queryable for a past date range, so we can reconstruct what
a compact digest would have looked like on a given historical `as_of` date
and then measure whether it actually predicted forward returns.

Which sources are historically reconstructable (queried with explicit date
ranges, see the `_hist_*` helpers below) and which are not:

  - arXiv (submittedDate range query)                    RECONSTRUCTABLE
  - Hacker News story buzz (Algolia created_at_i range)   RECONSTRUCTABLE
  - SEC EDGAR full-text search (startdt/enddt)            RECONSTRUCTABLE
  - HN "Who is hiring" keyword mentions (Algolia range)   RECONSTRUCTABLE
  - Underlying price history (Tradier daily history)      RECONSTRUCTABLE
  - GitHub stars                                          NOT RECONSTRUCTABLE
    (the GitHub API only ever returns the CURRENT star count; there is no
    historical star-count endpoint, so github_trends is entirely excluded
    from this walk-forward)

This module writes parallel `_hist_*` query functions rather than importing
or modifying the production collectors in src/collectors/ (those are
deliberately "always query the recent window" and are not date-range
parameterized) or scoring.py / config.py / main.py (owned by other work in
parallel). It reuses read-only patterns from price_backtest.py (trading-day
snapping helpers) and calibrate.py (Spearman rank correlation) instead of
duplicating that logic.

Simplified as-of scoring (score_universe_asof) uses only three components,
weighted as documented in WALKFORWARD_WEIGHTS:
  - divergence:     scoring.score_divergence() of the trailing 3-month price
                     move ending at as_of (same bucket logic as production).
  - theme_momentum: growth of the ticker's mapped theme(s) source-count sum
                     (current 90d window) vs. the immediately preceding 90d
                     baseline window — a lightweight momentum proxy.
  - breadth:        how many of the 4 reconstructable sources show non-zero
                     signal for the ticker's theme(s) in the current window.

Proposal cross-references:
  #2  walk-forward / point-in-time reconstruction — this module.
  #6  lead-lag — see `_compute_lead_lag`: per source, the horizon with the
      strongest |IC| is reported as that source's empirical "lead time".
  #12 temporal ordering — see `_compute_temporal_ordering`: what fraction of
      eventual 90d winners had a leading source (arxiv/jobs) signal while
      price (divergence bucket) had not yet moved.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.analysis import scoring, trend
from src.analysis.options_math import bs_call_price, solve_strike_for_delta
from src.backtest.calibrate import spearman
from src.backtest.price_backtest import (
    TRAILING_WINDOW_DAYS,
    _build_price_map,
    _nearest_trading_day,
    _parse_date,
    _realized_vol,
)
from src.config import DATA_DIR, load_settings
from src.http_utils import get_json, get_text
from src.options.tradier import TradierClient

logger = logging.getLogger(__name__)

REPORT_PATH = DATA_DIR / "walkforward_report.json"

# Phase A cache directory (CONCEPT_PROFIT.md): monthly count buckets + full
# per-ticker daily price history, persisted across runs (see .github/workflows
# /backtest.yml's actions/cache step) so re-runs over an overlapping date
# range make ~zero network calls instead of re-querying everything.
CACHE_DIR = DATA_DIR / "backtest_cache"
COUNTS_CACHE_PATH = CACHE_DIR / "counts.json"

DEFAULT_HORIZONS: tuple[int, ...] = (30, 60, 90, 180)
DEFAULT_BENCHMARKS: tuple[str, ...] = ("SPY", "QQQ", "SOXX")
DEFAULT_RECONSTRUCT_WINDOW_DAYS = 90
SOURCES = ("edgar_fts", "arxiv", "hn_buzz", "jobs")

# Phase B trend/regime gate (CONCEPT_PROFIT.md): the regime gate always reads
# SPY specifically (broad-market risk-on/off proxy), independent of whichever
# ticker/theme is being scored and independent of DEFAULT_BENCHMARKS' order.
REGIME_BENCHMARK = "SPY"

# momentum_12_1 (src.analysis.trend) needs >=253 daily closes ending ~1 month
# before as_of; trend_ok/regime_risk_on need >=200. 400 calendar days safely
# covers 253 trading days even accounting for weekends/holidays. Used to
# widen the very first (cold-cache) per-ticker price fetch in run_walkforward
# (see _price_map_for's `history_start`) so every as-of date in a run has
# enough trailing history for these components rather than silently falling
# back to trend.py's neutral/None on the earliest as-of dates only.
MOMENTUM_LOOKBACK_CALENDAR_DAYS = 400

# arXiv rate-limit hygiene: >=3.0s spacing between actual network calls
# (module-level last-call timestamp, shared across every theme/month query in
# the process), plus exponential backoff specifically on HTTP 429 before
# giving up on that one (theme, month) bucket.
_ARXIV_MIN_INTERVAL_SECONDS = 3.0
_ARXIV_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 15.0, 45.0)
_arxiv_last_call_ts: float = 0.0

# Synthetic option P/L (CONCEPT_PROFIT.md Phase A #4): a 120-DTE delta-0.60
# call, Black-Scholes re-valued at each horizon, minus an estimated
# half-spread trading cost on both legs. These mirror price_backtest.py's
# run_backtest() defaults (entry_dte=120, target_delta=0.60, r=0.04) so the
# two backtests price the "same" structural option; spread_pct=0.06 is new
# here and is a documented COST ASSUMPTION (a rough retail-illiquid-option
# round-trip spread), not a quoted market spread (no historical options-chain
# data exists to backtest against — see price_backtest.py's module docstring).
OPTION_ENTRY_DTE_DAYS = 120
OPTION_TARGET_DELTA = 0.60
OPTION_RISK_FREE_RATE = 0.04
OPTION_SPREAD_PCT = 0.06

# Simple, documented weights for the as-of walk-forward score. These are
# intentionally simpler than scoring.DEFAULT_WEIGHTS (no option_quality /
# stage_fit / emergence — those either need live option chains or the full
# emergence pipeline, neither of which is point-in-time reconstructable).
#
# CONCEPT_PROFIT.md Phase B/C: the first real walk-forward run (2026-07-12,
# n=96) measured breadth at IC=-0.305 and divergence at IC=-0.169 @90d (both
# anti-predictive on that sample), theme_momentum at IC=+0.051 (the only
# positive candidate), and had no price-momentum factor at all. These
# defaults react to that finding — momentum_12_1 (the best-documented free
# momentum factor, Phase B #1) gets the largest weight, theme_momentum keeps
# its prior (only-positive-so-far) weight, and divergence/breadth are cut
# back since neither has earned default trust. These are still just a
# starting point, not a claim of edge: `src/backtest/optimize.py` (Phase C)
# recalibrates weights from measured out-of-sample IC, and *that* calibration
# is what should actually gate production scoring, not these defaults.
WALKFORWARD_WEIGHTS = {
    "divergence": 0.20,
    "theme_momentum": 0.30,
    "breadth": 0.10,
    "momentum_12_1": 0.40,
}

HEADER_LINE = (
    "Walk-forward out-of-sample backtest — reconstructs point-in-time digests "
    "from historically-queryable sources (arXiv/HN/EDGAR/jobs + price); "
    "GitHub-stars excluded (not reconstructable). This measures whether the "
    "scores predicted forward returns."
)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Disk cache I/O (Phase A #1/#2 of CONCEPT_PROFIT.md) — all fault-tolerant:
# an unreadable/corrupt cache file starts empty rather than crashing the run.
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write `data` as JSON to `path`, via a tmp-file + replace (atomic-ish:
    a crash mid-write leaves either the old file or the new one, never a
    half-written one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp_path.replace(path)


def _load_counts_cache(path: Path | None = None) -> dict[str, int]:
    """Load the monthly source-count cache: {"<source>|<theme_id>|<YYYY-MM>": int}.

    `path` is overridable for tests; defaults to COUNTS_CACHE_PATH. Any
    missing file, unreadable JSON, or unexpected shape degrades to an empty
    cache (a cold-cache run, not a crash).
    """
    cache_path = path or COUNTS_CACHE_PATH
    try:
        with open(cache_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001
        logger.warning("walkforward: counts cache unreadable (%s), starting empty: %s", cache_path, exc)
    return {}


def _save_counts_cache(cache: dict[str, int], path: Path | None = None) -> None:
    cache_path = path or COUNTS_CACHE_PATH
    try:
        _atomic_write_json(cache_path, cache)
    except Exception as exc:  # noqa: BLE001
        logger.warning("walkforward: failed to save counts cache (%s): %s", cache_path, exc)


def _safe_ticker_filename(ticker: str) -> str:
    return "".join(ch for ch in str(ticker).upper() if ch.isalnum()) or "UNKNOWN"


def _price_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"prices_{_safe_ticker_filename(ticker)}.json"


def _load_disk_price_cache(ticker: str, path: Path | None = None) -> dict[str, float]:
    """Load the full cached daily-close span for `ticker` from
    data/backtest_cache/prices_<TICKER>.json ({"<YYYY-MM-DD>": close})."""
    cache_path = path or _price_cache_path(ticker)
    try:
        with open(cache_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "walkforward: price cache unreadable for %s (%s), starting empty: %s", ticker, cache_path, exc
        )
    return {}


def _save_disk_price_cache(ticker: str, series: dict[str, float], path: Path | None = None) -> None:
    cache_path = path or _price_cache_path(ticker)
    try:
        _atomic_write_json(cache_path, series)
    except Exception as exc:  # noqa: BLE001
        logger.warning("walkforward: failed to save price cache for %s (%s): %s", ticker, cache_path, exc)


# ---------------------------------------------------------------------------
# Monthly bucketing (Phase A #1) — replaces per-(theme, window, as-of) count
# queries with per-(theme, calendar-month) queries. A window's count is the
# sum of the calendar months it touches; this quantizes windows to whole
# months (e.g. a window spanning Jan 15 - Apr 15 counts all of Jan-Apr, not
# just the 3 months it partially covers) — an intentional approximation that
# is what makes month-level caching possible, and is acceptable for measuring
# count *trends* rather than exact daily windows.
# ---------------------------------------------------------------------------


def _month_range(start_date: date, end_date: date) -> list[str]:
    """List of 'YYYY-MM' calendar-month buckets covering [start_date, end_date]
    inclusive. Order-tolerant (swaps the bounds if given reversed)."""
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    months: list[str] = []
    y, m = start_date.year, start_date.month
    end_y, end_m = end_date.year, end_date.month
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def _month_bounds(month: str) -> tuple[date, date]:
    """(first day, last day) of the given 'YYYY-MM' calendar month."""
    y, m = int(month[:4]), int(month[5:7])
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m + 1, 1) - timedelta(days=1)
    return start, end


def _window_sum_from_buckets(buckets: dict[str, int], months: list[str]) -> int:
    """Sum the given months' buckets. A month missing from `buckets` (never
    queried, or a degraded/failed query — see `_bucketed_counts`) contributes
    0 rather than raising."""
    return sum(buckets.get(m, 0) for m in months)


def _bucketed_counts(
    source_fn: Callable[[list[dict[str, Any]], date, date], dict[str, Any]],
    cache: dict[str, int],
    source_name: str,
    theme: dict[str, Any],
    months: list[str],
    *,
    degraded: list[int] | None = None,
) -> dict[str, int]:
    """Per-(source, theme, calendar-month) counts, cache-first.

    For each month: a *completed* (non-current) month already present in
    `cache` is served straight from there — completed months are immutable,
    so the cache never goes stale. Otherwise `source_fn` is called for a
    single-theme list over that one calendar month's [start, end] range, and
    the result is written back into `cache` (mutated in place; the caller is
    responsible for persisting it — see `_save_counts_cache`).

    The CURRENT (still-incomplete) calendar month is always re-queried and
    deliberately never cached (its true count keeps changing until the month
    ends).

    A `source_fn` result of None for the theme (currently only
    `_hist_arxiv_counts`, after its 429 backoff is exhausted) marks that
    bucket as degraded: it is excluded from the returned dict (so it
    contributes 0 to any window sum built from it downstream) and never
    cached, and `degraded[0]` is incremented if a counter list is given —
    this is how the report surfaces "we dropped N buckets rather than
    silently recording a false zero."
    """
    theme_id = theme.get("id")
    current_month = date.today().strftime("%Y-%m")
    out: dict[str, int] = {}
    for month in months:
        cache_key = f"{source_name}|{theme_id}|{month}"
        if month != current_month and cache_key in cache:
            out[month] = cache[cache_key]
            continue
        month_start, month_end = _month_bounds(month)
        try:
            result = source_fn([theme], month_start, month_end)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "walkforward._bucketed_counts: %s failed for theme=%s month=%s: %s",
                source_name,
                theme_id,
                month,
                exc,
            )
            result = {}
        value = result.get(theme_id) if isinstance(result, dict) else None
        if value is None:
            if degraded is not None:
                degraded[0] += 1
            continue
        value = int(value)
        out[month] = value
        if month != current_month:
            cache[cache_key] = value
    return out


# ---------------------------------------------------------------------------
# Historical (point-in-time) source queries — parallel to src/collectors/*,
# but parameterized by an explicit [start, end] date range instead of always
# querying "now". Each is fault-tolerant per theme: a single failed request
# degrades to a 0 count for that theme rather than aborting the whole run.
# ---------------------------------------------------------------------------


def _pick_keyword(theme: dict[str, Any]) -> str | None:
    keywords = theme.get("keywords") or []
    if not keywords:
        return None
    return max(keywords, key=len)


def _extract_arxiv_total(xml_text: str) -> int:
    try:
        root = ET.fromstring(xml_text)
        el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        if el is not None and el.text:
            return int(el.text)
    except Exception:  # noqa: BLE001
        pass
    return 0


def _arxiv_paced_get_text(url: str, params: dict[str, Any]) -> str | None:
    """GET `url` respecting >=3.0s spacing since the last arXiv call (module-
    level timestamp, shared across all themes/months in this process), with
    exponential backoff specifically on HTTP 429 (5s, 15s, 45s) before giving
    up. Returns None (never raises) once backoff is exhausted, signalling the
    caller to treat this as a degraded/missing bucket rather than a fabricated
    zero count.
    """
    global _arxiv_last_call_ts
    delays: tuple[float, ...] = (0.0, *_ARXIV_BACKOFF_SECONDS)
    last_exc: Exception | None = None
    for attempt, extra_delay in enumerate(delays):
        if extra_delay:
            logger.warning(
                "walkforward: arXiv 429, backing off %.0fs before retry %d/%d",
                extra_delay,
                attempt,
                len(delays) - 1,
            )
            time.sleep(extra_delay)
        elapsed = time.monotonic() - _arxiv_last_call_ts
        if elapsed < _ARXIV_MIN_INTERVAL_SECONDS:
            time.sleep(_ARXIV_MIN_INTERVAL_SECONDS - elapsed)
        try:
            text = get_text(url, params=params)
            _arxiv_last_call_ts = time.monotonic()
            return text
        except Exception as exc:  # noqa: BLE001
            _arxiv_last_call_ts = time.monotonic()
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429 and attempt < len(delays) - 1:
                continue
            logger.warning("walkforward: arXiv request failed (giving up): %s", exc)
            return None
    logger.warning("walkforward: arXiv request exhausted all retries: %s", last_exc)
    return None


def _hist_arxiv_counts(themes: list[dict[str, Any]], start: date, end: date) -> dict[str, int | None]:
    """Historical per-theme arXiv paper counts submitted within [start, end].

    Uses arXiv's `submittedDate` range filter plus the single most specific
    keyword per theme (same heuristic as edgar_fts._pick_keyword), reading
    the Atom feed's `opensearch:totalResults` as the count. This is lighter
    than the production arxiv_trends collector (which fetches MAX_RESULTS
    recent papers and counts keyword hits locally) because that collector is
    not date-range scoped in the first place.

    A theme's count is None (rather than 0) when the paced/backed-off request
    (see `_arxiv_paced_get_text`) never got a usable response — a degraded
    bucket, not a genuine zero. `_bucketed_counts` treats None specially:
    excluded from the sum, never cached.
    """
    result: dict[str, int | None] = {}
    start_str = start.strftime("%Y%m%d") + "000000"
    end_str = end.strftime("%Y%m%d") + "235959"
    for theme in themes:
        theme_id = theme.get("id")
        if not theme_id:
            continue
        keyword = _pick_keyword(theme)
        if not keyword:
            result[theme_id] = 0
            continue
        query = (
            f'(cat:cs.AI OR cat:cs.LG OR cat:cs.RO) AND abs:"{keyword}" '
            f"AND submittedDate:[{start_str} TO {end_str}]"
        )
        xml_text = _arxiv_paced_get_text(
            "http://export.arxiv.org/api/query",
            {"search_query": query, "max_results": 1},
        )
        if xml_text is None:
            result[theme_id] = None
            continue
        result[theme_id] = _extract_arxiv_total(xml_text)
    return result


def _extract_edgar_total(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        return 0
    total = hits.get("total")
    value = total.get("value") if isinstance(total, dict) else total
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _hist_edgar_fts(
    themes: list[dict[str, Any]], start: date, end: date, forms: str = "10-K,10-Q,8-K"
) -> dict[str, int]:
    """Historical per-theme SEC EDGAR full-text-search hit counts in [start, end].

    Directly parallels src/collectors/edgar_fts.py's query shape, but with the
    date window passed in explicitly instead of derived from "today".
    """
    result: dict[str, int] = {}
    startdt, enddt = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    for theme in themes:
        theme_id = theme.get("id")
        if not theme_id:
            continue
        keyword = _pick_keyword(theme)
        if not keyword:
            result[theme_id] = 0
            continue
        try:
            payload = get_json(
                "https://efts.sec.gov/LATEST/search-index",
                params={"q": f'"{keyword}"', "forms": forms, "startdt": startdt, "enddt": enddt},
            )
            result[theme_id] = _extract_edgar_total(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("walkforward._hist_edgar_fts: failed for theme %s: %s", theme_id, exc)
            result[theme_id] = 0
        time.sleep(0.2)
    return result


def _hist_hn_buzz(themes: list[dict[str, Any]], start: date, end: date) -> dict[str, dict[str, int]]:
    """Historical per-theme HN story buzz (story count + total points) for
    stories created within [start, end], via Algolia's numericFilters range —
    the same query shape as src/collectors/hn_buzz.py, date-window parameterized.
    """
    result: dict[str, dict[str, int]] = {}
    start_ts = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    for theme in themes:
        theme_id = theme.get("id")
        if not theme_id:
            continue
        stories = 0
        points = 0
        for keyword in (theme.get("keywords") or [])[:2]:
            try:
                data = get_json(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "query": keyword,
                        "tags": "story",
                        "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
                        "hitsPerPage": 50,
                    },
                )
                hits = data.get("hits", [])
                stories += len(hits)
                points += sum(hit.get("points") or 0 for hit in hits)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "walkforward._hist_hn_buzz: failed for theme %s keyword '%s': %s", theme_id, keyword, exc
                )
            time.sleep(0.2)
        result[theme_id] = {"stories": stories, "points": points}
    return result


def _hist_jobs_hn(themes: list[dict[str, Any]], start: date, end: date) -> dict[str, int]:
    """Approximate historical per-theme HN "hiring" keyword mention counts in [start, end].

    The production jobs_hn collector scans the FULL comment text of the two
    most recent monthly "Who is hiring?" threads — meaningful only as "the
    current month's threads", not date-range queryable the same way. For
    point-in-time reconstruction we instead query Algolia's comment search
    directly (tagged to author_whoishiring threads) with a created_at_i range
    and use the returned `nbHits` total-match count as a lighter-weight proxy
    for keyword mention volume in the window. This is a documented
    approximation of the production per-thread comment scan, not an exact replay.
    """
    result: dict[str, int] = {}
    start_ts = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    for theme in themes:
        theme_id = theme.get("id")
        if not theme_id:
            continue
        total_hits = 0
        for keyword in (theme.get("keywords") or [])[:2]:
            try:
                data = get_json(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "query": keyword,
                        "tags": "comment,author_whoishiring",
                        "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts}",
                        "hitsPerPage": 0,
                    },
                )
                total_hits += int(data.get("nbHits") or 0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "walkforward._hist_jobs_hn: failed for theme %s keyword '%s': %s", theme_id, keyword, exc
                )
            time.sleep(0.2)
        result[theme_id] = total_hits
    return result


# ---------------------------------------------------------------------------
# Digest reconstruction
# ---------------------------------------------------------------------------


def _hist_hn_buzz_scalar(themes: list[dict[str, Any]], start: date, end: date) -> dict[str, int]:
    """Scalar wrapper around `_hist_hn_buzz` for use as a `_bucketed_counts`
    source_fn: combines story count + total points into one per-theme int,
    the same combination `reconstruct_digest` used to do inline before the
    monthly-bucketing rewrite.
    """
    raw = _hist_hn_buzz(themes, start, end)
    return {tid: v.get("stories", 0) + v.get("points", 0) for tid, v in raw.items()}


# Every reconstructable source, keyed by the digest field name it fills, all
# sharing the uniform (themes, start, end) -> {theme_id: count|None} shape
# `_bucketed_counts` expects.
SOURCE_FNS: dict[str, Callable[[list[dict[str, Any]], date, date], dict[str, Any]]] = {
    "edgar_fts": _hist_edgar_fts,
    "arxiv": _hist_arxiv_counts,
    "hn_buzz": _hist_hn_buzz_scalar,
    "jobs": _hist_jobs_hn,
}


def reconstruct_digest(
    as_of_date: date,
    themes: list[dict[str, Any]],
    *,
    window_days: int = DEFAULT_RECONSTRUCT_WINDOW_DAYS,
    mock_data: dict[str, Any] | None = None,
    cache: dict[str, int] | None = None,
    degraded_out: list[int] | None = None,
) -> dict[str, Any]:
    """Build a compact point-in-time digest for `as_of_date`.

    Queries the reconstructable sources (edgar_fts, arxiv, hn_buzz, jobs —
    GitHub omitted, see module docstring) over a trailing `window_days`
    window ending at `as_of_date`, plus the immediately preceding window of
    the same length as a momentum baseline.

    Phase A monthly bucketing (CONCEPT_PROFIT.md): rather than issuing fresh
    current-window and baseline-window queries every call, both windows are
    expressed as the calendar months they touch (`_month_range`), each
    (source, theme, month) bucket is fetched at most once ever via
    `_bucketed_counts` (cache-first, `cache` mutated in place so the caller
    can persist it across as-of dates and across runs), and a window's count
    is the sum of its covered months (`_window_sum_from_buckets`) — this
    quantizes windows to whole calendar months (see `_month_range`'s
    docstring), trading a little date precision for a ~large reduction in
    network calls across a multi-year, many-as-of-date walk-forward run.

    Returns:
      {theme_id: {"edgar_fts": int, "arxiv": int, "hn_buzz": int, "jobs": int,
                   "baseline": {"edgar_fts": int, "arxiv": int, "hn_buzz": int, "jobs": int}}}

    If `mock_data` is given, it is returned as-is and no network call is made
    at all — used for offline/test runs. `cache` defaults to a fresh in-memory
    dict when not given (so direct/offline callers still work), but a real
    run should pass in a dict loaded via `_load_counts_cache()` and persist it
    via `_save_counts_cache()` to get the cross-run benefit. `degraded_out`,
    when given, has its `[0]` incremented once per degraded (excluded, e.g.
    arXiv-429-exhausted) bucket — see `_bucketed_counts`.

    Fault-tolerant per source: a failure in one source's query never blocks
    the others (`_bucketed_counts` already catches per (source, theme, month)
    internally).
    """
    if mock_data is not None:
        return mock_data

    if cache is None:
        cache = {}

    theme_ids = [t.get("id") for t in themes if t.get("id")]
    current_start = as_of_date - timedelta(days=window_days)
    baseline_end = current_start
    baseline_start = baseline_end - timedelta(days=window_days)

    current_months = _month_range(current_start, as_of_date)
    baseline_months = _month_range(baseline_start, baseline_end)
    all_months = sorted(set(current_months) | set(baseline_months))

    digest: dict[str, Any] = {
        theme_id: {
            "edgar_fts": 0,
            "arxiv": 0,
            "hn_buzz": 0,
            "jobs": 0,
            "baseline": {"edgar_fts": 0, "arxiv": 0, "hn_buzz": 0, "jobs": 0},
        }
        for theme_id in theme_ids
    }

    for theme in themes:
        theme_id = theme.get("id")
        if not theme_id or theme_id not in digest:
            continue
        for source_name, fn in SOURCE_FNS.items():
            buckets = _bucketed_counts(fn, cache, source_name, theme, all_months, degraded=degraded_out)
            digest[theme_id][source_name] = _window_sum_from_buckets(buckets, current_months)
            digest[theme_id]["baseline"][source_name] = _window_sum_from_buckets(buckets, baseline_months)

    return digest


# ---------------------------------------------------------------------------
# As-of scoring
# ---------------------------------------------------------------------------


def _ticker_theme_ids(tickers: list[str], themes: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {t: [] for t in tickers}
    for theme in themes:
        theme_id = theme.get("id")
        if not theme_id:
            continue
        for ticker in theme.get("tickers") or []:
            ticker_u = str(ticker).upper()
            if ticker_u in mapping:
                mapping[ticker_u].append(theme_id)
    return mapping


def _theme_signal(theme_ids: list[str], digest: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Sum a ticker's mapped theme(s) source counts (current + baseline)."""
    current_sum = {s: 0 for s in SOURCES}
    baseline_sum = {s: 0 for s in SOURCES}
    for theme_id in theme_ids:
        block = digest.get(theme_id) or {}
        baseline = block.get("baseline") or {}
        for s in SOURCES:
            v = block.get(s)
            if isinstance(v, (int, float)):
                current_sum[s] += v
            bv = baseline.get(s)
            if isinstance(bv, (int, float)):
                baseline_sum[s] += bv
    return {"current": current_sum, "baseline": baseline_sum}


def _theme_momentum_score(current_sum: dict[str, int], baseline_sum: dict[str, int]) -> float:
    """Growth of the theme signal vs. its trailing baseline window, mapped to 0-100.

    0% growth (flat) maps to 50 (neutral); +200% or more growth maps to 100;
    -100% (signal vanished) maps to 0. No signal at all in either window is
    neutral (50); brand-new signal from a zero baseline maps to 100 (positive
    momentum, can't compute a percentage from zero).
    """
    total_current = sum(current_sum.values())
    total_baseline = sum(baseline_sum.values())
    if total_current == 0 and total_baseline == 0:
        return 50.0
    if total_baseline == 0:
        return 100.0 if total_current > 0 else 50.0
    growth_pct = (total_current - total_baseline) / total_baseline * 100.0
    return _clip((growth_pct + 100.0) / 300.0 * 100.0)


def _breadth_score(current_sum: dict[str, int]) -> float:
    nonzero = sum(1 for v in current_sum.values() if v > 0)
    return _clip(nonzero / len(SOURCES) * 100.0)


def _price_map_for(
    ticker: str,
    tradier: TradierClient | None,
    price_series: dict[str, dict[str, float]] | None,
    as_of_date: date,
    *,
    lookback_days: int = 120,
    price_cache: dict[str, dict[date, float]] | None = None,
    history_start: date | None = None,
) -> dict[date, float]:
    """Build a {date: close} map for `ticker`.

    Uses `price_series` (offline/test) verbatim when given. Otherwise (live
    mode) this consults a two-layer cache before ever calling Tradier:

      1. `price_cache` (in-memory, keyed by ticker, run-scoped) — populated
         once per ticker; every later call for the same ticker in this run is
         served straight from here regardless of `as_of_date`/`lookback_days`.
         Before this existed, a single walk-forward run re-fetched full price
         history separately per (ticker, as_of, horizon) call — including
         once per BENCHMARK per TICKER per horizon (`run_walkforward` checks
         each benchmark's forward return inside the per-ticker loop) — which
         for a 3-year/monthly/60-ticker run is on the order of tens of
         thousands of redundant Tradier calls, almost certainly the real
         cause of the multi-hour hang this cache fixes.
      2. `data/backtest_cache/prices_<TICKER>.json` on disk (see
         `_load_disk_price_cache` / `_save_disk_price_cache`) — persists the
         fetched span across separate runs/processes.

    `history_start`, when given, widens the very first (cold-cache) fetch for
    a ticker to start there instead of `as_of_date - lookback_days`, so the
    cached span is guaranteed wide enough for every as-of date used anywhere
    in the run regardless of which caller/date happens to trigger the first
    fetch. Tradier's history endpoint always fetches through "today" (no
    end-date param), so one sufficiently-early-starting fetch per ticker
    covers the whole run.
    """
    if price_series is not None:
        raw = price_series.get(ticker, {})
        out: dict[date, float] = {}
        for date_str, close in raw.items():
            try:
                d = _parse_date(str(date_str)[:10])
                out[d] = float(close)
            except (ValueError, TypeError):
                continue
        return out

    if tradier is None:
        return {}

    if price_cache is not None and ticker in price_cache:
        return price_cache[ticker]

    needed_start = as_of_date - timedelta(days=lookback_days)
    if history_start is not None and history_start < needed_start:
        needed_start = history_start

    disk = _load_disk_price_cache(ticker)
    disk_dates: list[date] = []
    for d in disk:
        try:
            disk_dates.append(_parse_date(d))
        except (ValueError, TypeError):
            continue

    if disk_dates and min(disk_dates) <= needed_start:
        out = {_parse_date(d): float(c) for d, c in disk.items()}
    else:
        try:
            bars = tradier.get_history(ticker, start=needed_start.strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("walkforward: history fetch failed for %s: %s", ticker, exc)
            bars = []
        fresh = _build_price_map(bars)
        merged: dict[str, float] = dict(disk)
        merged.update({d.isoformat(): c for d, c in fresh.items()})
        _save_disk_price_cache(ticker, merged)
        out = {_parse_date(d): float(c) for d, c in merged.items()}

    if price_cache is not None:
        price_cache[ticker] = out
    return out


def _trailing_perf_pct_asof(
    ticker: str,
    as_of_date: date,
    tradier: TradierClient | None,
    price_series: dict[str, dict[str, float]] | None,
    window_days: int = 90,
    *,
    price_cache: dict[str, dict[date, float]] | None = None,
    history_start: date | None = None,
) -> float | None:
    """Trailing `window_days` % price move ending at (the nearest trading day <=) as_of_date."""
    price_map = _price_map_for(
        ticker,
        tradier,
        price_series,
        as_of_date,
        lookback_days=window_days + 30,
        price_cache=price_cache,
        history_start=history_start,
    )
    sorted_dates = sorted(price_map.keys())
    if not sorted_dates:
        return None
    end_date = _nearest_trading_day(sorted_dates, as_of_date, allow_after=False)
    if end_date is None:
        return None
    start_date = _nearest_trading_day(sorted_dates, end_date - timedelta(days=window_days), allow_after=False)
    if start_date is None or start_date >= end_date:
        return None
    start_close = price_map[start_date]
    end_close = price_map[end_date]
    if start_close <= 0:
        return None
    return (end_close - start_close) / start_close * 100.0


def _trailing_closes_asof(
    ticker: str,
    as_of_date: date,
    tradier: TradierClient | None,
    price_series: dict[str, dict[str, float]] | None,
    *,
    lookback_days: int = MOMENTUM_LOOKBACK_CALENDAR_DAYS,
    price_cache: dict[str, dict[date, float]] | None = None,
    history_start: date | None = None,
) -> list[float]:
    """Chronological (oldest-first) daily closes for `ticker` up to and
    including the nearest trading day <= as_of_date.

    Feeds src.analysis.trend's momentum_12_1/trend_ok/regime_risk_on, which
    already handle too-short input gracefully (None/neutral rather than
    raising) — see trend.py's module docstring — so this helper doesn't need
    to enforce a minimum length itself; it just hands over whatever history
    is available in the price map.
    """
    price_map = _price_map_for(
        ticker,
        tradier,
        price_series,
        as_of_date,
        lookback_days=lookback_days,
        price_cache=price_cache,
        history_start=history_start,
    )
    if not price_map:
        return []
    sorted_dates = sorted(price_map.keys())
    end_date = _nearest_trading_day(sorted_dates, as_of_date, allow_after=False)
    if end_date is None:
        return []
    return [price_map[d] for d in sorted_dates if d <= end_date]


def score_universe_asof(
    as_of_date: date,
    tickers: list[str],
    digest: dict[str, Any],
    tradier: TradierClient | None,
    *,
    themes: list[dict[str, Any]] | None = None,
    price_series: dict[str, dict[str, float]] | None = None,
    price_cache: dict[str, dict[date, float]] | None = None,
    history_start: date | None = None,
    benchmark: str = REGIME_BENCHMARK,
) -> list[dict[str, Any]]:
    """Simplified as-of score for each ticker, from reconstructable components only.

    `themes` (same shape as config.yaml's `themes` list — each with "id" and
    "tickers") is used to map each ticker to its theme(s) in `digest`, driving
    the theme_momentum and breadth components. Without it (or for tickers
    mapped to no theme), theme_momentum falls back to neutral (50) and
    breadth to 0 (no thematic evidence found for that ticker).

    CONCEPT_PROFIT.md Phase B adds:
      - momentum_12_1: trend.score_momentum_12_1() of the ticker's trailing
        closes up to and including as_of — a WEIGHTED component like the
        other three (see WALKFORWARD_WEIGHTS).
      - trend_ok / regime_risk_on: trend.trend_ok() (ticker) and
        trend.regime_risk_on() (the `benchmark`, default SPY), both evaluated
        as-of the same date. These are GATES, not weighted score inputs —
        stored as metadata on each result so run_walkforward can measure
        (via `filtered_buckets`) whether restricting to trend_ok AND
        regime_risk_on actually improves outcomes, rather than assuming it
        does. Each is True/False, or None when there isn't enough trailing
        history (see src.analysis.trend) to evaluate it — None is a distinct,
        honest "unknown", not coerced to True or False.

    Returns a list of
      {"ticker", "as_of",
       "components": {"divergence", "theme_momentum", "breadth", "momentum_12_1"},
       "total", "source_signals": {source: raw_current_count},
       "trend_ok": bool | None, "regime_risk_on": bool | None}
    (the extra "source_signals" field feeds the lead-lag analysis in
    run_walkforward and is not part of the documented minimal contract).
    """
    themes = themes or []
    ticker_theme_map = _ticker_theme_ids(tickers, themes)

    # Regime gate is a function of (benchmark, as_of_date) only — the same
    # for every ticker at this as_of date — so it's computed once here
    # rather than once per ticker.
    benchmark_closes = _trailing_closes_asof(
        benchmark, as_of_date, tradier, price_series, price_cache=price_cache, history_start=history_start
    )
    regime_on = trend.regime_risk_on(benchmark_closes)

    results: list[dict[str, Any]] = []
    for ticker in tickers:
        trailing_closes = _trailing_closes_asof(
            ticker, as_of_date, tradier, price_series, price_cache=price_cache, history_start=history_start
        )
        perf_3m = _trailing_perf_pct_asof(
            ticker, as_of_date, tradier, price_series, price_cache=price_cache, history_start=history_start
        )
        divergence = scoring.score_divergence(perf_3m)
        momentum_12_1 = trend.score_momentum_12_1(trailing_closes)

        theme_ids = ticker_theme_map.get(ticker, [])
        signal = _theme_signal(theme_ids, digest)
        theme_momentum = _theme_momentum_score(signal["current"], signal["baseline"])
        breadth = _breadth_score(signal["current"])

        total = _clip(
            divergence * WALKFORWARD_WEIGHTS["divergence"]
            + theme_momentum * WALKFORWARD_WEIGHTS["theme_momentum"]
            + breadth * WALKFORWARD_WEIGHTS["breadth"]
            + momentum_12_1 * WALKFORWARD_WEIGHTS["momentum_12_1"]
        )

        results.append(
            {
                "ticker": ticker,
                "as_of": as_of_date.isoformat(),
                "components": {
                    "divergence": round(divergence, 2),
                    "theme_momentum": round(theme_momentum, 2),
                    "breadth": round(breadth, 2),
                    "momentum_12_1": round(momentum_12_1, 2),
                },
                "total": round(total, 2),
                "source_signals": dict(signal["current"]),
                "trend_ok": trend.trend_ok(trailing_closes),
                "regime_risk_on": regime_on,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Forward returns (no look-ahead)
# ---------------------------------------------------------------------------


def forward_return(
    ticker: str,
    as_of_date: date,
    horizon_days: int,
    tradier: TradierClient | None,
    *,
    price_series: dict[str, dict[str, float]] | None = None,
    max_snap_days: int = 7,
    price_cache: dict[str, dict[date, float]] | None = None,
    history_start: date | None = None,
) -> float | None:
    """% return of `ticker`'s underlying from as_of_date to as_of_date + horizon_days.

    Both the entry and exit dates are snapped BACKWARD ONLY to the nearest
    available trading day (never forward) — this guarantees no look-ahead:
    the entry price is the most recent close known as of `as_of_date`, and
    the exit price is the most recent close known as of the target exit date.

    If no close exists within `max_snap_days` of the target exit date (the
    horizon has not truly elapsed in the available data, or price history
    simply doesn't extend that far), returns None rather than silently
    reusing a stale earlier close as if the horizon had elapsed.
    """
    price_map = _price_map_for(
        ticker,
        tradier,
        price_series,
        as_of_date,
        lookback_days=10,
        price_cache=price_cache,
        history_start=history_start,
    )
    if not price_map:
        return None
    sorted_dates = sorted(price_map.keys())

    entry_date = _nearest_trading_day(sorted_dates, as_of_date, allow_after=False)
    if entry_date is None:
        return None

    target_exit = as_of_date + timedelta(days=horizon_days)
    exit_date = _nearest_trading_day(sorted_dates, target_exit, allow_after=False)
    if exit_date is None or exit_date <= entry_date:
        return None
    if (target_exit - exit_date).days > max_snap_days:
        return None  # horizon hasn't elapsed in the available data yet

    entry_price = price_map[entry_date]
    exit_price = price_map[exit_date]
    if entry_price <= 0:
        return None
    return round((exit_price - entry_price) / entry_price * 100.0, 4)


# ---------------------------------------------------------------------------
# Option P/L (Phase A #4 of CONCEPT_PROFIT.md) — validates the instrument that
# is actually traded (a long call), not just the underlying's % return.
# ---------------------------------------------------------------------------


def _compute_option_metrics(
    ticker: str,
    as_of_date: date,
    tradier: TradierClient | None,
    price_series: dict[str, dict[str, float]] | None,
    horizons: tuple[int, ...],
    *,
    price_cache: dict[str, dict[date, float]] | None = None,
    history_start: date | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Synthetic 120-DTE delta-0.60 call P/L for `ticker`, entered at (the
    nearest trading day <=) `as_of_date` and re-valued at each horizon in
    `horizons` that does not exceed OPTION_ENTRY_DTE_DAYS (a horizon beyond
    the option's modeled life has no meaningful re-valuation and is omitted —
    None — rather than zero-filled; with the default DEFAULT_HORIZONS this
    means 180d never gets an option entry).

    Entry IV is the annualized realized volatility of the trailing ~63 daily
    log returns (`price_backtest._realized_vol`, clipped to [0.15, 1.5]) —
    the same IV proxy price_backtest.py's mechanical backtest uses, reused
    here rather than reimplemented, since no historical options-chain data
    exists to backtest a real quoted IV against. Both the entry and exit
    theoretical Black-Scholes values are reduced by an estimated half-spread
    trading cost of `0.5 * OPTION_SPREAD_PCT * <that leg's raw BS value>` — a
    documented COST ASSUMPTION, not a quoted market spread.

    Returns {str(horizon): {"return": pct_change_as_fraction, "hit": bool} |
    None}; None for a horizon whenever entry/exit price data or IV is
    unavailable, the horizon hasn't elapsed in the available data yet (mirrors
    forward_return's max_snap_days=7 rule), or the horizon exceeds the
    option's DTE at entry.
    """
    out: dict[str, dict[str, Any] | None] = {str(h): None for h in horizons}

    price_map = _price_map_for(
        ticker,
        tradier,
        price_series,
        as_of_date,
        lookback_days=TRAILING_WINDOW_DAYS + 30,
        price_cache=price_cache,
        history_start=history_start,
    )
    if not price_map:
        return out
    sorted_dates = sorted(price_map.keys())

    entry_date = _nearest_trading_day(sorted_dates, as_of_date, allow_after=False)
    if entry_date is None:
        return out

    iv = _realized_vol(sorted_dates, price_map, entry_date)
    if iv is None:
        return out

    s_entry = price_map.get(entry_date)
    if not s_entry or s_entry <= 0:
        return out

    entry_t_years = OPTION_ENTRY_DTE_DAYS / 365.0
    strike = solve_strike_for_delta(s_entry, OPTION_TARGET_DELTA, entry_t_years, OPTION_RISK_FREE_RATE, iv)
    raw_entry_value = bs_call_price(s_entry, strike, entry_t_years, OPTION_RISK_FREE_RATE, iv)
    if raw_entry_value <= 0:
        return out
    entry_value = raw_entry_value * (1.0 - 0.5 * OPTION_SPREAD_PCT)
    if entry_value <= 0:
        return out

    for h in horizons:
        if h > OPTION_ENTRY_DTE_DAYS:
            continue  # beyond this option's modeled life, no re-valuation possible
        target_exit = as_of_date + timedelta(days=h)
        exit_date = _nearest_trading_day(sorted_dates, target_exit, allow_after=False)
        if exit_date is None or exit_date <= entry_date:
            continue
        if (target_exit - exit_date).days > 7:
            continue  # horizon hasn't elapsed in the available data yet
        s_exit = price_map.get(exit_date)
        if not s_exit or s_exit <= 0:
            continue

        remaining_t_years = (OPTION_ENTRY_DTE_DAYS - h) / 365.0
        raw_exit_value = bs_call_price(s_exit, strike, remaining_t_years, OPTION_RISK_FREE_RATE, iv)
        exit_value = raw_exit_value * (1.0 - 0.5 * OPTION_SPREAD_PCT)

        option_return = exit_value / entry_value - 1.0
        out[str(h)] = {"return": round(option_return, 4), "hit": option_return > 0}

    return out


def _aggregate_option_bucket(items: list[dict[str, Any]], horizon_key: str) -> dict[str, Any]:
    valid = [it for it in items if (it.get("option") or {}).get(horizon_key) is not None]
    n = len(valid)
    if n == 0:
        return {"n": 0, "hit_rate": None, "avg_option_return_pct": None}
    rets = [it["option"][horizon_key]["return"] for it in valid]
    hits = sum(1 for r in rets if r > 0)
    return {
        "n": n,
        "hit_rate": round(hits / n, 4),
        "avg_option_return_pct": round(sum(rets) / n * 100.0, 4),
    }


def _build_option_buckets(samples: list[dict[str, Any]], horizon_key: str = "90") -> dict[str, Any]:
    """Option hit-rate & avg return by total-score quartile at `horizon_key`
    (primary horizon: 90d). Quartiles are computed over the subset of samples
    that actually have an option entry at this horizon (see
    `_compute_option_metrics`'s None-for-unavailable rule), same convention
    `_build_buckets` uses per forward-return horizon.
    """
    valid = [s for s in samples if (s.get("option") or {}).get(horizon_key) is not None]
    bucketed = _bucket_by_quartile(valid)
    return {label: _aggregate_option_bucket(items, horizon_key) for label, items in bucketed.items()}


# ---------------------------------------------------------------------------
# Full walk-forward run
# ---------------------------------------------------------------------------


def _asof_grid(start: date, end: date, cadence_days: int) -> list[date]:
    grid: list[date] = []
    cursor = start
    step = max(1, cadence_days)
    while cursor <= end:
        grid.append(cursor)
        cursor = cursor + timedelta(days=step)
    return grid


def _generate_mock_digest(theme_ids: list[str], rng: random.Random) -> dict[str, Any]:
    """Deterministic synthetic per-theme source counts for the `--mock` CLI run.

    Not used by the offline unit tests (which inject explicit mock_digests /
    reconstruct_digest(mock_data=...) instead for full reproducibility); this
    is only for a quick, no-network `--mock` CLI smoke run.
    """
    digest: dict[str, Any] = {}
    for theme_id in theme_ids:
        digest[theme_id] = {
            "edgar_fts": rng.randint(0, 15),
            "arxiv": rng.randint(0, 25),
            "hn_buzz": rng.randint(0, 40),
            "jobs": rng.randint(0, 10),
            "baseline": {
                "edgar_fts": rng.randint(0, 15),
                "arxiv": rng.randint(0, 25),
                "hn_buzz": rng.randint(0, 40),
                "jobs": rng.randint(0, 10),
            },
        }
    return digest


def _quartile_label(rank_fraction: float) -> str:
    if rank_fraction <= 0.25:
        return "Q1"
    if rank_fraction <= 0.50:
        return "Q2"
    if rank_fraction <= 0.75:
        return "Q3"
    return "Q4"


def _bucket_by_quartile(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    if not samples:
        return buckets
    ordered = sorted(samples, key=lambda s: s["total"])
    n = len(ordered)
    for i, s in enumerate(ordered):
        frac = (i + 1) / n
        buckets[_quartile_label(frac)].append(s)
    return buckets


def _aggregate_bucket(items: list[dict[str, Any]], horizon_key: str, primary_benchmark: str | None) -> dict[str, Any]:
    valid = [it for it in items if it.get("forward", {}).get(horizon_key) is not None]
    n = len(valid)
    if n == 0:
        return {"n": 0, "hit_rate": None, "avg_forward_return_pct": None, "avg_alpha_pct": None}
    abs_rets = [it["forward"][horizon_key]["abs"] for it in valid]
    hits = sum(1 for r in abs_rets if r > 0)
    alphas: list[float] = []
    if primary_benchmark:
        alphas = [
            it["forward"][horizon_key].get(f"vs_{primary_benchmark}")
            for it in valid
            if it["forward"][horizon_key].get(f"vs_{primary_benchmark}") is not None
        ]
    return {
        "n": n,
        "hit_rate": round(hits / n, 4),
        "avg_forward_return_pct": round(sum(abs_rets) / n, 4),
        "avg_alpha_pct": round(sum(alphas) / len(alphas), 4) if alphas else None,
    }


def _build_ic(samples: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    components = ("divergence", "theme_momentum", "breadth", "momentum_12_1", "total")
    out: dict[str, Any] = {}
    for h in horizons:
        h_key = str(h)
        valid = [s for s in samples if s["forward"].get(h_key) is not None]
        ys = [s["forward"][h_key]["abs"] for s in valid]
        row: dict[str, Any] = {"n": len(valid)}
        for comp in components:
            xs = [s["total"] for s in valid] if comp == "total" else [s["components"].get(comp, 0.0) for s in valid]
            ic = spearman(xs, ys) if len(valid) >= 3 else None
            row[comp] = round(ic, 4) if ic is not None else None
        out[h_key] = row
    return out


def _build_buckets(samples: list[dict[str, Any]], horizons: tuple[int, ...], primary_benchmark: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for h in horizons:
        h_key = str(h)
        valid = [s for s in samples if s["forward"].get(h_key) is not None]
        bucketed = _bucket_by_quartile(valid)
        out[h_key] = {label: _aggregate_bucket(items, h_key, primary_benchmark) for label, items in bucketed.items()}
    return out


def _build_filtered_buckets(
    samples: list[dict[str, Any]], primary_benchmark: str | None, horizon_key: str = "90"
) -> dict[str, Any]:
    """CONCEPT_PROFIT.md Phase B/C: does gating on trend_ok AND regime_risk_on
    actually help? Restricts to the subset of `samples` where both gates are
    True (None — insufficient trailing history, see src.analysis.trend — is
    treated as "gate not confirmed", i.e. excluded, not assumed to pass), then
    reports the TOP quartile's underlying (`_aggregate_bucket`) and synthetic
    option (`_aggregate_option_bucket`) metrics at `horizon_key` — the same
    aggregation `_build_buckets`/`_build_option_buckets` use for the ungated
    report, so the two are directly comparable.
    """
    gated = [s for s in samples if s.get("trend_ok") is True and s.get("regime_risk_on") is True]

    valid_underlying = [s for s in gated if s["forward"].get(horizon_key) is not None]
    top_underlying = _bucket_by_quartile(valid_underlying).get("Q4", [])
    underlying = _aggregate_bucket(top_underlying, horizon_key, primary_benchmark)

    valid_option = [s for s in gated if (s.get("option") or {}).get(horizon_key) is not None]
    top_option = _bucket_by_quartile(valid_option).get("Q4", [])
    option = _aggregate_option_bucket(top_option, horizon_key)

    return {
        "horizon": horizon_key,
        "n_gated": len(gated),
        "underlying": underlying,
        "option": option,
    }


def _compute_lead_lag(samples: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    """#6 lead-lag: per source, the horizon whose forward-return IC is strongest."""
    result: dict[str, Any] = {}
    for source in SOURCES:
        ic_by_horizon: dict[str, float | None] = {}
        for h in horizons:
            h_key = str(h)
            valid = [s for s in samples if s["forward"].get(h_key) is not None]
            if len(valid) < 3:
                ic_by_horizon[h_key] = None
                continue
            xs = [s["source_signals"].get(source, 0) for s in valid]
            ys = [s["forward"][h_key]["abs"] for s in valid]
            ic = spearman(xs, ys)
            ic_by_horizon[h_key] = round(ic, 4) if ic is not None else None

        scored = [(h_key, ic) for h_key, ic in ic_by_horizon.items() if ic is not None]
        if scored:
            best_h, best_ic = max(scored, key=lambda pair: abs(pair[1]))
        else:
            best_h, best_ic = None, None

        result[source] = {
            "ic_by_horizon": ic_by_horizon,
            "strongest_horizon_days": int(best_h) if best_h is not None else None,
            "strongest_ic": best_ic,
        }
    return result


def _compute_temporal_ordering(samples: list[dict[str, Any]], primary_benchmark: str | None) -> dict[str, Any]:
    """#12 temporal ordering: among eventual 90d winners, what fraction had a
    leading source (arxiv/jobs) signal while the divergence bucket was still
    high/flat at as_of (proxy: divergence component >= 70, i.e. the trailing
    price move had NOT already run hard — see scoring.score_divergence)?
    """
    horizon_key = "90"
    candidates = [s for s in samples if s["forward"].get(horizon_key) is not None]
    if not candidates:
        return {"n_winners": 0, "pct_leading_before_price_move": None}

    def _alpha(sample: dict[str, Any]) -> float:
        fwd = sample["forward"][horizon_key]
        if primary_benchmark:
            v = fwd.get(f"vs_{primary_benchmark}")
            if v is not None:
                return v
        return fwd["abs"]

    winners = [s for s in candidates if _alpha(s) > 0]
    if not winners:
        return {"n_winners": 0, "pct_leading_before_price_move": None}

    count = 0
    for s in winners:
        divergence_high_flat = s["components"].get("divergence", 0) >= 70
        leading_signal = (s["source_signals"].get("arxiv", 0) or 0) > 0 or (
            s["source_signals"].get("jobs", 0) or 0
        ) > 0
        if divergence_high_flat and leading_signal:
            count += 1

    return {
        "n_winners": len(winners),
        "pct_leading_before_price_move": round(count / len(winners) * 100, 2),
    }


def _compact_samples(samples: list[dict[str, Any]], horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    """Compact per-(ticker, as_of) records for `report["samples"]` (only
    populated when `run_walkforward(..., include_samples=True)`) — enough for
    `src/backtest/optimize.py` to recompute total scores under different
    weights and validate out-of-sample WITHOUT re-fetching any digest/price
    data:

      {"as_of", "ticker", "components": {divergence, theme_momentum, breadth,
       momentum_12_1}, "trend_ok", "regime_risk_on",
       "fwd": {horizon_str: pct_return | None},
       "opt": {horizon_str: option_pct_return | None}}

    `fwd`/`opt` cover every horizon in `horizons` (not just "90") so a caller
    can calibrate/validate against any horizon, not only the primary one;
    values are None wherever the underlying `forward`/`option` entry wasn't
    available for that (ticker, as_of, horizon) — never fabricated.
    """
    compact: list[dict[str, Any]] = []
    for s in samples:
        fwd: dict[str, float | None] = {}
        opt: dict[str, float | None] = {}
        for h in horizons:
            h_key = str(h)
            fwd_entry = s["forward"].get(h_key)
            fwd[h_key] = fwd_entry["abs"] if fwd_entry is not None else None
            opt_entry = (s.get("option") or {}).get(h_key)
            opt[h_key] = opt_entry["return"] if opt_entry is not None else None
        compact.append(
            {
                "as_of": s["as_of"],
                "ticker": s["ticker"],
                "components": dict(s["components"]),
                "trend_ok": s.get("trend_ok"),
                "regime_risk_on": s.get("regime_risk_on"),
                "fwd": fwd,
                "opt": opt,
            }
        )
    return compact


def run_walkforward(
    tradier: TradierClient | None,
    tickers: list[str],
    themes: list[dict[str, Any]],
    *,
    start: date,
    end: date,
    cadence_days: int = 30,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    mock: bool = False,
    price_series: dict[str, dict[str, float]] | None = None,
    mock_digests: dict[str, dict[str, Any]] | None = None,
    include_samples: bool = False,
) -> dict[str, Any]:
    """Run the full walk-forward backtest across an as-of date grid.

    For each as_of date in [start, end] stepped by cadence_days: reconstruct
    (or look up, if `mock_digests` is injected) the point-in-time digest,
    score the ticker universe as-of that date, and record each ticker's
    forward return (absolute, and vs. each benchmark) at every horizon,
    snapped backward-only (no look-ahead — see forward_return).

    `mock_digests`, when given, maps `as_of_date.isoformat()` -> digest (the
    same shape reconstruct_digest returns) and is used instead of any network
    call, taking priority over `mock` for any date it covers. When `mock` is
    True and a date has no entry in `mock_digests`, a deterministic synthetic
    digest is generated instead (see `_generate_mock_digest`). Otherwise
    (`mock=False`, no mock_digests) reconstruct_digest() queries live sources.

    `price_series`, when given, is used for all trailing/forward price
    lookups (offline/test) instead of `tradier`.

    Phase A (CONCEPT_PROFIT.md): in live mode (no `price_series`), a run-scoped
    in-memory `price_cache` (ticker -> {date: close}) is threaded through
    every `_trailing_perf_pct_asof` / `forward_return` / `_compute_option_metrics`
    call, backed by a per-ticker on-disk cache (see `_price_map_for`) — this
    collapses what used to be a fresh Tradier fetch per (ticker, as_of,
    horizon, +benchmark) combination down to one fetch per ticker for the
    whole run. Similarly, `reconstruct_digest`'s per-(source, theme,
    calendar-month) counts go through a persisted `counts_cache` (loaded once,
    saved after each as-of date) instead of a fresh query per (theme, window,
    as-of). Neither cache is touched when `mock_digests`/`mock`/`price_series`
    already cover a given date (offline/test paths never hit disk).

    Returns a report dict with `params`, `n_samples`, `ic_by_component` (per
    horizon, now including momentum_12_1 — CONCEPT_PROFIT.md Phase B),
    `buckets` (hit-rate & avg alpha by total-score quartile, per horizon),
    `option_by_quartile` (synthetic 120-DTE delta-0.60 call hit-rate & avg
    return by quartile @90d — see `_compute_option_metrics`),
    `filtered_buckets` (Phase B/C: top-quartile underlying+option metrics
    restricted to samples where trend_ok AND regime_risk_on are both True —
    see `_build_filtered_buckets` — measuring whether the trend/regime gates
    actually help), `lead_lag` (#6), `temporal_ordering` (#12),
    `degraded_buckets` (count of excluded arXiv-429-exhausted monthly
    buckets), and `notes`.

    `include_samples`, when True, additionally sets `report["samples"]` to a
    compact per-(ticker, as_of) list — components, trend_ok/regime_risk_on,
    forward returns and option returns by horizon — so
    `src/backtest/optimize.py` can recalibrate weights and validate
    out-of-sample WITHOUT re-fetching any data. Off by default so the report
    written by this module's own CLI/workflow stays small.
    """
    theme_ids = [t.get("id") for t in themes if t.get("id")]
    as_of_grid = _asof_grid(start, end, cadence_days)
    primary_benchmark = benchmarks[0] if benchmarks else None
    rng = random.Random(20260710)

    # Run-scoped price memoization (in-memory) + a wide-enough history_start
    # so the first fetch for any ticker (traded or benchmark) covers every
    # as-of date/horizon used anywhere in this run. See _price_map_for's
    # docstring for why this is the single biggest lever against the
    # multi-hour hangs a naive per-call fetch produced. Widened (Phase B) to
    # also cover momentum_12_1's ~253-trading-day trailing requirement, not
    # just the option-IV window, so early as-of dates aren't starved of
    # history relative to later ones (see MOMENTUM_LOOKBACK_CALENDAR_DAYS).
    price_cache: dict[str, dict[date, float]] = {}
    history_start = start - timedelta(days=max(TRAILING_WINDOW_DAYS + 60, MOMENTUM_LOOKBACK_CALENDAR_DAYS))

    # Monthly count-bucket cache (Phase A #1/#2): lazily loaded only if we
    # actually reach the live reconstruct_digest() branch below, so offline/
    # mock/test runs never touch disk.
    counts_cache: dict[str, int] | None = None
    degraded_counter = [0]

    samples: list[dict[str, Any]] = []

    for as_of in as_of_grid:
        as_of_str = as_of.isoformat()

        if mock_digests is not None and as_of_str in mock_digests:
            digest = mock_digests[as_of_str]
        elif mock:
            digest = _generate_mock_digest(theme_ids, rng)
        else:
            if counts_cache is None:
                counts_cache = _load_counts_cache()
            digest = reconstruct_digest(as_of, themes, cache=counts_cache, degraded_out=degraded_counter)
            _save_counts_cache(counts_cache)

        scored = score_universe_asof(
            as_of,
            tickers,
            digest,
            tradier,
            themes=themes,
            price_series=price_series,
            price_cache=price_cache,
            history_start=history_start,
        )

        for entry in scored:
            ticker = entry["ticker"]
            record: dict[str, Any] = {
                "as_of": as_of_str,
                "ticker": ticker,
                "components": entry["components"],
                "total": entry["total"],
                "source_signals": entry["source_signals"],
                "trend_ok": entry.get("trend_ok"),
                "regime_risk_on": entry.get("regime_risk_on"),
                "forward": {},
            }
            any_horizon_data = False
            for h in horizons:
                fwd = forward_return(
                    ticker, as_of, h, tradier, price_series=price_series,
                    price_cache=price_cache, history_start=history_start,
                )
                if fwd is None:
                    record["forward"][str(h)] = None
                    continue
                any_horizon_data = True
                horizon_entry = {"abs": fwd}
                for bench in benchmarks:
                    bfwd = forward_return(
                        bench, as_of, h, tradier, price_series=price_series,
                        price_cache=price_cache, history_start=history_start,
                    )
                    horizon_entry[f"vs_{bench}"] = round(fwd - bfwd, 4) if bfwd is not None else None
                record["forward"][str(h)] = horizon_entry
            if any_horizon_data:
                record["option"] = _compute_option_metrics(
                    ticker, as_of, tradier, price_series, horizons,
                    price_cache=price_cache, history_start=history_start,
                )
                samples.append(record)

    ic_by_component = _build_ic(samples, horizons)
    buckets = _build_buckets(samples, horizons, primary_benchmark)
    option_by_quartile = _build_option_buckets(samples, "90")
    filtered_buckets = _build_filtered_buckets(samples, primary_benchmark, "90")
    lead_lag = _compute_lead_lag(samples, horizons)
    temporal_ordering = _compute_temporal_ordering(samples, primary_benchmark)

    notes = [
        "GitHub star counts are NOT historically reconstructable (the API only "
        "returns the CURRENT star count, with no point-in-time history), so "
        "github_trends is excluded entirely from this walk-forward — the "
        "as-of score uses divergence, theme_momentum "
        "(edgar_fts+arxiv+hn_buzz+jobs), breadth, and momentum_12_1.",
        "forward_return() snaps both entry and exit strictly backward-only to "
        "the nearest available trading day, and returns None (rather than a "
        "stale estimate) when no close exists within max_snap_days of the "
        "target exit date — this avoids look-ahead bias.",
        "jobs_hn's point-in-time reconstruction uses Algolia's nbHits count "
        "for keyword-tagged 'who is hiring' comments in the date window as an "
        "approximation of the production collector's full per-thread comment "
        "scan (which is not point-in-time queryable the same way).",
        "Historical source counts (edgar_fts/arxiv/hn_buzz/jobs) are bucketed "
        "and cached per (source, theme, calendar month) rather than per "
        "(theme, window, as-of date) — a window's count is the sum of the "
        "whole calendar months it touches, which quantizes windows to month "
        "granularity in exchange for a large cut in network calls across a "
        "multi-year, many-as-of-date run (see _bucketed_counts).",
        f"{degraded_counter[0]} arXiv monthly count bucket(s) were dropped "
        "(429 exhausted its backoff — 5s/15s/45s — for that bucket) rather "
        "than recorded as a false zero; they simply contribute 0 to the "
        "affected window sums. See 'degraded_buckets' in this report.",
        "Option P/L uses a synthetic 120-DTE delta-0.60 Black-Scholes call "
        "(entry IV = trailing realized volatility, NOT a quoted IV — see "
        "price_backtest.py), re-valued at each horizon <= 120d (180d exceeds "
        "this option's modeled life and has no option entry); both entry and "
        "exit theoretical values are reduced by an assumed half-spread cost "
        f"of 0.5 * {OPTION_SPREAD_PCT} * value (a cost ASSUMPTION, not a "
        "quoted market spread — no historical options-chain data exists to "
        "backtest a real spread against).",
        "Small universes / short date ranges can produce Information "
        "Coefficients with wide sampling error — treat n_samples and each "
        "bucket's own n as the primary indicator of how much to trust any IC.",
        "momentum_12_1 (trend.score_momentum_12_1 of trailing daily closes) is "
        "now a WEIGHTED component (see WALKFORWARD_WEIGHTS); trend_ok "
        "(price > 50-SMA > 200-SMA) and regime_risk_on (SPY > 200-SMA) are "
        "stored as per-sample GATE metadata instead — 'filtered_buckets' "
        "reports the top-quartile underlying+option metrics restricted to "
        "samples where both gates are True, to measure whether trend/regime "
        "filtering actually helps rather than assuming it does. None "
        "(insufficient trailing history for a gate) is treated as 'not "
        "confirmed passing', not as True.",
    ]

    return {
        "header": HEADER_LINE,
        "params": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "cadence_days": cadence_days,
            "horizons": list(horizons),
            "benchmarks": list(benchmarks),
            "tickers": tickers,
            "theme_ids": theme_ids,
            "mock": mock,
            "n_asof_dates": len(as_of_grid),
        },
        "n_samples": len(samples),
        "ic_by_component": ic_by_component,
        "buckets": buckets,
        "option_by_quartile": option_by_quartile,
        "filtered_buckets": filtered_buckets,
        "lead_lag": lead_lag,
        "temporal_ordering": temporal_ordering,
        "degraded_buckets": degraded_counter[0],
        "notes": notes,
        **(
            {"samples": _compact_samples(samples, horizons)}
            if include_samples
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# Mock price generation (deterministic GBM, no network) for the --mock CLI
# ---------------------------------------------------------------------------

WF_TRADING_DAYS_PER_YEAR = 252

MOCK_TICKERS: dict[str, dict[str, float]] = {
    "WFA": {"drift": 0.20, "vol": 0.35},  # steady uptrend
    "WFB": {"drift": 0.05, "vol": 0.50},  # flat/choppy
    "WFC": {"drift": 0.35, "vol": 0.45},  # already ran hard
}
MOCK_BENCHMARKS: dict[str, dict[str, float]] = {
    "SPY": {"drift": 0.10, "vol": 0.18},
    "QQQ": {"drift": 0.14, "vol": 0.24},
    "SOXX": {"drift": 0.16, "vol": 0.30},
}
MOCK_THEMES: list[dict[str, Any]] = [
    {"id": "wf_mock_theme_a", "name": "Mock Theme A", "keywords": ["mockfoo"], "tickers": ["WFA"]},
    {"id": "wf_mock_theme_b", "name": "Mock Theme B", "keywords": ["mockbar"], "tickers": ["WFB", "WFC"]},
]


def _generate_mock_price_series(start: date, end: date, seed: int = 20260710) -> dict[str, dict[str, float]]:
    """Deterministic synthetic daily GBM price series covering [start - 200d, end + 250d]."""
    rng = random.Random(seed)
    dt = 1.0 / WF_TRADING_DAYS_PER_YEAR
    series_start = start - timedelta(days=200)
    n_days = (end - series_start).days + 250

    series: dict[str, dict[str, float]] = {}
    for ticker, params in {**MOCK_TICKERS, **MOCK_BENCHMARKS}.items():
        mu, sigma = params["drift"], params["vol"]
        price = 100.0
        bars: dict[str, float] = {}
        current = series_start
        for _ in range(n_days):
            if current.weekday() < 5:
                z = rng.gauss(0.0, 1.0)
                price *= math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
                bars[current.isoformat()] = round(price, 4)
            current += timedelta(days=1)
        series[ticker] = bars
    return series


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    print(report["header"])
    print()
    params = report["params"]
    print(
        f"start={params['start']} end={params['end']} cadence_days={params['cadence_days']} "
        f"horizons={params['horizons']} benchmarks={params['benchmarks']} mock={params['mock']}"
    )
    print(f"tickers ({len(params['tickers'])}): {params['tickers']}")
    print(f"themes: {params['theme_ids']}")
    print(f"n_asof_dates={params['n_asof_dates']}  n_samples={report['n_samples']}")
    print()

    def _fmt(v: Any) -> str:
        return f"{v:+.3f}" if isinstance(v, (int, float)) else "n/a"

    print("Information Coefficient (Spearman) by component and horizon:")
    print(
        f"{'horizon':>8} {'n':>5} {'divergence':>11} {'theme_mom':>10} {'breadth':>8} "
        f"{'mom_12_1':>9} {'total':>8}"
    )
    for h_key, row in report["ic_by_component"].items():
        print(
            f"{h_key + 'd':>8} {row['n']:>5} {_fmt(row.get('divergence')):>11} "
            f"{_fmt(row.get('theme_momentum')):>10} {_fmt(row.get('breadth')):>8} "
            f"{_fmt(row.get('momentum_12_1')):>9} {_fmt(row.get('total')):>8}"
        )
    print()

    print("Hit-rate & avg forward alpha by total-score quartile:")
    for h_key, buckets in report["buckets"].items():
        print(f"  horizon={h_key}d")
        for label in ("Q1", "Q2", "Q3", "Q4"):
            b = buckets.get(label, {})
            if b.get("n"):
                aa = b["avg_alpha_pct"]
                aa_str = f"{aa:+.2f}%" if aa is not None else "n/a"
                print(
                    f"    {label}: n={b['n']:>4} hit_rate={b['hit_rate']:.1%} "
                    f"avg_return={b['avg_forward_return_pct']:+.2f}% avg_alpha={aa_str}"
                )
            else:
                print(f"    {label}: no samples")
    print()

    print("Lead-lag (#6) — strongest IC horizon per source:")
    for source, info in report["lead_lag"].items():
        sh = info.get("strongest_horizon_days")
        si = info.get("strongest_ic")
        if sh is not None:
            print(f"  {source:<10}: strongest at {sh}d (IC={si:+.3f})  ic_by_horizon={info['ic_by_horizon']}")
        else:
            print(f"  {source:<10}: not enough data")
    print()

    to = report["temporal_ordering"]
    print("Temporal ordering (#12):")
    if to.get("pct_leading_before_price_move") is not None:
        print(
            f"  {to['pct_leading_before_price_move']:.1f}% of {to['n_winners']} eventual 90d winners had a "
            "leading source (arxiv/jobs) signal while divergence was still high/flat at as_of."
        )
    else:
        print(f"  Not enough 90d winners to compute ({to.get('n_winners', 0)} winners).")
    print()

    fb = report.get("filtered_buckets") or {}
    print("Trend+regime gate (Phase B/C) — top-quartile @90d on samples where trend_ok AND regime_risk_on:")
    fu = fb.get("underlying") or {}
    fo = fb.get("option") or {}

    def _rate(v: Any) -> str:
        return f"{v:.1%}" if isinstance(v, (int, float)) else "n/a"

    print(
        f"  n_gated={fb.get('n_gated', 0)}  underlying: n={fu.get('n', 0)} hit_rate={_rate(fu.get('hit_rate'))}"
        f"  |  option: n={fo.get('n', 0)} hit_rate={_rate(fo.get('hit_rate'))}"
    )
    print()

    print("Notes / caveats:")
    for note in report["notes"]:
        print(f"  - {note}")


def _fmt_ic(v: Any) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "n/a"


def _print_summary(report: dict[str, Any]) -> None:
    """Print a compact, clearly-delimited, grep-friendly summary block.

    Purely additive to `_print_report` (same report dict, no new computation):
    intended to make CI/Actions logs easy to scan without wading through the
    full tabular report above. Defensive against missing/partial keys (e.g. a
    horizon or source with too few samples) — always prints "n/a" rather than
    raising.
    """
    print("=== WALK-FORWARD SUMMARY ===")
    print(f"samples: {report.get('n_samples', 'n/a')}")

    ic90 = (report.get("ic_by_component") or {}).get("90") or {}
    ic_parts = ", ".join(
        f"{comp}={_fmt_ic(ic90.get(comp))}"
        for comp in ("divergence", "theme_momentum", "breadth", "momentum_12_1", "total")
    )
    print(f"IC (Spearman) by component @90d: {ic_parts}")

    buckets90 = (report.get("buckets") or {}).get("90") or {}
    hit_parts = []
    for label in ("Q4", "Q3", "Q2", "Q1"):
        b = buckets90.get(label) or {}
        hr = b.get("hit_rate")
        hit_parts.append(f"{label}={hr:.1%}" if isinstance(hr, (int, float)) else f"{label}=n/a")
    print(f"hit-rate by top-score quartile @90d: {', '.join(hit_parts)}")

    opt90 = report.get("option_by_quartile") or {}
    opt_parts = []
    for label in ("Q4", "Q3", "Q2", "Q1"):
        b = opt90.get(label) or {}
        hr = b.get("hit_rate")
        ar = b.get("avg_option_return_pct")
        if isinstance(hr, (int, float)):
            ar_str = f"/{ar:+.1f}%" if isinstance(ar, (int, float)) else ""
            opt_parts.append(f"{label}={hr:.1%}{ar_str}")
        else:
            opt_parts.append(f"{label}=n/a")
    print(f"option hit-rate/avg-return by quartile @90d: {', '.join(opt_parts)}")

    lead_lag = report.get("lead_lag") or {}
    ll_parts = []
    for source, info in lead_lag.items():
        info = info or {}
        sh = info.get("strongest_horizon_days")
        si = info.get("strongest_ic")
        if sh is not None and isinstance(si, (int, float)):
            ll_parts.append(f"{source}@{sh}d(IC={si:+.3f})")
        else:
            ll_parts.append(f"{source}=n/a")
    print(f"lead-lag (best horizon per source): {', '.join(ll_parts) if ll_parts else 'n/a'}")

    to = report.get("temporal_ordering") or {}
    pct = to.get("pct_leading_before_price_move")
    if isinstance(pct, (int, float)):
        print(f"temporal-ordering: {pct:.1f}%")
    else:
        print("temporal-ordering: n/a")

    fb = report.get("filtered_buckets") or {}
    fu = fb.get("underlying") or {}
    fo = fb.get("option") or {}
    u_hit = fu.get("hit_rate")
    o_hit = fo.get("hit_rate")
    u_hit_str = f"{u_hit:.1%}" if isinstance(u_hit, (int, float)) else "n/a"
    o_hit_str = f"{o_hit:.1%}" if isinstance(o_hit, (int, float)) else "n/a"
    print(
        f"trend+regime-filtered top-quartile @90d: hit={u_hit_str} option_hit={o_hit_str} "
        f"(n={fu.get('n', 0)})"
    )

    notes = report.get("notes") or []
    print(f"notes: {' | '.join(notes) if notes else 'n/a'}")
    print("=== END SUMMARY ===")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NXT LVL — walk-forward out-of-sample backtest (point-in-time digest reconstruction)"
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: 3 years before --end)")
    parser.add_argument(
        "--end", type=str, default=None, help="End date YYYY-MM-DD (default: today minus the longest horizon)"
    )
    parser.add_argument("--cadence", type=int, default=30, dest="cadence_days", help="Days between as-of samples (default 30)")
    parser.add_argument("--mock", action="store_true", help="Use deterministic synthetic data, no network required")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    args = parse_args(argv)

    horizons = DEFAULT_HORIZONS
    end = _parse_date(args.end) if args.end else date.today() - timedelta(days=max(horizons))
    start = _parse_date(args.start) if args.start else end - timedelta(days=365 * 3)

    if args.mock:
        price_series = _generate_mock_price_series(start, end)
        tickers = list(MOCK_TICKERS.keys())
        themes = MOCK_THEMES
        report = run_walkforward(
            None,
            tickers,
            themes,
            start=start,
            end=end,
            cadence_days=args.cadence_days,
            horizons=horizons,
            benchmarks=DEFAULT_BENCHMARKS,
            mock=True,
            price_series=price_series,
        )
    else:
        settings = load_settings()
        if not settings.tradier_api_key:
            print(
                "TRADIER_API_KEY is not set. Live mode needs a Tradier API key to fetch "
                "historical prices. Set TRADIER_API_KEY, or run with --mock for an "
                "offline synthetic-data run."
            )
            return 1
        tradier = TradierClient(settings.tradier_api_key, settings.tradier_env)
        tickers = sorted(
            settings.watchlist_tickers()
            | {str(t).upper() for theme in settings.themes for t in (theme.get("tickers") or [])}
        )
        themes = settings.themes
        if not tickers or not themes:
            print("No tickers/themes found in config.yaml (stages[].tickers / themes[]).")
            return 1
        report = run_walkforward(
            tradier,
            tickers,
            themes,
            start=start,
            end=end,
            cadence_days=args.cadence_days,
            horizons=horizons,
            benchmarks=DEFAULT_BENCHMARKS,
            mock=False,
        )

    _print_report(report)
    _print_summary(report)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"\nReport written to {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
