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
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.analysis import scoring
from src.backtest.calibrate import spearman
from src.backtest.price_backtest import _build_price_map, _nearest_trading_day, _parse_date
from src.config import DATA_DIR, load_settings
from src.http_utils import get_json, get_text
from src.options.tradier import TradierClient

logger = logging.getLogger(__name__)

REPORT_PATH = DATA_DIR / "walkforward_report.json"

DEFAULT_HORIZONS: tuple[int, ...] = (30, 60, 90, 180)
DEFAULT_BENCHMARKS: tuple[str, ...] = ("SPY", "QQQ", "SOXX")
DEFAULT_RECONSTRUCT_WINDOW_DAYS = 90
SOURCES = ("edgar_fts", "arxiv", "hn_buzz", "jobs")

# Simple, documented weights for the as-of walk-forward score. These are
# intentionally simpler than scoring.DEFAULT_WEIGHTS (no option_quality /
# stage_fit / emergence — those either need live option chains or the full
# emergence pipeline, neither of which is point-in-time reconstructable).
WALKFORWARD_WEIGHTS = {
    "divergence": 0.40,
    "theme_momentum": 0.35,
    "breadth": 0.25,
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


def _hist_arxiv_counts(themes: list[dict[str, Any]], start: date, end: date) -> dict[str, int]:
    """Historical per-theme arXiv paper counts submitted within [start, end].

    Uses arXiv's `submittedDate` range filter plus the single most specific
    keyword per theme (same heuristic as edgar_fts._pick_keyword), reading
    the Atom feed's `opensearch:totalResults` as the count. This is lighter
    than the production arxiv_trends collector (which fetches MAX_RESULTS
    recent papers and counts keyword hits locally) because that collector is
    not date-range scoped in the first place.
    """
    result: dict[str, int] = {}
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
        try:
            xml_text = get_text(
                "http://export.arxiv.org/api/query",
                params={"search_query": query, "max_results": 1},
            )
            result[theme_id] = _extract_arxiv_total(xml_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("walkforward._hist_arxiv_counts: failed for theme %s: %s", theme_id, exc)
            result[theme_id] = 0
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


def reconstruct_digest(
    as_of_date: date,
    themes: list[dict[str, Any]],
    *,
    window_days: int = DEFAULT_RECONSTRUCT_WINDOW_DAYS,
    mock_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact point-in-time digest for `as_of_date`.

    Queries the reconstructable sources (edgar_fts, arxiv, hn_buzz, jobs —
    GitHub omitted, see module docstring) over a trailing `window_days`
    window ending at `as_of_date`, plus the immediately preceding window of
    the same length as a momentum baseline.

    Returns:
      {theme_id: {"edgar_fts": int, "arxiv": int, "hn_buzz": int, "jobs": int,
                   "baseline": {"edgar_fts": int, "arxiv": int, "hn_buzz": int, "jobs": int}}}

    If `mock_data` is given, it is returned as-is and no network call is made
    at all — used for offline/test runs.

    Fault-tolerant per source: a failure in one source's query never blocks
    the others (each `_hist_*` helper already degrades to 0 per theme
    internally; the try/except here is an extra safety net).
    """
    if mock_data is not None:
        return mock_data

    theme_ids = [t.get("id") for t in themes if t.get("id")]
    current_start = as_of_date - timedelta(days=window_days)
    baseline_end = current_start
    baseline_start = baseline_end - timedelta(days=window_days)

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

    def _merge(field: str, counts: dict[str, int], into_baseline: bool) -> None:
        for theme_id, value in counts.items():
            if theme_id not in digest:
                continue
            if into_baseline:
                digest[theme_id]["baseline"][field] = value
            else:
                digest[theme_id][field] = value

    try:
        _merge("edgar_fts", _hist_edgar_fts(themes, current_start, as_of_date), False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: edgar_fts current window failed: %s", exc)
    try:
        _merge("edgar_fts", _hist_edgar_fts(themes, baseline_start, baseline_end), True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: edgar_fts baseline window failed: %s", exc)

    try:
        _merge("arxiv", _hist_arxiv_counts(themes, current_start, as_of_date), False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: arxiv current window failed: %s", exc)
    try:
        _merge("arxiv", _hist_arxiv_counts(themes, baseline_start, baseline_end), True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: arxiv baseline window failed: %s", exc)

    try:
        hn = _hist_hn_buzz(themes, current_start, as_of_date)
        _merge("hn_buzz", {tid: v.get("stories", 0) + v.get("points", 0) for tid, v in hn.items()}, False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: hn_buzz current window failed: %s", exc)
    try:
        hn_base = _hist_hn_buzz(themes, baseline_start, baseline_end)
        _merge(
            "hn_buzz", {tid: v.get("stories", 0) + v.get("points", 0) for tid, v in hn_base.items()}, True
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: hn_buzz baseline window failed: %s", exc)

    try:
        _merge("jobs", _hist_jobs_hn(themes, current_start, as_of_date), False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: jobs current window failed: %s", exc)
    try:
        _merge("jobs", _hist_jobs_hn(themes, baseline_start, baseline_end), True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconstruct_digest: jobs baseline window failed: %s", exc)

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
) -> dict[date, float]:
    """Build a {date: close} map for `ticker`.

    Uses `price_series` (offline/test) verbatim when given, else queries
    `tradier.get_history` starting `lookback_days` before `as_of_date`
    (Tradier's history endpoint fetches through "today", which is fine here
    since in live mode `as_of_date` is always in the past).
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
    start = (as_of_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        bars = tradier.get_history(ticker, start=start)
    except Exception as exc:  # noqa: BLE001
        logger.warning("walkforward: history fetch failed for %s: %s", ticker, exc)
        return {}
    return _build_price_map(bars)


def _trailing_perf_pct_asof(
    ticker: str,
    as_of_date: date,
    tradier: TradierClient | None,
    price_series: dict[str, dict[str, float]] | None,
    window_days: int = 90,
) -> float | None:
    """Trailing `window_days` % price move ending at (the nearest trading day <=) as_of_date."""
    price_map = _price_map_for(ticker, tradier, price_series, as_of_date, lookback_days=window_days + 30)
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


def score_universe_asof(
    as_of_date: date,
    tickers: list[str],
    digest: dict[str, Any],
    tradier: TradierClient | None,
    *,
    themes: list[dict[str, Any]] | None = None,
    price_series: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Simplified as-of score for each ticker, from reconstructable components only.

    `themes` (same shape as config.yaml's `themes` list — each with "id" and
    "tickers") is used to map each ticker to its theme(s) in `digest`, driving
    the theme_momentum and breadth components. Without it (or for tickers
    mapped to no theme), theme_momentum falls back to neutral (50) and
    breadth to 0 (no thematic evidence found for that ticker).

    Returns a list of
      {"ticker", "as_of", "components": {"divergence", "theme_momentum", "breadth"},
       "total", "source_signals": {source: raw_current_count}}
    (the extra "source_signals" field feeds the lead-lag analysis in
    run_walkforward and is not part of the documented minimal contract).
    """
    themes = themes or []
    ticker_theme_map = _ticker_theme_ids(tickers, themes)

    results: list[dict[str, Any]] = []
    for ticker in tickers:
        perf_3m = _trailing_perf_pct_asof(ticker, as_of_date, tradier, price_series)
        divergence = scoring.score_divergence(perf_3m)

        theme_ids = ticker_theme_map.get(ticker, [])
        signal = _theme_signal(theme_ids, digest)
        theme_momentum = _theme_momentum_score(signal["current"], signal["baseline"])
        breadth = _breadth_score(signal["current"])

        total = _clip(
            divergence * WALKFORWARD_WEIGHTS["divergence"]
            + theme_momentum * WALKFORWARD_WEIGHTS["theme_momentum"]
            + breadth * WALKFORWARD_WEIGHTS["breadth"]
        )

        results.append(
            {
                "ticker": ticker,
                "as_of": as_of_date.isoformat(),
                "components": {
                    "divergence": round(divergence, 2),
                    "theme_momentum": round(theme_momentum, 2),
                    "breadth": round(breadth, 2),
                },
                "total": round(total, 2),
                "source_signals": dict(signal["current"]),
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
    price_map = _price_map_for(ticker, tradier, price_series, as_of_date, lookback_days=10)
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
    components = ("divergence", "theme_momentum", "breadth", "total")
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

    Returns a report dict with `params`, `n_samples`, `ic_by_component` (per
    horizon), `buckets` (hit-rate & avg alpha by total-score quartile, per
    horizon), `lead_lag` (#6), `temporal_ordering` (#12), and `notes`.
    """
    theme_ids = [t.get("id") for t in themes if t.get("id")]
    as_of_grid = _asof_grid(start, end, cadence_days)
    primary_benchmark = benchmarks[0] if benchmarks else None
    rng = random.Random(20260710)

    samples: list[dict[str, Any]] = []

    for as_of in as_of_grid:
        as_of_str = as_of.isoformat()

        if mock_digests is not None and as_of_str in mock_digests:
            digest = mock_digests[as_of_str]
        elif mock:
            digest = _generate_mock_digest(theme_ids, rng)
        else:
            digest = reconstruct_digest(as_of, themes)

        scored = score_universe_asof(as_of, tickers, digest, tradier, themes=themes, price_series=price_series)

        for entry in scored:
            ticker = entry["ticker"]
            record: dict[str, Any] = {
                "as_of": as_of_str,
                "ticker": ticker,
                "components": entry["components"],
                "total": entry["total"],
                "source_signals": entry["source_signals"],
                "forward": {},
            }
            any_horizon_data = False
            for h in horizons:
                fwd = forward_return(ticker, as_of, h, tradier, price_series=price_series)
                if fwd is None:
                    record["forward"][str(h)] = None
                    continue
                any_horizon_data = True
                horizon_entry = {"abs": fwd}
                for bench in benchmarks:
                    bfwd = forward_return(bench, as_of, h, tradier, price_series=price_series)
                    horizon_entry[f"vs_{bench}"] = round(fwd - bfwd, 4) if bfwd is not None else None
                record["forward"][str(h)] = horizon_entry
            if any_horizon_data:
                samples.append(record)

    ic_by_component = _build_ic(samples, horizons)
    buckets = _build_buckets(samples, horizons, primary_benchmark)
    lead_lag = _compute_lead_lag(samples, horizons)
    temporal_ordering = _compute_temporal_ordering(samples, primary_benchmark)

    notes = [
        "GitHub star counts are NOT historically reconstructable (the API only "
        "returns the CURRENT star count, with no point-in-time history), so "
        "github_trends is excluded entirely from this walk-forward — the "
        "as-of score uses only divergence, theme_momentum "
        "(edgar_fts+arxiv+hn_buzz+jobs), and breadth.",
        "forward_return() snaps both entry and exit strictly backward-only to "
        "the nearest available trading day, and returns None (rather than a "
        "stale estimate) when no close exists within max_snap_days of the "
        "target exit date — this avoids look-ahead bias.",
        "jobs_hn's point-in-time reconstruction uses Algolia's nbHits count "
        "for keyword-tagged 'who is hiring' comments in the date window as an "
        "approximation of the production collector's full per-thread comment "
        "scan (which is not point-in-time queryable the same way).",
        "This walk-forward measures underlying % returns only; it does not "
        "model option P/L (see price_backtest.py for the Black-Scholes "
        "option-mechanics backtest).",
        "Small universes / short date ranges can produce Information "
        "Coefficients with wide sampling error — treat n_samples and each "
        "bucket's own n as the primary indicator of how much to trust any IC.",
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
        "lead_lag": lead_lag,
        "temporal_ordering": temporal_ordering,
        "notes": notes,
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
    print(f"{'horizon':>8} {'n':>5} {'divergence':>11} {'theme_mom':>10} {'breadth':>8} {'total':>8}")
    for h_key, row in report["ic_by_component"].items():
        print(
            f"{h_key + 'd':>8} {row['n']:>5} {_fmt(row.get('divergence')):>11} "
            f"{_fmt(row.get('theme_momentum')):>10} {_fmt(row.get('breadth')):>8} {_fmt(row.get('total')):>8}"
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
        f"{comp}={_fmt_ic(ic90.get(comp))}" for comp in ("divergence", "theme_momentum", "breadth", "total")
    )
    print(f"IC (Spearman) by component @90d: {ic_parts}")

    buckets90 = (report.get("buckets") or {}).get("90") or {}
    hit_parts = []
    for label in ("Q4", "Q3", "Q2", "Q1"):
        b = buckets90.get(label) or {}
        hr = b.get("hit_rate")
        hit_parts.append(f"{label}={hr:.1%}" if isinstance(hr, (int, float)) else f"{label}=n/a")
    print(f"hit-rate by top-score quartile @90d: {', '.join(hit_parts)}")

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
