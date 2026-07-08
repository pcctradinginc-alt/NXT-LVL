"""SEC EDGAR collector: capex growth of the large AI infrastructure spenders.

Pulls quarterly capex XBRL facts for the configured `capex_companies` and
derives QoQ / YoY growth. This is a leading indicator for stage 2-3
(datacenter buildout, power & cooling demand).

No API key required, but the SEC requires a descriptive User-Agent with a
contact address — see http_utils.DEFAULT_USER_AGENT.

Real-world XBRL quirk this module has to handle: `PaymentsToAcquirePropertyPlantAndEquipment`
is the standard capex tag, but several large filers (Amazon since FY2017, Nvidia
since FY2020) stop tagging new filings under it and switch to
`PaymentsToAcquireProductiveAssets` instead — the old tag then simply has no
recent data. We therefore try a short list of known-equivalent concepts per
company and use whichever yields the most recent facts.

A second quirk: many filers (Google, Meta, Oracle, and Microsoft/Nvidia in
some quarters) report only **fiscal-year-to-date cumulative** values in this
XBRL concept (e.g. "9 months ended Sep 30" instead of "Q3 alone"), rather than
discrete quarterly amounts. When no genuine short-duration (~1 quarter) fact
is available for a period, we reconstruct the discrete quarter by
differencing consecutive year-to-date cumulative facts that share the same
fiscal-year start date (Q1 = YTD_Q1, Q2 = YTD_H1 - YTD_Q1, etc.), falling back
to the annual 10-K minus the 9-month YTD for Q4.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

from src.http_utils import get_json

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
CONCEPT_URL_TMPL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json"
MAX_QUARTER_SPAN_DAYS = 120

# Concepts tried in order per company; the first one with usable (non-stale)
# data wins. PaymentsToAcquirePropertyPlantAndEquipment is the concept named
# in CONCEPT.md and is tried first for every ticker; PaymentsToAcquireProductiveAssets
# is the documented fallback for filers who migrated away from it.
CONCEPT_CANDIDATES = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
]

# A fact is considered "stale" (not worth preferring over a fallback concept)
# if its most recent `end` date is older than this many days.
STALE_AFTER_DAYS = 400


def _parse_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def build_ticker_to_cik_map() -> dict[str, int]:
    """Fetch the SEC ticker->CIK map once. Returns {} on failure."""
    try:
        raw = get_json(TICKERS_URL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_capex: failed to load company_tickers.json: %s", exc)
        return {}

    mapping: dict[str, int] = {}
    try:
        for entry in raw.values():
            ticker = str(entry.get("ticker", "")).upper()
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                mapping[ticker] = int(cik)
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_capex: malformed company_tickers.json: %s", exc)
        return {}
    return mapping


def _dedupe_by_period(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate (start, end) facts, keeping the most recently filed one."""
    by_period: dict[tuple[str, str], dict[str, Any]] = {}
    for fact in facts:
        start, end = fact.get("start"), fact.get("end")
        if not start or not end:
            continue
        key = (start, end)
        existing = by_period.get(key)
        if existing is None or fact.get("filed", "") >= existing.get("filed", ""):
            by_period[key] = fact
    return list(by_period.values())


def _reconstruct_quarters(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn raw (possibly YTD-cumulative) XBRL facts into discrete quarters.

    Algorithm per fiscal-year group (grouped by `start` date, since YTD facts
    within one fiscal year all share the same start):
    1. If a fact's own span is already <= MAX_QUARTER_SPAN_DAYS, it is a
       genuine discrete quarter — use it directly.
    2. Otherwise, sort the fiscal year's cumulative facts by `end` date and
       take consecutive differences: quarter_n = cumulative_n - cumulative_(n-1).
       The first cumulative fact in the group (typically Q1) is used as-is.
    """
    deduped = _dedupe_by_period(facts)

    genuine_quarters: dict[str, dict[str, Any]] = {}
    cumulative_by_start: dict[str, list[dict[str, Any]]] = {}

    for fact in deduped:
        start_date = _parse_date(fact.get("start", ""))
        end_date = _parse_date(fact.get("end", ""))
        if start_date is None or end_date is None:
            continue
        span = (end_date - start_date).days
        if 0 < span <= MAX_QUARTER_SPAN_DAYS:
            # Genuine discrete quarter; prefer the most recently filed if we
            # somehow see the same `end` twice with different spans.
            existing = genuine_quarters.get(fact["end"])
            if existing is None or fact.get("filed", "") >= existing.get("filed", ""):
                genuine_quarters[fact["end"]] = fact
        else:
            cumulative_by_start.setdefault(fact["start"], []).append(fact)

    reconstructed: dict[str, dict[str, Any]] = dict(genuine_quarters)

    # Genuine quarters, indexed by their `start` date, so a cumulative group
    # sharing that fiscal-year start can seed its running total from the
    # correct prior discrete quarter (e.g. a standalone Q1 fact must offset
    # a later "6 months ended" cumulative fact for the same fiscal year).
    genuine_by_start: dict[str, list[dict[str, Any]]] = {}
    for fact in genuine_quarters.values():
        genuine_by_start.setdefault(fact["start"], []).append(fact)
    for facts in genuine_by_start.values():
        facts.sort(key=lambda f: f["end"])

    for start, group in cumulative_by_start.items():
        group.sort(key=lambda f: f["end"])

        # Seed the running total from any genuine quarter(s) that share this
        # fiscal-year start and end before the first cumulative fact.
        prev_val = None
        prev_end = None
        for genuine_fact in genuine_by_start.get(start, []):
            if genuine_fact["end"] < group[0]["end"]:
                base = prev_val or 0
                prev_val = base + (genuine_fact.get("val") or 0)
                prev_end = genuine_fact["end"]

        for fact in group:
            end = fact["end"]
            val = fact.get("val")
            if end in reconstructed:
                # Already have a genuine quarter for this end date; still use
                # this cumulative fact to seed the running total for later
                # quarters in the same fiscal year.
                prev_val = val
                prev_end = end
                continue
            if prev_val is None:
                # First cumulative period in the fiscal year (usually Q1) —
                # its value already represents a single quarter.
                derived_val = val
            else:
                derived_val = val - prev_val if val is not None else None
            reconstructed[end] = {
                "end": end,
                "start": prev_end or fact.get("start"),
                "val": derived_val,
                "form": fact.get("form"),
                "filed": fact.get("filed"),
            }
            prev_val = val
            prev_end = end

    series = list(reconstructed.values())
    series.sort(key=lambda f: f["end"])
    return series


def _growth_pct(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return round((current - prior) / abs(prior) * 100, 2)


def _fetch_concept(cik: int, concept: str) -> list[dict[str, Any]]:
    url = CONCEPT_URL_TMPL.format(cik=cik, concept=concept)
    payload = get_json(url)
    return payload.get("units", {}).get("USD", [])


def _fetch_company_capex(cik: int) -> list[dict[str, Any]] | None:
    """Fetch and reconstruct the quarterly capex series, trying fallback concepts.

    Tries each concept in CONCEPT_CANDIDATES in order and keeps the result
    from the first concept whose most recent quarter is not stale. If every
    concept attempt fails or all are stale, returns the best (most recent)
    series found, or None if nothing was retrievable at all.
    """
    best_series: list[dict[str, Any]] | None = None
    best_recency: date | None = None

    for concept in CONCEPT_CANDIDATES:
        try:
            raw_facts = _fetch_concept(cik, concept)
        except Exception as exc:  # noqa: BLE001
            logger.info("edgar_capex: concept %s unavailable for CIK %s: %s", concept, cik, exc)
            continue

        try:
            series = _reconstruct_quarters(raw_facts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("edgar_capex: failed to reconstruct quarters for CIK %s/%s: %s", cik, concept, exc)
            continue

        if not series:
            continue

        latest_end = _parse_date(series[-1]["end"])
        is_stale = latest_end is None or (date.today() - latest_end).days > STALE_AFTER_DAYS

        if best_series is None or (latest_end is not None and (best_recency is None or latest_end > best_recency)):
            best_series = series
            best_recency = latest_end

        if not is_stale:
            # Good enough — no need to try further fallback concepts.
            return series

    return best_series


def collect(tickers: list[str] | None = None) -> dict[str, Any]:
    """Collect quarterly capex growth metrics for the configured companies.

    Returns a compact, JSON-serializable dict. Never raises — on any failure
    it returns as much partial data as it could gather (possibly empty).
    """
    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("edgar_capex: NXT_OFFLINE=1, skipping network calls")
        return {"source": "edgar_capex", "companies": {}, "aggregate_capex_yoy_pct": None}

    tickers = tickers or ["MSFT", "GOOGL", "AMZN", "META", "ORCL", "NVDA"]
    result: dict[str, Any] = {"source": "edgar_capex", "companies": {}, "aggregate_capex_yoy_pct": None}

    try:
        ticker_to_cik = build_ticker_to_cik_map()
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_capex: unexpected error building CIK map: %s", exc)
        ticker_to_cik = {}

    if not ticker_to_cik:
        logger.warning("edgar_capex: no CIK map available, returning empty result")
        return result

    yoy_values: list[float] = []

    for ticker in tickers:
        cik = ticker_to_cik.get(ticker.upper())
        if cik is None:
            logger.warning("edgar_capex: no CIK found for ticker %s", ticker)
            continue

        try:
            series = _fetch_company_capex(cik)
        except Exception as exc:  # noqa: BLE001
            logger.warning("edgar_capex: unexpected error for %s: %s", ticker, exc)
            series = None

        if not series:
            continue

        last4 = series[-4:]
        last_val = last4[-1]["val"] if last4 else None
        qoq_prior = last4[-2]["val"] if len(last4) >= 2 else None
        # YoY needs a quarter ~4 quarters back; use series length to find it.
        yoy_prior = series[-5]["val"] if len(series) >= 5 else None

        qoq_growth = _growth_pct(last_val, qoq_prior)
        yoy_growth = _growth_pct(last_val, yoy_prior)

        if yoy_growth is not None:
            yoy_values.append(yoy_growth)

        result["companies"][ticker.upper()] = {
            "last_4_quarters": [
                {"end": q["end"], "val": q["val"], "form": q["form"]} for q in last4
            ],
            "qoq_growth_pct": qoq_growth,
            "yoy_growth_pct": yoy_growth,
        }

    if yoy_values:
        result["aggregate_capex_yoy_pct"] = round(sum(yoy_values) / len(yoy_values), 2)

    logger.info(
        "edgar_capex: collected data for %d/%d companies", len(result["companies"]), len(tickers)
    )
    return result
