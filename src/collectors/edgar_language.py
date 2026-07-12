"""SEC EDGAR collector: AI-buildout language ACCELERATION in hyperscaler filings.

The alpha here is in the rate-of-change of specific phrases (backlog,
capacity constrained, liquid cooling, ...) between a company's two most
recent 10-Q/10-K filings, not the raw mention count — a company that
suddenly starts talking a lot more about "capacity constrained" or "grid
interconnect" than it did last quarter is signaling something that a static
keyword count misses.

No API key required, but the SEC requires a descriptive User-Agent with a
contact address — see http_utils.DEFAULT_USER_AGENT (reused automatically by
get_json/get_text). CIK resolution reuses edgar_capex.build_ticker_to_cik_map
(shared, in-process-cached).

Fault-tolerant throughout: any per-company failure (missing CIK, network
error, malformed submissions payload, HTML parse issue) is caught and simply
skips that company; the run never breaks. NXT_OFFLINE=1 short-circuits
before any network call.
"""

from __future__ import annotations

import html as html_module
import logging
import os
import re
import time
from typing import Any

from src.collectors.edgar_capex import build_ticker_to_cik_map
from src.http_utils import get_json, get_text

logger = logging.getLogger(__name__)

SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}/{primary_document}"

# Cap the amount of a filing's raw HTML we process, purely for safety/speed —
# 10-Q/10-K primary documents can be several MB; the phrases we care about
# almost always show up in the MD&A section near the front.
MAX_DOC_CHARS = 2_000_000

MAX_REQUESTS = 20
REQUEST_PAUSE_SECONDS = 0.2

DEFAULT_COMPANIES = ["MSFT", "GOOGL", "AMZN", "META", "ORCL", "NVDA"]
DEFAULT_PHRASES = [
    "ai demand",
    "backlog",
    "capacity constrained",
    "lead times",
    "liquid cooling",
    "grid interconnect",
    "custom silicon",
    "data center",
    "power constraints",
    "customer concentration",
]

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# MD&A section markers (#5 hardening): narrative language ("backlog is
# accelerating", "capacity constrained", ...) concentrates in the
# Management's Discussion & Analysis section. Slicing down to it before
# counting avoids picking up unrelated boilerplate/XBRL-taxonomy text
# elsewhere in the document. Both are lowercase literals since callers pass
# already-lowercased text (see _html_to_text).
_MDNA_START_MARKER = "management's discussion and analysis"
_MDNA_END_MARKERS = (
    "quantitative and qualitative disclosures about market risk",
    "controls and procedures",
)


def _html_to_text(raw_html: str) -> str:
    """Crude HTML->text: strip tags, unescape entities, lowercase, collapse whitespace.

    Deliberately not a real HTML parser (no external deps beyond stdlib) —
    good enough to count phrase occurrences in filing prose, and any garbling
    at the edges just slightly under/over-counts rather than crashing.
    """
    text = _TAG_RE.sub(" ", raw_html)
    text = html_module.unescape(text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _extract_mdna(text: str) -> str:
    """Slice `text` down to the Management's Discussion & Analysis section.

    Finds the first occurrence of `_MDNA_START_MARKER`, then the nearest
    subsequent occurrence of any `_MDNA_END_MARKERS` entry, and returns the
    text in between. Falls back to the FULL `text` when either the start
    marker is absent, or no end marker follows it — a filing whose headings
    don't match this boilerplate should still get counted (just without the
    section focus), not silently lose all its data.

    Pure and side-effect-free so it's trivially unit-testable. Expects
    already-lowercased text (as produced by `_html_to_text`).
    """
    start_idx = text.find(_MDNA_START_MARKER)
    if start_idx == -1:
        return text

    search_from = start_idx + len(_MDNA_START_MARKER)
    end_idx = -1
    for marker in _MDNA_END_MARKERS:
        idx = text.find(marker, search_from)
        if idx != -1 and (end_idx == -1 or idx < end_idx):
            end_idx = idx

    if end_idx == -1:
        return text

    return text[start_idx:end_idx]


def _phrase_deltas(
    latest_text: str, prior_text: str, phrases: list[str]
) -> dict[str, dict[str, int]]:
    """Pure helper: count each phrase's occurrences in latest vs. prior text.

    Returns {phrase: {"latest": n, "prior": n, "delta": n}}. Case-insensitive
    (callers are expected to pass already-lowercased text, but phrases are
    lowercased here too for safety). Counts use word-boundary regex matching
    (`\\bphrase\\b`) rather than naive substring counting, so a phrase like
    "cloud" doesn't get credited for appearing inside an unrelated longer
    token such as "intelligentcloudsegmentmember" (XBRL-taxonomy noise).
    Multi-word phrases still match literally (internal spaces are treated as
    plain characters), just anchored so they can't match inside a longer word.
    """
    result: dict[str, dict[str, int]] = {}
    for phrase in phrases:
        needle = phrase.lower()
        pattern = re.compile(r"\b" + re.escape(needle) + r"\b")
        latest_count = len(pattern.findall(latest_text))
        prior_count = len(pattern.findall(prior_text))
        result[phrase] = {
            "latest": latest_count,
            "prior": prior_count,
            "delta": latest_count - prior_count,
        }
    return result


def _recent_10q_or_10k(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the two most recent 10-Q/10-K filings (newest first) from a submissions payload."""
    try:
        recent = submissions.get("filings", {}).get("recent", {}) or {}
        forms = recent.get("form") or []
        accession_numbers = recent.get("accessionNumber") or []
        primary_docs = recent.get("primaryDocument") or []
        filing_dates = recent.get("filingDate") or []
    except AttributeError:
        return []

    filings: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form not in ("10-Q", "10-K"):
            continue
        try:
            accession = accession_numbers[i]
            primary_doc = primary_docs[i]
        except IndexError:
            continue
        if not accession or not primary_doc:
            continue
        filing_date = filing_dates[i] if i < len(filing_dates) else None
        filings.append(
            {"form": form, "accessionNumber": accession, "primaryDocument": primary_doc, "filingDate": filing_date}
        )

    # filings.recent arrays are already newest-first per SEC's docs, but sort
    # defensively by filingDate descending in case that ever isn't true.
    filings.sort(key=lambda f: f.get("filingDate") or "", reverse=True)
    return filings[:2]


def _fetch_filing_text(cik_int: int, filing: dict[str, Any]) -> str:
    accession_nodashes = str(filing["accessionNumber"]).replace("-", "")
    url = ARCHIVE_URL_TMPL.format(
        cik_int=cik_int,
        accession_nodashes=accession_nodashes,
        primary_document=filing["primaryDocument"],
    )
    raw = get_text(url)
    if len(raw) > MAX_DOC_CHARS:
        raw = raw[:MAX_DOC_CHARS]
    return _html_to_text(raw)


def collect(
    companies: list[str] | None = None,
    phrases: list[str] | None = None,
) -> dict[str, Any]:
    """Collect phrase-acceleration metrics (latest vs. prior 10-Q/10-K) for `companies`.

    Returns:
      {
        "source": "edgar_language",
        "companies": {ticker: {phrase: {latest, prior, delta}}},
        "aggregate": {phrase: total_delta_across_companies},
        "aggregate_direction": {phrase: "accelerating"|"decelerating"|"flat"},
      }

    Never raises — NXT_OFFLINE=1 short-circuits immediately to the empty
    structure with no network calls; any per-company failure is caught and
    that company is simply skipped.
    """
    result: dict[str, Any] = {"source": "edgar_language", "companies": {}, "aggregate": {}}

    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("edgar_language: NXT_OFFLINE=1, skipping network calls")
        return result

    companies = companies or DEFAULT_COMPANIES
    phrases = phrases or DEFAULT_PHRASES

    try:
        ticker_to_cik = build_ticker_to_cik_map()
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_language: unexpected error building CIK map: %s", exc)
        ticker_to_cik = {}

    if not ticker_to_cik:
        logger.warning("edgar_language: no CIK map available, returning empty result")
        return result

    aggregate: dict[str, int] = {phrase: 0 for phrase in phrases}
    requests_made = 0

    for ticker in companies:
        if requests_made >= MAX_REQUESTS:
            logger.info("edgar_language: reached max request budget, stopping early")
            break

        cik = ticker_to_cik.get(ticker.upper())
        if cik is None:
            logger.warning("edgar_language: no CIK found for ticker %s", ticker)
            continue

        try:
            submissions = get_json(SUBMISSIONS_URL_TMPL.format(cik=cik))
            requests_made += 1
            time.sleep(REQUEST_PAUSE_SECONDS)

            filings = _recent_10q_or_10k(submissions)
            if len(filings) < 2:
                logger.info(
                    "edgar_language: fewer than 2 recent 10-Q/10-K filings for %s, skipping", ticker
                )
                continue

            latest_filing, prior_filing = filings[0], filings[1]

            if requests_made >= MAX_REQUESTS:
                logger.info("edgar_language: reached max request budget, stopping early")
                break
            latest_text = _fetch_filing_text(cik, latest_filing)
            requests_made += 1
            time.sleep(REQUEST_PAUSE_SECONDS)

            if requests_made >= MAX_REQUESTS:
                logger.info("edgar_language: reached max request budget, stopping early")
                break
            prior_text = _fetch_filing_text(cik, prior_filing)
            requests_made += 1
            time.sleep(REQUEST_PAUSE_SECONDS)

            # #5: focus on the MD&A narrative before counting (falls back to
            # the full text when the section markers aren't found).
            deltas = _phrase_deltas(
                _extract_mdna(latest_text), _extract_mdna(prior_text), phrases
            )
            result["companies"][ticker.upper()] = deltas
            for phrase, d in deltas.items():
                aggregate[phrase] = aggregate.get(phrase, 0) + d["delta"]

        except Exception as exc:  # noqa: BLE001
            logger.warning("edgar_language: unexpected error for %s: %s", ticker, exc)
            continue

    result["aggregate"] = aggregate
    result["aggregate_direction"] = {
        phrase: ("accelerating" if delta > 0 else "decelerating" if delta < 0 else "flat")
        for phrase, delta in aggregate.items()
    }

    logger.info(
        "edgar_language: collected data for %d/%d companies", len(result["companies"]), len(companies)
    )
    return result
