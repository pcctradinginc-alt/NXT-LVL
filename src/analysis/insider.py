"""Free parts of the deceleration filter (#10): insider selling + RS roll-over.

Both signals are computed only for the single top_pick at signal time (see
src/main.py), so they add at most a handful of extra requests per run, not
per candidate.

Fault-tolerant throughout: any network/parse failure degrades to None, never
raises — these are "nice to have" risk flags, not load-bearing pipeline
state.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Callable

from src.http_utils import get_json, get_text

logger = logging.getLogger(__name__)

SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL_TMPL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodashes}/{primary_document}"

FORM4_LOOKBACK_DAYS = 90
# Best-effort XML parsing of the actual Form 4 documents to net sold vs.
# bought shares is nice-to-have but adds a network request per document, so
# it's capped small — the recent_form4 COUNT (from the single submissions
# call) is the primary, cheap signal; the XML parse just tries to sharpen it
# into a directional hint when it cleanly can.
MAX_FORM4_XML_FETCHES = 3
REQUEST_PAUSE_SECONDS = 0.2

# Form 4 XML: each transaction lives inside a <nonDerivativeTransaction> or
# <derivativeTransaction> block, which contains both the transactionCode
# (S=open-market sale, P=open-market purchase, among others) and the
# transactionShares/value. We scope the code/shares search to within a
# single matched block so we don't accidentally pair a code from one
# transaction with the share count of another.
_TRANSACTION_BLOCK_RE = re.compile(
    r"<(non[Dd]erivative|[Dd]erivative)Transaction>(.*?)</\1Transaction>", re.DOTALL
)
_TRANSACTION_CODE_RE = re.compile(r"<transactionCode>\s*([A-Za-z])\s*</transactionCode>")
_TRANSACTION_SHARES_RE = re.compile(
    r"<transactionShares>.*?<value>\s*([\d.,]+)\s*</value>", re.DOTALL
)


def _recent_form4_filings(
    submissions: dict[str, Any], lookback_days: int = FORM4_LOOKBACK_DAYS
) -> list[dict[str, Any]]:
    """Extract Form 4 filings from the last `lookback_days` days, newest first."""
    try:
        recent = submissions.get("filings", {}).get("recent", {}) or {}
        forms = recent.get("form") or []
        accession_numbers = recent.get("accessionNumber") or []
        primary_docs = recent.get("primaryDocument") or []
        filing_dates = recent.get("filingDate") or []
    except AttributeError:
        return []

    cutoff = date.today() - timedelta(days=lookback_days)
    filings: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        filing_date_str = filing_dates[i] if i < len(filing_dates) else None
        try:
            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date() if filing_date_str else None
        except ValueError:
            filing_date = None
        if filing_date is not None and filing_date < cutoff:
            continue
        try:
            accession = accession_numbers[i]
            primary_doc = primary_docs[i]
        except IndexError:
            continue
        if not accession or not primary_doc:
            continue
        filings.append({"accessionNumber": accession, "primaryDocument": primary_doc, "filingDate": filing_date_str})

    filings.sort(key=lambda f: f.get("filingDate") or "", reverse=True)
    return filings


def _parse_form4_net_shares(xml_text: str) -> tuple[float, float]:
    """Best-effort regex parse of a Form 4 document's transaction codes.

    Returns (sold_shares, bought_shares) summed across "S" (open-market
    sale) and "P" (open-market purchase) coded transactions found in the
    document. Any transaction block that doesn't cleanly yield both a code
    and a share count inside itself is simply skipped — this is a
    best-effort heuristic, not a full Form 4 parser.
    """
    sold = 0.0
    bought = 0.0
    for match in _TRANSACTION_BLOCK_RE.finditer(xml_text):
        block = match.group(2)
        code_match = _TRANSACTION_CODE_RE.search(block)
        shares_match = _TRANSACTION_SHARES_RE.search(block)
        if not code_match or not shares_match:
            continue
        code = code_match.group(1).upper()
        try:
            shares = float(shares_match.group(1).replace(",", ""))
        except ValueError:
            continue
        if code == "S":
            sold += shares
        elif code == "P":
            bought += shares
    return sold, bought


def insider_selling_signal(
    ticker: str,
    cik_resolver: Callable[[str], int | None],
) -> dict[str, Any] | None:
    """Best-effort insider-selling signal from recent Form 4 filings.

    `cik_resolver(ticker) -> cik | None` decouples this from any one CIK
    lookup implementation (main.py passes a lookup into the shared,
    in-process-cached edgar_capex.build_ticker_to_cik_map() map; tests can
    inject a fake).

    Returns {"recent_form4": int, "net_sell_hint": "sell"|"buy"|"mixed"|None}
    — net_sell_hint is None whenever the best-effort XML parse couldn't
    confidently determine a direction (recent_form4 alone remains a valid,
    weaker activity proxy in that case). Returns None on any hard failure
    (offline, no CIK, network/parse error) or when no CIK could be resolved.
    """
    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("insider: NXT_OFFLINE=1, skipping network calls")
        return None

    try:
        cik = cik_resolver(ticker.upper())
    except Exception as exc:  # noqa: BLE001
        logger.warning("insider: cik_resolver failed for %s: %s", ticker, exc)
        return None

    if cik is None:
        logger.info("insider: no CIK resolved for %s", ticker)
        return None

    try:
        submissions = get_json(SUBMISSIONS_URL_TMPL.format(cik=int(cik)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("insider: failed to fetch submissions for %s (CIK %s): %s", ticker, cik, exc)
        return None

    try:
        filings = _recent_form4_filings(submissions)
    except Exception as exc:  # noqa: BLE001
        logger.warning("insider: failed to extract Form 4 filings for %s: %s", ticker, exc)
        return None

    recent_form4 = len(filings)
    net_sell_hint: str | None = None
    total_sold = 0.0
    total_bought = 0.0
    any_parsed = False

    for filing in filings[:MAX_FORM4_XML_FETCHES]:
        try:
            accession_nodashes = str(filing["accessionNumber"]).replace("-", "")
            url = ARCHIVE_URL_TMPL.format(
                cik_int=int(cik),
                accession_nodashes=accession_nodashes,
                primary_document=filing["primaryDocument"],
            )
            xml_text = get_text(url)
            time.sleep(REQUEST_PAUSE_SECONDS)
            sold, bought = _parse_form4_net_shares(xml_text)
            if sold or bought:
                any_parsed = True
            total_sold += sold
            total_bought += bought
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "insider: Form 4 doc parse failed for %s (%s): %s",
                ticker,
                filing.get("accessionNumber"),
                exc,
            )
            continue

    if any_parsed:
        if total_sold > total_bought:
            net_sell_hint = "sell"
        elif total_bought > total_sold:
            net_sell_hint = "buy"
        else:
            net_sell_hint = "mixed"

    return {"recent_form4": recent_form4, "net_sell_hint": net_sell_hint}


def relative_strength_decel(
    underlying_closes: list[float] | None,
    benchmark_closes: list[float] | None,
    short_window: int = 21,
    long_window: int = 63,
) -> dict[str, Any] | None:
    """Compare the candidate's relative strength over a short vs. a long window.

    RS = underlying_return - benchmark_return, each computed over the last
    `window` closes of its own series (both series independently, endpoint
    returns — not aligned by calendar date, matching how `closes` histories
    are consumed elsewhere in this pipeline). "Decelerating" means the
    short-window RS has rolled over below the longer-window RS — the stock
    was outperforming its benchmark over the quarter but has stopped doing
    so recently, a classic exhaustion pattern even amid good news.

    Returns {"rs_21": float, "rs_63": float, "decelerating": bool}, or None
    if either series doesn't have enough history for the long window.
    """
    if not underlying_closes or not benchmark_closes:
        return None
    if len(underlying_closes) < long_window + 1 or len(benchmark_closes) < long_window + 1:
        return None

    def _window_return(closes: list[float], window: int) -> float | None:
        segment = closes[-(window + 1):]
        if len(segment) < window + 1:
            return None
        try:
            first, last = float(segment[0]), float(segment[-1])
        except (TypeError, ValueError):
            return None
        if not first:
            return None
        return (last - first) / first * 100

    u_short = _window_return(underlying_closes, short_window)
    u_long = _window_return(underlying_closes, long_window)
    b_short = _window_return(benchmark_closes, short_window)
    b_long = _window_return(benchmark_closes, long_window)

    if None in (u_short, u_long, b_short, b_long):
        return None

    rs_21 = round(u_short - b_short, 2)
    rs_63 = round(u_long - b_long, 2)

    return {"rs_21": rs_21, "rs_63": rs_63, "decelerating": rs_21 < rs_63}
