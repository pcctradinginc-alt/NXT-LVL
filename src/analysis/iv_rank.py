"""Forward-accumulating IV rank (#6).

Tradier exposes no historical implied-volatility series, so we accumulate the
IV of every signalled option ourselves (one JSONL line per signal) and rank a
new pick's IV against that ticker's OWN past observations. This sharpens the
realized-vol heuristic in `structures.iv_expensive` with a real, if slowly
growing, "is this option historically expensive for this name?" gauge.

All functions are pure/offline-testable and fully fault-tolerant: a missing or
malformed history file never raises, it just yields an empty history / a None
percentile (a percentile is only reported once enough prior observations exist
for the ticker).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def append_iv(path: Path | str, ticker: str, iv: float | None, run_date: str | None = None) -> None:
    """Append one `{date, ticker, iv}` record to the IV-history JSONL file.

    Skips silently when `iv` is missing/non-positive. Never raises — any I/O
    failure is logged and swallowed (this is an analytics side-channel, not
    load-bearing pipeline state).
    """
    if iv is None:
        return
    try:
        iv_val = float(iv)
    except (TypeError, ValueError):
        return
    if iv_val <= 0:
        return

    record = {"date": run_date or date.today().isoformat(), "ticker": str(ticker).upper(), "iv": iv_val}
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("iv_rank.append_iv failed for %s: %s", ticker, exc)


def load_iv_history(path: Path | str) -> list[dict[str, Any]]:
    """Load the IV-history JSONL, tolerating (and skipping) malformed lines."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except Exception as exc:  # noqa: BLE001
        logger.warning("iv_rank.load_iv_history failed: %s", exc)
        return []
    return records


def iv_percentile(
    ticker: str,
    current_iv: float | None,
    history: list[dict[str, Any]],
    min_samples: int = 8,
) -> float | None:
    """Percentile rank (0-100) of `current_iv` among this ticker's prior IVs.

    Uses the average-rank / mid-point convention for ties:
        percentile = (n_below + 0.5 * n_equal) / n * 100
    Returns None when `current_iv` is missing or there are fewer than
    `min_samples` prior observations for this ticker (not enough history to
    rank meaningfully).
    """
    if current_iv is None:
        return None
    try:
        cur = float(current_iv)
    except (TypeError, ValueError):
        return None

    ticker_up = str(ticker).upper()
    prior: list[float] = []
    for rec in history:
        if str(rec.get("ticker", "")).upper() != ticker_up:
            continue
        val = rec.get("iv")
        try:
            prior.append(float(val))
        except (TypeError, ValueError):
            continue

    n = len(prior)
    if n < min_samples:
        return None

    n_below = sum(1 for v in prior if v < cur)
    n_equal = sum(1 for v in prior if v == cur)
    percentile = (n_below + 0.5 * n_equal) / n * 100
    return round(percentile, 1)
