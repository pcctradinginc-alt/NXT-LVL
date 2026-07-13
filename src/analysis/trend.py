"""Pure, deterministic trend analysis functions for momentum, SMA, and regime filtering.

All functions are fault-tolerant: empty lists, None values, non-numeric entries, and
series shorter than required are handled gracefully, returning None or False as appropriate.
No exceptions are raised.

Functions work with daily closing prices in chronological order (oldest first, newest last).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _clean(closes: list[float]) -> list[float]:
    """Filter and validate a price series.

    Keeps only float-convertible positive values. Returns empty list if input is
    None, empty, or contains no valid values.

    Args:
        closes: List of closing prices (may contain None, strings, negatives, zeros).

    Returns:
        List of positive floats in original order, or empty list.
    """
    if closes is None:
        return []

    cleaned = []
    for val in closes:
        try:
            f = float(val)
            if f > 0:
                cleaned.append(f)
        except (TypeError, ValueError):
            continue

    return cleaned


def sma(closes: list[float], window: int) -> float | None:
    """Simple moving average of the last window cleaned closes.

    Args:
        closes: List of closing prices.
        window: Number of periods for the average.

    Returns:
        The SMA, or None if fewer than window valid values.
    """
    cleaned = _clean(closes)

    if window <= 0 or len(cleaned) < window:
        return None

    last_n = cleaned[-window:]
    return sum(last_n) / window


def momentum_12_1(closes: list[float]) -> float | None:
    """Classic 12-1 price momentum: 12-month return excluding the most recent month.

    Uses ~21 trading days per month: p_end = closes[-22] (close ~1 month ago) and
    p_start = closes[-253] (close ~12 months ago). Formula: (p_end / p_start - 1) * 100.

    Requires at least 253 cleaned closes, else None.

    Args:
        closes: List of closing prices.

    Returns:
        Momentum in percent, or None if insufficient data.
    """
    cleaned = _clean(closes)

    if len(cleaned) < 253:
        return None

    p_start = cleaned[-253]
    p_end = cleaned[-22]

    if p_start <= 0:
        return None

    return (p_end / p_start - 1) * 100


def above_sma(closes: list[float], window: int) -> bool | None:
    """Check if the last cleaned close is strictly above its SMA.

    Args:
        closes: List of closing prices.
        window: Window for the SMA.

    Returns:
        True if last close > SMA, False if last close <= SMA, None if SMA cannot be computed.
    """
    sma_val = sma(closes, window)

    if sma_val is None:
        return None

    cleaned = _clean(closes)

    if not cleaned:
        return None

    return cleaned[-1] > sma_val


def trend_ok(closes: list[float]) -> bool | None:
    """Trend filter: last close > 50-day SMA AND 50-day SMA > 200-day SMA.

    Requires at least 200 cleaned closes, else None.

    Args:
        closes: List of closing prices.

    Returns:
        True if both conditions met, False if either fails, None if insufficient data.
    """
    cleaned = _clean(closes)

    if len(cleaned) < 200:
        return None

    sma_50 = sma(cleaned, 50)
    sma_200 = sma(cleaned, 200)

    if sma_50 is None or sma_200 is None:
        return None

    last_close = cleaned[-1]

    return last_close > sma_50 and sma_50 > sma_200


def regime_risk_on(benchmark_closes: list[float]) -> bool | None:
    """Risk-on regime filter: benchmark last close > 200-day SMA.

    Typically used with SPY or similar broad market index.

    Args:
        benchmark_closes: List of benchmark closing prices.

    Returns:
        True if last close > 200-day SMA (risk-on), False otherwise, None if insufficient data.
    """
    cleaned = _clean(benchmark_closes)

    if len(cleaned) < 200:
        return None

    sma_200 = sma(cleaned, 200)

    if sma_200 is None:
        return None

    return cleaned[-1] > sma_200


def score_momentum_12_1(closes: list[float]) -> float:
    """Map momentum_12_1 to a 0–100 score for the scoring engine.

    Scoring buckets:
    - momentum is None (insufficient data) → 50.0 (neutral)
    - momentum <= -20 → 10
    - -20 < momentum <= 0 → 30
    - 0 < momentum <= 15 → 60
    - 15 < momentum <= 40 → 85
    - momentum > 40 → 70 (very hot momentum gets haircut for crash risk)

    Args:
        closes: List of closing prices.

    Returns:
        Float between 0 and 100.
    """
    m = momentum_12_1(closes)

    if m is None:
        return 50.0

    if m <= -20:
        return 10.0
    elif m <= 0:
        return 30.0
    elif m <= 15:
        return 60.0
    elif m <= 40:
        return 85.0
    else:  # m > 40
        return 70.0
