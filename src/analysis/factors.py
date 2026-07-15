"""Pure, deterministic candidate-factor scores for the walk-forward calibration.

CONCEPT_PROFIT.md Phase B/C: the first real walk-forward run found the buzz/
theme signals (divergence, breadth) anti-predictive and theme_momentum only
weakly positive. This module adds a set of empirically-documented, FREE,
historically-reconstructable factors so `src/backtest/optimize.py`'s fold
calibration can measure -- not assume -- whether they carry out-of-sample
edge:

  - reversal_1m_score: 1-month short-term price reversal (Jegadeesh 1990).
  - low_vol_score:     low-volatility anomaly (Ang et al. 2006).
  - high_52w_score:    proximity to the 52-week high (George & Hwang 2004).
  - rs_score:          relative strength vs. a benchmark (cross-sectional
                       momentum, a close cousin of momentum_12_1).
  - revenue_growth_score: fundamental revenue-growth momentum, fed by SEC
                       EDGAR XBRL data (see walkforward._revenue_yoy_asof).

Every function is fault-tolerant like src.analysis.trend: bad/short/None
input never raises, it degrades to a neutral 50.0 (or the documented
insufficient-data behavior). Each score is 0-100 where HIGHER means MORE
attractive per the function's documented direction (not necessarily "higher
raw factor value" -- e.g. low_vol_score is highest when volatility is LOW).

Price-based functions take `closes: list[float]` in chronological order
(oldest first, newest last), same convention as src.analysis.trend.
"""

from __future__ import annotations

import logging
import math
import statistics

logger = logging.getLogger(__name__)


def _clean(closes: list[float] | None) -> list[float]:
    """Filter a price series down to float-convertible positive values.

    Same approach as src.analysis.trend._clean: None/empty input, or input
    with no valid values, returns an empty list rather than raising.
    """
    if closes is None:
        return []

    cleaned: list[float] = []
    for val in closes:
        try:
            f = float(val)
            if f > 0:
                cleaned.append(f)
        except (TypeError, ValueError):
            continue

    return cleaned


def _pct_return(cleaned: list[float], window: int) -> float | None:
    """% return from `window` trading days ago to the last close.

    Needs at least `window + 1` cleaned closes (the close `window` days back
    plus the last close); returns None otherwise, or if the older close is
    not positive.
    """
    if window <= 0 or len(cleaned) < window + 1:
        return None
    p_start = cleaned[-(window + 1)]
    p_end = cleaned[-1]
    if p_start <= 0:
        return None
    return (p_end / p_start - 1.0) * 100.0


# ---------------------------------------------------------------------------
# 1-month short-term reversal (Jegadeesh 1990): recent losers tend to bounce,
# recent (1-month) winners tend to give some of it back. NOTE this is the
# opposite sign convention from momentum_12_1 -- it is specifically the
# SHORT-HORIZON reversal effect, measured over the same ~21-trading-day
# window that momentum_12_1 deliberately EXCLUDES from its 12-month window.
# ---------------------------------------------------------------------------


def reversal_1m_score(closes: list[float]) -> float:
    """Score the 1-month short-term reversal factor, 0-100 (higher = more
    attractive under the reversal hypothesis, i.e. a recent loser).

    r = % return over the trailing ~21 trading days (closes[-22] to
    closes[-1]). Mapping: r<=-15 -> 85, -15<r<=-5 -> 70, -5<r<=5 -> 55,
    5<r<=15 -> 40, r>15 -> 25. Fewer than 22 cleaned closes -> 50 (neutral).
    """
    cleaned = _clean(closes)
    if len(cleaned) < 22:
        return 50.0

    r = _pct_return(cleaned, 21)
    if r is None:
        return 50.0

    if r <= -15:
        return 85.0
    elif r <= -5:
        return 70.0
    elif r <= 5:
        return 55.0
    elif r <= 15:
        return 40.0
    else:
        return 25.0


# ---------------------------------------------------------------------------
# Low-volatility anomaly (Ang, Hodrick, Xing, Zhang 2006): lower-volatility
# names have historically delivered better risk-adjusted (and often better
# raw) forward returns than the "lottery ticket" high-vol names.
# ---------------------------------------------------------------------------


def low_vol_score(closes: list[float]) -> float:
    """Score the low-volatility anomaly, 0-100 (higher = lower realized vol
    = more attractive).

    rv = annualized realized volatility (population stdev of the trailing
    ~63 daily log returns, * sqrt(252)), clipped to [0.05, 3.0]. Mapping:
    rv<=0.25 -> 80, <=0.4 -> 65, <=0.6 -> 50, <=0.9 -> 35, else -> 20.
    Fewer than 64 cleaned closes (needed for 63 trailing returns) -> 50.
    """
    cleaned = _clean(closes)
    if len(cleaned) < 64:
        return 50.0

    tail = cleaned[-64:]
    log_returns: list[float] = []
    for prev, cur in zip(tail[:-1], tail[1:]):
        if prev <= 0 or cur <= 0:
            continue
        log_returns.append(math.log(cur / prev))

    if len(log_returns) < 2:
        return 50.0

    try:
        daily_vol = statistics.pstdev(log_returns)
    except statistics.StatisticsError:
        return 50.0

    rv = daily_vol * math.sqrt(252)
    rv = max(0.05, min(3.0, rv))

    if rv <= 0.25:
        return 80.0
    elif rv <= 0.4:
        return 65.0
    elif rv <= 0.6:
        return 50.0
    elif rv <= 0.9:
        return 35.0
    else:
        return 20.0


# ---------------------------------------------------------------------------
# Proximity to the 52-week high (George & Hwang 2004): stocks near their
# 52-week high tend to keep outperforming (an anchoring/underreaction
# effect), distinct from plain momentum.
# ---------------------------------------------------------------------------


def high_52w_score(closes: list[float]) -> float:
    """Score proximity to the 52-week high, 0-100 (higher = closer to the
    trailing high).

    ratio = last close / max(trailing ~252 closes). Mapping: ratio>=0.95 ->
    85, >=0.85 -> 70, >=0.70 -> 50, >=0.50 -> 35, else -> 20. Fewer than 60
    cleaned closes -> 50 (neutral; not enough history for a meaningful
    52-week proxy).
    """
    cleaned = _clean(closes)
    if len(cleaned) < 60:
        return 50.0

    window = cleaned[-252:] if len(cleaned) >= 252 else cleaned
    last = cleaned[-1]
    high = max(window)
    if high <= 0:
        return 50.0

    ratio = last / high
    if ratio >= 0.95:
        return 85.0
    elif ratio >= 0.85:
        return 70.0
    elif ratio >= 0.70:
        return 50.0
    elif ratio >= 0.50:
        return 35.0
    else:
        return 20.0


# ---------------------------------------------------------------------------
# Relative strength vs. a benchmark: cross-sectional momentum relative to
# the broad market, distinct from momentum_12_1's absolute 12-1 window.
# ---------------------------------------------------------------------------


def rs_score(closes: list[float], benchmark_closes: list[float]) -> float:
    """Score relative strength vs. a benchmark, 0-100 (higher = bigger
    outperformance over the trailing ~63 trading days).

    rs = (ticker's ~63-trading-day % return) - (benchmark's ~63-trading-day
    % return). Mapping: rs>15 -> 85, >5 -> 70, >=-5 -> 55, >=-15 -> 40, else
    -> 25. Either series having fewer than 64 cleaned closes -> 50 (neutral).
    """
    ticker_ret = _pct_return(_clean(closes), 63)
    bench_ret = _pct_return(_clean(benchmark_closes), 63)
    if ticker_ret is None or bench_ret is None:
        return 50.0

    rs = ticker_ret - bench_ret
    if rs > 15:
        return 85.0
    elif rs > 5:
        return 70.0
    elif rs >= -5:
        return 55.0
    elif rs >= -15:
        return 40.0
    else:
        return 25.0


# ---------------------------------------------------------------------------
# Fundamental revenue-growth momentum (fed by SEC EDGAR XBRL company-concept
# data, see src/backtest/walkforward.py's _revenue_yoy_asof).
# ---------------------------------------------------------------------------


def revenue_growth_score(yoy_growth_pct: float | None) -> float:
    """Score YoY revenue growth, 0-100 (higher = faster growth).

    Mapping: g>=40 -> 85, >=20 -> 70, >=5 -> 55, >=0 -> 45, g<0 -> 25.
    None (no filing data available as-of the measurement date) -> 50
    (neutral, not a penalty for missing fundamentals data).
    """
    if yoy_growth_pct is None:
        return 50.0
    try:
        g = float(yoy_growth_pct)
    except (TypeError, ValueError):
        return 50.0

    if g >= 40:
        return 85.0
    elif g >= 20:
        return 70.0
    elif g >= 5:
        return 55.0
    elif g >= 0:
        return 45.0
    else:
        return 25.0
