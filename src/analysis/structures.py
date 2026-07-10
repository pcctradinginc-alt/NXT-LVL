"""Deterministic, free "cost & structure" half of an options EV check.

Given a selected long call (from `tradier.select_option`), a candidate short
leg (from `tradier.select_short_leg`), and the underlying's realized
volatility, decide whether a plain long call, a call spread (to cut IV/theta
risk when the option looks expensive), or plain stock is the more sensible
structure to present in the signal email. This is a rule-based, no-LLM layer
— purely arithmetic on data already fetched from Tradier — so it never adds
network calls or nondeterminism of its own.

Fault-tolerant throughout: any missing input (no IV, no history, no short
leg) degrades to a `long_call` recommendation rather than raising, so the
existing long-call-only pipeline behavior is preserved when nothing special
applies.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from src.analysis.options_math import bs_call_theta

logger = logging.getLogger(__name__)


def realized_vol(closes: list[float], window: int = 63) -> float | None:
    """Annualized realized volatility from daily log returns.

    Uses the last `window` closes (default ~63 trading days / one quarter),
    population stdev of daily log returns * sqrt(252). Returns None if fewer
    than ~20 usable log returns are available (too little history to trust).
    Result is clipped to [0.05, 3.0] to avoid degenerate extremes feeding
    downstream comparisons.
    """
    if not closes or len(closes) < 2:
        return None
    try:
        tail = [float(c) for c in closes[-window:] if c is not None and float(c) > 0]
    except (TypeError, ValueError):
        return None

    log_returns: list[float] = []
    for prev, curr in zip(tail, tail[1:]):
        if prev <= 0 or curr <= 0:
            continue
        try:
            log_returns.append(math.log(curr / prev))
        except (ValueError, ZeroDivisionError):
            continue

    if len(log_returns) < 20:
        return None

    try:
        daily_stdev = statistics.pstdev(log_returns)
    except statistics.StatisticsError:
        return None

    annualized = daily_stdev * math.sqrt(252)
    return max(0.05, min(3.0, annualized))


def iv_expensive(
    option_iv: float | None, realized_vol_val: float | None, ratio_threshold: float
) -> bool:
    """True iff both IV and realized vol are known and IV exceeds the threshold ratio."""
    if option_iv is None or realized_vol_val is None:
        return False
    if realized_vol_val <= 0:
        return False
    return option_iv > ratio_threshold * realized_vol_val


def long_call_metrics(
    underlying: float,
    strike: float,
    mid: float,
    delta: float | None,
    iv: float | None,
    dte_days: int | None,
    r: float = 0.04,
) -> dict[str, Any]:
    """Cost/break-even/theta metrics for a plain long call, one contract (100x multiplier)."""
    break_even = strike + mid
    break_even_move_pct = (break_even / underlying - 1) * 100 if underlying else None
    max_loss = round(mid * 100, 2)

    theta_per_day = None
    if dte_days is not None and dte_days > 0:
        sigma = iv if iv else 0.5
        try:
            theta_year = bs_call_theta(underlying, strike, dte_days / 365.0, r, sigma)
            theta_per_day = theta_year / 365.0
        except Exception as exc:  # noqa: BLE001
            logger.debug("long_call_metrics: theta calc failed: %s", exc)
            theta_per_day = None

    return {
        "strike": strike,
        "mid": mid,
        "delta": delta,
        "iv": iv,
        "dte": dte_days,
        "break_even": break_even,
        "break_even_move_pct": break_even_move_pct,
        "max_loss": max_loss,
        "theta_per_day": theta_per_day,
    }


def call_spread_metrics(
    underlying: float,
    long_strike: float,
    long_mid: float,
    short_strike: float,
    short_mid: float | None,
    dte_days: int | None,
) -> dict[str, Any] | None:
    """Cost/break-even/max-profit metrics for a debit call spread, one contract.

    Returns None when the short leg is missing or does not form a valid
    (strictly higher-strike, positive-debit) spread.
    """
    if short_mid is None:
        return None
    if short_strike <= long_strike:
        return None

    net_debit = long_mid - short_mid
    if net_debit <= 0:
        return None

    width = short_strike - long_strike
    max_profit = round((width - net_debit) * 100, 2)
    max_loss = round(net_debit * 100, 2)
    break_even = long_strike + net_debit
    break_even_move_pct = (break_even / underlying - 1) * 100 if underlying else None

    return {
        "long_strike": long_strike,
        "long_mid": long_mid,
        "short_strike": short_strike,
        "short_mid": short_mid,
        "dte": dte_days,
        "net_debit": round(net_debit, 4),
        "width": width,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "break_even": break_even,
        "break_even_move_pct": break_even_move_pct,
    }


def choose_structure(
    underlying: float,
    long_call: dict[str, Any],
    short_leg: dict[str, Any] | None,
    realized_vol_val: float | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Pick long_call / call_spread / stock based on whether IV looks expensive.

    `long_call` is the selected option dict (strike/mid/delta/iv/dte, as
    returned by `TradierClient.select_option`). `short_leg` is a dict with at
    least {strike, mid} (as returned by `TradierClient.select_short_leg`), or
    None if no suitable short leg was found. `cfg` is the `options:` config
    block (uses `max_iv_realized_ratio`).
    """
    option_iv = long_call.get("iv")
    ratio_threshold = float(cfg.get("max_iv_realized_ratio", 1.6))
    expensive = iv_expensive(option_iv, realized_vol_val, ratio_threshold)

    result: dict[str, Any] = {
        "iv_expensive": expensive,
        "realized_vol": realized_vol_val,
    }

    if not expensive:
        result["structure"] = "long_call"
        result["reason"] = "IV im Rahmen (nicht teuer ggü. realisierter Vola)"
        result["metrics"] = long_call_metrics(
            underlying,
            long_call.get("strike"),
            long_call.get("mid"),
            long_call.get("delta"),
            long_call.get("iv"),
            long_call.get("dte"),
        )
        return result

    spread_metrics = None
    if short_leg is not None:
        spread_metrics = call_spread_metrics(
            underlying,
            long_call.get("strike"),
            long_call.get("mid"),
            short_leg.get("strike"),
            short_leg.get("mid"),
            long_call.get("dte"),
        )

    if spread_metrics is not None:
        result["structure"] = "call_spread"
        result["reason"] = "IV teuer → Call-Spread reduziert Zeitwert-/IV-Risiko"
        result["metrics"] = spread_metrics
        result["short_leg"] = short_leg
        return result

    result["structure"] = "stock"
    result["reason"] = "IV teuer und kein sinnvoller Spread → Aktie statt Call erwägen"
    result["metrics"] = {"underlying": underlying}
    return result
