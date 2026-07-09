"""Retroactive horizon evaluation (30/60/90/180 days) for all signals.

For every signal and every configured horizon that is "due" (signal age in
days >= horizon and that horizon has not been evaluated yet), this module
looks up the underlying's and the benchmark's price history via Tradier and
computes: absolute return, benchmark return, alpha, max drawdown, hit,
whether the underlying moved before the benchmark (early relative strength),
option profitability (Black-Scholes re-check), option liquidity bucket, and
whether the original thesis was confirmed.

Fully None-/error-tolerant: missing history for a given horizon simply skips
that horizon (it stays unevaluated and will be retried on a future run) —
never crashes the pipeline. Because it operates on the entire signals.json
on every run, signals can be evaluated retroactively at any time.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from src.analysis import options_math
from src import tracking

logger = logging.getLogger(__name__)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _closes_by_date(history: list[dict[str, Any]]) -> dict[date, float]:
    out: dict[date, float] = {}
    for bar in history or []:
        d = _parse_date(bar.get("date"))
        close = bar.get("close")
        if d is not None and close is not None:
            try:
                out[d] = float(close)
            except (TypeError, ValueError):
                continue
    return out


def _price_on_or_before(closes_by_date: dict[date, float], target: date) -> float | None:
    """Return the close on `target` or the closest earlier trading day."""
    candidates = [d for d in closes_by_date if d <= target]
    if not candidates:
        return None
    best = max(candidates)
    return closes_by_date[best]


def _max_drawdown(closes: list[float]) -> float:
    """Maximum peak-to-trough decline (as a positive percentage) over the series."""
    if not closes:
        return 0.0
    peak = closes[0]
    max_dd = 0.0
    for price in closes:
        if price > peak:
            peak = price
        if peak > 0:
            dd = (peak - price) / peak * 100
            if dd > max_dd:
                max_dd = dd
    return round(max_dd, 2)


def _liquidity_bucket(option_idea: dict[str, Any] | None) -> str | None:
    if not option_idea:
        return None
    oi = option_idea.get("oi") if option_idea.get("oi") is not None else option_idea.get("open_interest")
    spread_pct = option_idea.get("spread") if option_idea.get("spread") is not None else option_idea.get("spread_pct")

    if oi is None and spread_pct is None:
        return None

    oi = oi or 0
    spread_pct = spread_pct if spread_pct is not None else 1.0

    if oi >= 300 and spread_pct <= 0.05:
        return "gut"
    if oi >= 100 and spread_pct <= 0.10:
        return "mittel"
    return "schlecht"


def _evaluate_horizon(
    signal: dict[str, Any],
    horizon: int,
    tradier: Any,
    benchmark_symbol: str,
    current_emergence_scores: dict[str, float] | None,
) -> dict[str, Any] | None:
    """Compute one horizon's evaluation dict, or None if data is unavailable."""
    ticker = signal.get("ticker")
    sig_date = _parse_date(signal.get("date"))
    if not ticker or sig_date is None:
        return None

    target_date = sig_date + timedelta(days=horizon)
    history_start = sig_date.strftime("%Y-%m-%d")

    try:
        underlying_history = tradier.get_history(ticker, start=history_start)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reward.evaluator: get_history failed for %s: %s", ticker, exc)
        return None

    try:
        benchmark_history = tradier.get_history(benchmark_symbol, start=history_start)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reward.evaluator: get_history failed for benchmark %s: %s", benchmark_symbol, exc)
        benchmark_history = []

    underlying_closes = _closes_by_date(underlying_history)
    benchmark_closes = _closes_by_date(benchmark_history)

    price_at_signal = signal.get("price_at_signal") or signal.get("entry_underlying")
    if price_at_signal is None:
        price_at_signal = _price_on_or_before(underlying_closes, sig_date)
    if price_at_signal is None:
        return None
    price_at_signal = float(price_at_signal)

    price_at_horizon = _price_on_or_before(underlying_closes, target_date)
    if price_at_horizon is None:
        return None

    benchmark_at_signal = signal.get("benchmark_at_signal")
    if benchmark_at_signal is None:
        benchmark_at_signal = _price_on_or_before(benchmark_closes, sig_date)
    benchmark_at_horizon = _price_on_or_before(benchmark_closes, target_date)

    if price_at_signal == 0:
        return None
    absolute_return = round((price_at_horizon - price_at_signal) / price_at_signal * 100, 2)

    benchmark_return = None
    alpha = None
    if benchmark_at_signal and benchmark_at_horizon and benchmark_at_signal != 0:
        benchmark_return = round((benchmark_at_horizon - benchmark_at_signal) / benchmark_at_signal * 100, 2)
        alpha = round(absolute_return - benchmark_return, 2)
    # else: benchmark data unavailable -> benchmark_return/alpha stay None.
    # Fix 2/8b: do NOT silently fall back to alpha = absolute_return here —
    # that would quietly redefine "beat the benchmark" as "went up at all".
    # alpha is informational only; `hit` (below) has its own, option-aware
    # fallback for the no-benchmark case.

    # Max drawdown on underlying closes within [signal_date, target_date].
    window_closes = [
        price for d, price in sorted(underlying_closes.items()) if sig_date <= d <= target_date
    ]
    max_drawdown = _max_drawdown(window_closes)

    # preceded_move: did the underlying move ahead of the benchmark in the
    # first half of the window (early relative strength)?
    preceded_move = False
    half_days = horizon // 2
    if half_days > 0:
        half_date = sig_date + timedelta(days=half_days)
        price_half = _price_on_or_before(underlying_closes, half_date)
        bench_half = _price_on_or_before(benchmark_closes, half_date)
        if price_half is not None and price_at_signal != 0:
            underlying_half_ret = (price_half - price_at_signal) / price_at_signal * 100
            if bench_half is not None and benchmark_at_signal and benchmark_at_signal != 0:
                bench_half_ret = (bench_half - benchmark_at_signal) / benchmark_at_signal * 100
                preceded_move = bool(underlying_half_ret > bench_half_ret)
            else:
                preceded_move = bool(underlying_half_ret > 0)

    # Option profitability (Black-Scholes re-check), if an option idea exists.
    option_profitable = None
    option_idea = signal.get("option_idea")
    if option_idea:
        strike = option_idea.get("strike")
        entry_mid = option_idea.get("mid")
        expiration = option_idea.get("exp") or option_idea.get("expiration")
        entry_iv = option_idea.get("iv") or option_idea.get("entry_iv")
        exp_date = _parse_date(expiration) if isinstance(expiration, str) else None
        if strike is not None and entry_mid is not None and exp_date is not None:
            dte_remaining = (exp_date - target_date).days
            try:
                theoretical_value = options_math.estimate_call_value(
                    underlying_now=price_at_horizon,
                    strike=float(strike),
                    dte_days_remaining=max(0, dte_remaining),
                    entry_iv=entry_iv,
                )
                option_profitable = bool(theoretical_value > float(entry_mid))
            except Exception as exc:  # noqa: BLE001
                logger.warning("reward.evaluator: option re-valuation failed for %s: %s", ticker, exc)
                option_profitable = None

    option_liquidity = _liquidity_bucket(option_idea)

    # Fix 2/8b: `hit` reflects the actual instrument that was recommended.
    # If a call option was proposed and we could BS-re-check its
    # profitability, hit means "the option would still be worth more than
    # entry mid" — not merely "the stock went up" (a stock can rise while
    # the option, after time decay / delta, is worth less than paid).
    # Only when there is no option (or it isn't valuable-checkable) do we
    # fall back to a stock-based definition: beats the benchmark (alpha) if
    # we have one, else simply "went up".
    if option_idea and option_profitable is not None:
        hit = bool(option_profitable)
        hit_basis = "option"
    else:
        hit = bool(alpha > 0) if alpha is not None else bool(absolute_return > 0)
        hit_basis = "underlying"

    # thesis_confirmed: compare current emergence score for the signal's
    # theme (if any) against the score recorded at signal time; otherwise
    # heuristically fall back to alpha > 0.
    thesis_confirmed = None
    discovery = signal.get("discovery") or {}
    theme_id = discovery.get("theme_id")
    emergence_at_signal = signal.get("emergence_at_signal")
    if current_emergence_scores is not None and theme_id and emergence_at_signal is not None:
        current_score = current_emergence_scores.get(theme_id)
        if current_score is not None:
            thesis_confirmed = bool(current_score >= emergence_at_signal)
    if thesis_confirmed is None:
        thesis_confirmed = bool(alpha > 0) if alpha is not None else None

    return {
        "abs_return": absolute_return,
        "benchmark_return": benchmark_return,
        "alpha": alpha,
        "max_drawdown": max_drawdown,
        "hit": hit,
        "hit_basis": hit_basis,
        "preceded_move": preceded_move,
        "option_profitable": option_profitable,
        "option_liquidity": option_liquidity,
        "thesis_confirmed": thesis_confirmed,
        "evaluated_date": date.today().isoformat(),
    }


def evaluate_signals(
    signals: list[dict[str, Any]],
    tradier: Any,
    reward_cfg: dict[str, Any],
    current_emergence_scores: dict[str, float] | None = None,
    path: Any = tracking.DEFAULT_SIGNALS_PATH,
) -> list[dict[str, Any]]:
    """Fill due horizon_evals for every signal in-place and persist via tracking.

    Returns the updated signals list.
    """
    horizons = reward_cfg.get("horizons", [30, 60, 90, 180])
    benchmark_symbol = reward_cfg.get("benchmark_symbol", "SPY")
    today = date.today()

    for signal in signals:
        sig_date = _parse_date(signal.get("date"))
        if sig_date is None:
            continue
        age_days = (today - sig_date).days

        horizon_evals = signal.setdefault("horizon_evals", {})

        for horizon in horizons:
            key = str(horizon)
            if key in horizon_evals and horizon_evals[key]:
                continue  # already evaluated
            if age_days < horizon:
                continue  # not due yet

            try:
                evaluation = _evaluate_horizon(
                    signal, horizon, tradier, benchmark_symbol, current_emergence_scores
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "reward.evaluator: unexpected error evaluating %s horizon=%d: %s",
                    signal.get("ticker"),
                    horizon,
                    exc,
                )
                evaluation = None

            if evaluation is not None:
                horizon_evals[key] = evaluation
                logger.info(
                    "reward.evaluator: %s horizon=%d alpha=%s hit=%s",
                    signal.get("ticker"),
                    horizon,
                    evaluation.get("alpha"),
                    evaluation.get("hit"),
                )
            # else: leave unevaluated, retry on a future run

    tracking.save_signals(signals, path)
    return signals
