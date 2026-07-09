"""Mechanical price backtest — validates option/divergence building blocks only.

Honest scope: this backtest tests two mechanical hypotheses the scoring
system encodes, using REAL historical equity prices:

  1. Divergence rule: do stocks that were flat/slightly-up over the trailing
     3 months produce better forward CALL-option P/L than ones that already
     ran up a lot? (scoring.score_divergence buckets the trailing move; we
     check whether the bucket actually predicts forward option return.)
  2. Delta selection: is a slightly-in-the-money call (delta ~0.6) a sound
     structural choice, measured via Black-Scholes re-valuation over a fixed
     holding horizon on real underlying price paths?

It does NOT backtest the collector-driven signal logic (breadth, momentum,
stage_fit, emergence, source confirmation) at all — those depend on
GitHub/EDGAR/HN/arXiv data that is point-in-time and was never archived
historically, so no retroactive replay of "what would the scanner have said
on date X" is possible. That question can only be answered going forward, by
archiving digests now (see src/main.py's append_digest_history) and later
running src/backtest/calibrate.py once enough history has accumulated.

Every option value in this backtest is a theoretical Black-Scholes
re-valuation using realized volatility as a stand-in for market-quoted IV,
priced on the true historical underlying close price path, using
src.analysis.options_math (the exact same pricing code the reward evaluator
uses live). No real historical options-chain data is used (Tradier does not
provide historical options quotes), so the option P/L figures are a
best-effort structural approximation, not what an actual fill would have
been.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import statistics
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.analysis import scoring
from src.analysis.options_math import bs_call_price, solve_strike_for_delta
from src.config import DATA_DIR, load_settings
from src.options.tradier import TradierClient

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
TRAILING_WINDOW_DAYS = 63  # ~3 trading months
REPORT_PATH = DATA_DIR / "backtest_report.json"

HEADER_LINE = (
    "Mechanical price backtest — validates the option/divergence building "
    "blocks on historical prices only; it does NOT backtest the collector "
    "signals (point-in-time, never archived)."
)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _build_price_map(bars: list[dict[str, Any]]) -> dict[date, float]:
    """Turn a list of {date, close, ...} bars into a sorted date->close map."""
    out: dict[date, float] = {}
    for bar in bars:
        try:
            d = _parse_date(str(bar["date"])[:10])
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            continue
        if close > 0:
            out[d] = close
    return out


def _sorted_dates(price_map: dict[date, float]) -> list[date]:
    return sorted(price_map.keys())


def _nearest_trading_day(
    sorted_dates: list[date], target: date, *, allow_after: bool = True
) -> date | None:
    """Snap `target` to the nearest available trading day in `sorted_dates`.

    If `allow_after` is False, only dates <= target are considered (used for
    exit dates, so we never look into the future relative to the horizon).
    """
    if not sorted_dates:
        return None
    if allow_after:
        best = min(sorted_dates, key=lambda d: abs((d - target).days))
        return best
    candidates = [d for d in sorted_dates if d <= target]
    if not candidates:
        return None
    return max(candidates)


def _trailing_perf_pct(sorted_dates: list[date], price_map: dict[date, float], entry_date: date) -> float | None:
    """3-month trailing performance (%) ending at (closest bar <=) entry_date."""
    idx = None
    for i, d in enumerate(sorted_dates):
        if d <= entry_date:
            idx = i
        else:
            break
    if idx is None or idx < TRAILING_WINDOW_DAYS:
        return None
    start_close = price_map[sorted_dates[idx - TRAILING_WINDOW_DAYS]]
    end_close = price_map[sorted_dates[idx]]
    if start_close <= 0:
        return None
    return (end_close - start_close) / start_close * 100.0


def _realized_vol(sorted_dates: list[date], price_map: dict[date, float], entry_date: date) -> float | None:
    """Annualized realized volatility from trailing daily log returns."""
    idx = None
    for i, d in enumerate(sorted_dates):
        if d <= entry_date:
            idx = i
        else:
            break
    if idx is None or idx < TRAILING_WINDOW_DAYS:
        return None
    window = [price_map[sorted_dates[i]] for i in range(idx - TRAILING_WINDOW_DAYS, idx + 1)]
    log_returns = []
    for prev, cur in zip(window[:-1], window[1:]):
        if prev <= 0 or cur <= 0:
            continue
        log_returns.append(math.log(cur / prev))
    if len(log_returns) < 2:
        return None
    try:
        daily_vol = statistics.pstdev(log_returns)
    except statistics.StatisticsError:
        return None
    annualized = daily_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    return max(0.15, min(1.5, annualized))


def _entry_dates(
    sorted_dates: list[date], window_start: date, window_end: date, cadence_days: int
) -> list[date]:
    """Walk from window_start to window_end in cadence_days steps, snapping
    each step to the nearest available trading day."""
    if not sorted_dates:
        return []
    out: list[date] = []
    cursor = window_start
    while cursor <= window_end:
        snapped = _nearest_trading_day(sorted_dates, cursor)
        if snapped is not None and window_start <= snapped <= window_end:
            if not out or out[-1] != snapped:
                out.append(snapped)
        cursor = cursor + timedelta(days=cadence_days)
    return out


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(samples)
    if n == 0:
        return {"n": 0, "hit_rate": None, "avg_option_return": None, "avg_underlying_return": None}
    hits = sum(1 for s in samples if s["hit"])
    return {
        "n": n,
        "hit_rate": round(hits / n, 4),
        "avg_option_return": round(sum(s["option_return"] for s in samples) / n, 4),
        "avg_underlying_return": round(sum(s["underlying_return"] for s in samples) / n, 4),
    }


def run_backtest(
    tradier: TradierClient | None,
    tickers: list[str],
    *,
    years: int = 3,
    horizon_days: int = 90,
    entry_dte: int = 120,
    target_delta: float = 0.60,
    cadence_days: int = 30,
    r: float = 0.04,
    price_series: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Run the mechanical price backtest over `tickers`.

    `price_series`, when provided, is {ticker: {date_str: close}} and is used
    instead of calling `tradier` (offline/mock mode). Otherwise `tradier` must
    be a TradierClient and is queried for full daily history.
    """
    today = date.today()
    buffer_days = TRAILING_WINDOW_DAYS + entry_dte + 30  # extra history for trailing/entry math
    window_start = today - timedelta(days=365 * years)
    fetch_start = window_start - timedelta(days=buffer_days)
    entry_window_end = today - timedelta(days=horizon_days)

    samples: list[dict[str, Any]] = []
    tickers_used: list[str] = []
    tickers_skipped: list[str] = []

    for ticker in tickers:
        price_map: dict[date, float] = {}
        if price_series is not None:
            raw = price_series.get(ticker, {})
            for date_str, close in raw.items():
                try:
                    d = _parse_date(str(date_str)[:10])
                    price_map[d] = float(close)
                except (ValueError, TypeError):
                    continue
        elif tradier is not None:
            bars = tradier.get_history(ticker, start=fetch_start.strftime("%Y-%m-%d"))
            price_map = _build_price_map(bars)
        else:
            logger.warning("run_backtest: no tradier client and no price_series for %s, skipping", ticker)
            tickers_skipped.append(ticker)
            continue

        sorted_dates = _sorted_dates(price_map)
        if len(sorted_dates) < TRAILING_WINDOW_DAYS + 10:
            logger.info("run_backtest: %s has too few bars (%d), skipping", ticker, len(sorted_dates))
            tickers_skipped.append(ticker)
            continue

        entry_dates = _entry_dates(sorted_dates, window_start, entry_window_end, cadence_days)
        if not entry_dates:
            tickers_skipped.append(ticker)
            continue

        used_this_ticker = False
        for entry_date in entry_dates:
            perf_3m = _trailing_perf_pct(sorted_dates, price_map, entry_date)
            if perf_3m is None:
                continue
            bucket = scoring.score_divergence(perf_3m)

            iv = _realized_vol(sorted_dates, price_map, entry_date)
            if iv is None:
                continue

            s_entry = price_map.get(entry_date)
            if s_entry is None:
                # entry_date came from snapping, should be present, but guard anyway.
                continue

            strike = solve_strike_for_delta(s_entry, target_delta, entry_dte / 365.0, r, iv)
            entry_value = bs_call_price(s_entry, strike, entry_dte / 365.0, r, iv)
            if entry_value <= 0:
                continue

            raw_exit_target = entry_date + timedelta(days=horizon_days)
            exit_date = _nearest_trading_day(sorted_dates, raw_exit_target, allow_after=False)
            if exit_date is None:
                continue
            s_exit = price_map.get(exit_date)
            if s_exit is None:
                continue

            remaining_dte = max(0, entry_dte - horizon_days)
            exit_value = bs_call_price(s_exit, strike, remaining_dte / 365.0, r, iv)

            option_return = exit_value / entry_value - 1.0
            underlying_return = s_exit / s_entry - 1.0
            hit = exit_value > entry_value

            samples.append(
                {
                    "ticker": ticker,
                    "entry_date": entry_date.isoformat(),
                    "perf_3m": round(perf_3m, 2),
                    "divergence_bucket": bucket,
                    "iv": round(iv, 4),
                    "option_return": option_return,
                    "underlying_return": underlying_return,
                    "hit": hit,
                }
            )
            used_this_ticker = True

        if used_this_ticker:
            tickers_used.append(ticker)
        else:
            tickers_skipped.append(ticker)

    by_bucket: dict[str, Any] = {}
    for bucket_value in sorted({s["divergence_bucket"] for s in samples}, reverse=True):
        bucket_samples = [s for s in samples if s["divergence_bucket"] == bucket_value]
        by_bucket[str(bucket_value)] = _aggregate(bucket_samples)

    report = {
        "header": HEADER_LINE,
        "params": {
            "years": years,
            "horizon_days": horizon_days,
            "entry_dte": entry_dte,
            "target_delta": target_delta,
            "cadence_days": cadence_days,
            "r": r,
            "tickers_requested": tickers,
            "tickers_used": tickers_used,
            "tickers_skipped": tickers_skipped,
            "run_date": today.isoformat(),
        },
        "overall": _aggregate(samples),
        "by_divergence_bucket": by_bucket,
        "n_samples": len(samples),
    }
    return report


# ---------------------------------------------------------------------------
# Mock price generation (deterministic GBM, no network)
# ---------------------------------------------------------------------------

MOCK_TICKERS: dict[str, dict[str, float]] = {
    "MOCKA": {"drift": 0.15, "vol": 0.35},   # steady uptrend, moderate vol
    "MOCKB": {"drift": 0.05, "vol": 0.55},   # flat-ish, choppy
    "MOCKC": {"drift": 0.35, "vol": 0.45},   # already ran hard
}


def _generate_mock_price_series(years: int = 4) -> dict[str, dict[str, float]]:
    """Deterministic synthetic GBM daily price series, seeded for reproducibility."""
    rng = random.Random(42)
    n_days = int(365.25 * years)
    start_date = date.today() - timedelta(days=n_days)
    dt = 1.0 / TRADING_DAYS_PER_YEAR

    series: dict[str, dict[str, float]] = {}
    for ticker, params in MOCK_TICKERS.items():
        mu = params["drift"]
        sigma = params["vol"]
        price = 100.0
        bars: dict[str, float] = {}
        current = start_date
        for _ in range(n_days):
            # Only step price on "trading days" (skip weekends) for realism,
            # but this is a mock series so exact calendar fidelity doesn't matter.
            if current.weekday() < 5:
                z = rng.gauss(0.0, 1.0)
                price *= math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
                bars[current.isoformat()] = round(price, 4)
            current += timedelta(days=1)
        series[ticker] = bars
    return series


def _print_summary(report: dict[str, Any]) -> None:
    print(report["header"])
    print()
    params = report["params"]
    print(
        f"years={params['years']} horizon_days={params['horizon_days']} "
        f"entry_dte={params['entry_dte']} target_delta={params['target_delta']} "
        f"cadence_days={params['cadence_days']}"
    )
    print(f"tickers used: {params['tickers_used']}")
    if params["tickers_skipped"]:
        print(f"tickers skipped (insufficient data): {params['tickers_skipped']}")
    print()

    overall = report["overall"]
    print(f"n_samples = {report['n_samples']}")
    if overall["n"]:
        print(
            f"OVERALL: n={overall['n']} hit_rate={overall['hit_rate']:.2%} "
            f"avg_option_return={overall['avg_option_return']:.2%} "
            f"avg_underlying_return={overall['avg_underlying_return']:.2%}"
        )
    else:
        print("OVERALL: no samples produced.")
    print()

    print("By divergence bucket (scoring.score_divergence value):")
    for bucket, stats in report["by_divergence_bucket"].items():
        if stats["n"]:
            print(
                f"  bucket={bucket:>4}: n={stats['n']:>4} hit_rate={stats['hit_rate']:.2%} "
                f"avg_option_return={stats['avg_option_return']:.2%} "
                f"avg_underlying_return={stats['avg_underlying_return']:.2%}"
            )
        else:
            print(f"  bucket={bucket:>4}: no samples")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NXT LVL — mechanical price backtest (option/divergence building blocks only)"
    )
    parser.add_argument("--years", type=int, default=3, help="Lookback window in years (default 3)")
    parser.add_argument("--horizon", type=int, default=90, dest="horizon_days", help="Holding horizon in days (default 90)")
    parser.add_argument("--entry-dte", type=int, default=120, dest="entry_dte", help="Option DTE at entry (default 120)")
    parser.add_argument("--delta", type=float, default=0.60, dest="target_delta", help="Target call delta (default 0.60)")
    parser.add_argument("--cadence", type=int, default=30, dest="cadence_days", help="Days between entry samples (default 30)")
    parser.add_argument("--mock", action="store_true", help="Use deterministic synthetic prices, no network required")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    args = parse_args(argv)

    if args.mock:
        price_series = _generate_mock_price_series(years=max(4, args.years + 1))
        tickers = list(MOCK_TICKERS.keys())
        report = run_backtest(
            None,
            tickers,
            years=args.years,
            horizon_days=args.horizon_days,
            entry_dte=args.entry_dte,
            target_delta=args.target_delta,
            cadence_days=args.cadence_days,
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
        tickers = sorted(settings.watchlist_tickers() | {
            str(t).upper() for theme in settings.themes for t in (theme.get("tickers") or [])
        })
        if not tickers:
            print("No tickers found in config.yaml (stages[].tickers / themes[].tickers).")
            return 1
        report = run_backtest(
            tradier,
            tickers,
            years=args.years,
            horizon_days=args.horizon_days,
            entry_dte=args.entry_dte,
            target_delta=args.target_delta,
            cadence_days=args.cadence_days,
        )

    _print_summary(report)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import json

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"\nReport written to {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
