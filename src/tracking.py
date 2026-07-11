"""Signal track record: persists signals.json and computes rolling hit rate.

Signal object shape:
{
  "id": str,
  "date": "YYYY-MM-DD",
  "ticker": str,
  "occ_symbol": str | null,
  "strike": float | null,
  "expiration": "YYYY-MM-DD" | null,
  "entry_option_mid": float | null,
  "entry_underlying": float | null,
  "score": float,
  "thesis": str,
  "status": "open" | "closed",
  "checkpoints": [{"date": "YYYY-MM-DD", "option_mid": float | null, "dte": int | null}],
  "result": null | {"closed_date": str, "exit_mid": float | null, "pnl_pct": float | null, "hit": bool}
}
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIGNALS_PATH = PROJECT_ROOT / "data" / "signals.json"

# Approximate trading-day -> calendar-day conversion.
# US markets trade ~5/7 days minus holidays; 0.69 is a documented rule-of-thumb
# multiplier (5/7 ≈ 0.714, minus ~holiday slack ≈ 0.69) so that
# calendar_days * 0.69 ≈ elapsed trading days.
TRADING_DAY_CALENDAR_FACTOR = 0.69


def _today_str() -> str:
    return date.today().isoformat()


def load_signals(path: Path | str = DEFAULT_SIGNALS_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        logger.warning("tracking: signals.json did not contain a list, ignoring")
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("tracking: failed to load signals.json: %s", exc)
        return []


def save_signals(signals: list[dict[str, Any]], path: Path | str = DEFAULT_SIGNALS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(signals, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def in_cooldown(ticker: str, days: int, signals: list[dict[str, Any]] | None = None,
                 path: Path | str = DEFAULT_SIGNALS_PATH) -> bool:
    """True if `ticker` had a signal created within the last `days` days."""
    signals = signals if signals is not None else load_signals(path)
    cutoff = date.today() - timedelta(days=days)
    for sig in signals:
        if sig.get("ticker") != ticker:
            continue
        try:
            sig_date = datetime.strptime(sig.get("date", ""), "%Y-%m-%d").date()
        except ValueError:
            continue
        if sig_date >= cutoff:
            return True
    return False


def add_signal(
    ticker: str,
    score: float,
    thesis: str,
    occ_symbol: str | None = None,
    strike: float | None = None,
    expiration: str | None = None,
    entry_option_mid: float | None = None,
    entry_underlying: float | None = None,
    signals: list[dict[str, Any]] | None = None,
    path: Path | str = DEFAULT_SIGNALS_PATH,
    data_sources: list[str] | None = None,
    reasoning: str = "",
    recommended_horizon_days: int | None = None,
    price_at_signal: float | None = None,
    benchmark_symbol: str | None = None,
    benchmark_at_signal: float | None = None,
    option_idea: dict[str, Any] | None = None,
    data_quality_score: float | None = None,
    source_attribution: list[str] | None = None,
    feature_attribution: dict[str, float] | None = None,
    discovery: dict[str, Any] | None = None,
    emergence_at_signal: float | None = None,
    structure: dict[str, Any] | None = None,
    realized_vol: float | None = None,
    earnings_date: str | None = None,
    invalidation: dict[str, Any] | None = None,
    insider: dict[str, Any] | None = None,
    rs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and persist a new open signal. Returns the created signal dict.

    All Reward-Engine fields are optional and default to values that keep
    existing callers/tests (which only pass the original parameters) working
    unchanged.
    """
    signals = signals if signals is not None else load_signals(path)

    signal = {
        "id": uuid.uuid4().hex[:12],
        "date": _today_str(),
        "ticker": ticker,
        "occ_symbol": occ_symbol,
        "strike": strike,
        "expiration": expiration,
        "entry_option_mid": entry_option_mid,
        "entry_underlying": entry_underlying,
        "score": score,
        "thesis": thesis,
        "status": "open",
        "checkpoints": [],
        "result": None,
        # --- Reward Engine extensions (all optional, see CONCEPT_EMERGENCE.md B.1) ---
        "data_sources": data_sources or [],
        "reasoning": reasoning,
        "recommended_horizon_days": recommended_horizon_days,
        "price_at_signal": price_at_signal,
        "benchmark_symbol": benchmark_symbol,
        "benchmark_at_signal": benchmark_at_signal,
        "option_idea": option_idea,
        "data_quality_score": data_quality_score,
        "source_attribution": source_attribution or [],
        "feature_attribution": feature_attribution or {},
        "discovery": discovery or {},
        "emergence_at_signal": emergence_at_signal,
        "horizon_evals": {},
        # --- Options structure-selection layer (see src/analysis/structures.py) ---
        "structure": structure,
        "realized_vol": realized_vol,
        "earnings_date": earnings_date,
        # --- Invalidation levels (#14) ---
        "invalidation": invalidation,
        # --- Deceleration filter, free parts (#10) ---
        "insider": insider,
        "rs": rs,
    }
    signals.append(signal)
    save_signals(signals, path)
    logger.info("tracking: added signal %s for %s (score=%s)", signal["id"], ticker, score)
    return signal


def _dte(expiration: str | None) -> int | None:
    if not expiration:
        return None
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (exp_date - date.today()).days


def evaluate_open_signals(
    tradier: Any,
    close_after_trading_days: int = 60,
    close_at_dte: int = 40,
    path: Path | str = DEFAULT_SIGNALS_PATH,
) -> list[dict[str, Any]]:
    """Re-evaluate all open signals against current Tradier quotes.

    For each open signal:
    - fetch current option mid via tradier.get_quote(occ_symbol)
    - append a checkpoint
    - close it if elapsed calendar days * TRADING_DAY_CALENDAR_FACTOR >=
      close_after_trading_days, OR current DTE < close_at_dte, OR the option
      is no longer quotable (missing bid/ask).
    Hit = exit_mid > entry_mid.

    Returns the updated full signals list (also persisted to disk).
    """
    signals = load_signals(path)
    today = date.today()

    for sig in signals:
        if sig.get("status") != "open":
            continue

        occ_symbol = sig.get("occ_symbol")
        entry_mid = sig.get("entry_option_mid")

        current_mid = None
        current_dte = _dte(sig.get("expiration"))

        if occ_symbol:
            try:
                quote = tradier.get_quote(occ_symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("tracking: quote lookup failed for %s: %s", occ_symbol, exc)
                quote = None

            if quote:
                bid = quote.get("bid")
                ask = quote.get("ask")
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    current_mid = round((bid + ask) / 2, 4)

        sig.setdefault("checkpoints", []).append(
            {"date": today.isoformat(), "option_mid": current_mid, "dte": current_dte}
        )

        try:
            sig_date = datetime.strptime(sig.get("date", ""), "%Y-%m-%d").date()
        except ValueError:
            sig_date = today
        elapsed_calendar_days = (today - sig_date).days
        elapsed_trading_days = elapsed_calendar_days * TRADING_DAY_CALENDAR_FACTOR

        should_close = False
        if elapsed_trading_days >= close_after_trading_days:
            should_close = True
        if current_dte is not None and current_dte < close_at_dte:
            should_close = True
        if occ_symbol and current_mid is None:
            # Option no longer quotable (expired, delisted, or no market)
            should_close = True

        if should_close:
            exit_mid = current_mid
            pnl_pct = None
            hit = False
            if exit_mid is not None and entry_mid:
                pnl_pct = round((exit_mid - entry_mid) / entry_mid * 100, 2)
                hit = exit_mid > entry_mid
            sig["status"] = "closed"
            sig["result"] = {
                "closed_date": today.isoformat(),
                "exit_mid": exit_mid,
                "pnl_pct": pnl_pct,
                "hit": hit,
            }
            logger.info(
                "tracking: closed signal %s (%s) hit=%s pnl_pct=%s",
                sig.get("id"),
                sig.get("ticker"),
                hit,
                pnl_pct,
            )

    save_signals(signals, path)
    return signals


def stats(signals: list[dict[str, Any]] | None = None, path: Path | str = DEFAULT_SIGNALS_PATH) -> dict[str, Any]:
    """Compute rolling hit rate and average P/L across closed signals."""
    signals = signals if signals is not None else load_signals(path)

    closed = [s for s in signals if s.get("status") == "closed"]
    open_signals = [s for s in signals if s.get("status") == "open"]

    hits = sum(1 for s in closed if (s.get("result") or {}).get("hit"))
    pnl_values = [
        s["result"]["pnl_pct"]
        for s in closed
        if (s.get("result") or {}).get("pnl_pct") is not None
    ]

    hit_rate = round(hits / len(closed) * 100, 1) if closed else None
    avg_pnl = round(sum(pnl_values) / len(pnl_values), 1) if pnl_values else None

    return {
        "closed": len(closed),
        "open": len(open_signals),
        "hits": hits,
        "hit_rate": hit_rate,
        "avg_pnl_pct": avg_pnl,
    }
