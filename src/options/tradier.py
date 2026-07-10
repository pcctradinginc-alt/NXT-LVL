"""Tradier REST client: quotes, history, option chains, and contract selection.

No SDK — plain requests calls. All public methods are None-tolerant: any API
failure is logged and results in None (or an empty structure) rather than an
exception, so callers can degrade gracefully.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from src.http_utils import get_json

logger = logging.getLogger(__name__)

PROD_BASE_URL = "https://api.tradier.com"
SANDBOX_BASE_URL = "https://sandbox.tradier.com"


class TradierClient:
    """Thin wrapper around the Tradier market data / options REST endpoints."""

    def __init__(self, api_key: str, env: str = "prod"):
        self.api_key = api_key
        self.base_url = SANDBOX_BASE_URL if env == "sandbox" else PROD_BASE_URL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = f"{self.base_url}{path}"
        try:
            return get_json(url, headers=self._headers(), params=params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tradier: GET %s failed: %s", path, exc)
            return None

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Fetch a single quote. Handles both dict and list response shapes."""
        data = self._get("/v1/markets/quotes", params={"symbols": symbol})
        if not data:
            return None
        try:
            quotes = data.get("quotes", {}).get("quote")
        except AttributeError:
            return None
        if quotes is None:
            return None
        if isinstance(quotes, list):
            return quotes[0] if quotes else None
        return quotes

    def get_history(self, symbol: str, start: str, interval: str = "daily") -> list[dict[str, Any]]:
        """Fetch daily history from `start` (YYYY-MM-DD) to today."""
        data = self._get(
            "/v1/markets/history",
            params={"symbol": symbol, "interval": interval, "start": start},
        )
        if not data:
            return []
        try:
            day = data.get("history", {}).get("day")
        except AttributeError:
            return []
        if day is None:
            return []
        if isinstance(day, dict):
            return [day]
        return day

    def get_expirations(self, symbol: str) -> list[str]:
        data = self._get(
            "/v1/markets/options/expirations",
            params={"symbol": symbol, "includeAllRoots": "true", "strikes": "false"},
        )
        if not data:
            return []
        try:
            dates = data.get("expirations", {}).get("date")
        except AttributeError:
            return []
        if dates is None:
            return []
        if isinstance(dates, str):
            return [dates]
        return list(dates)

    def get_chain(self, symbol: str, expiration: str) -> list[dict[str, Any]]:
        data = self._get(
            "/v1/markets/options/chains",
            params={"symbol": symbol, "expiration": expiration, "greeks": "true"},
        )
        if not data:
            return []
        try:
            options = data.get("options", {}).get("option")
        except AttributeError:
            return []
        if options is None:
            return []
        if isinstance(options, dict):
            return [options]
        return options

    def get_three_month_performance_pct(self, symbol: str) -> float | None:
        """Percent price change over the last ~3 months, or None if unavailable."""
        start = (date.today() - timedelta(days=95)).strftime("%Y-%m-%d")
        history = self.get_history(symbol, start=start)
        if not history or len(history) < 2:
            return None
        try:
            first_close = float(history[0]["close"])
            last_close = float(history[-1]["close"])
        except (KeyError, TypeError, ValueError):
            return None
        if first_close == 0:
            return None
        return round((last_close - first_close) / first_close * 100, 2)

    def select_option(
        self,
        symbol: str,
        dte_min: int = 90,
        dte_max: int = 180,
        delta_min: float = 0.60,
        delta_max: float = 0.70,
        min_open_interest: int = 100,
        max_spread_pct: float = 0.10,
    ) -> dict[str, Any] | None:
        """Select the best-fitting long call for the given symbol.

        Strategy:
        1. Filter expirations to the DTE window; prefer the shortest expiry
           that is still >= 120 DTE (to bias toward the middle of the
           window), otherwise the longest available expiry inside the window.
        2. Load that expiration's chain, filter calls by delta, open
           interest, spread.
        3. Rank by (spread ascending, open interest descending) and return
           the best candidate, or None with a logged reason.
        """
        quote = self.get_quote(symbol)
        underlying_price = None
        if quote:
            underlying_price = quote.get("last") or quote.get("close")

        expirations = self.get_expirations(symbol)
        if not expirations:
            logger.info("tradier.select_option(%s): no expirations available", symbol)
            return None

        today = date.today()
        candidates_in_window: list[tuple[int, str]] = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            dte = (exp_date - today).days
            if dte_min <= dte <= dte_max:
                candidates_in_window.append((dte, exp_str))

        if not candidates_in_window:
            logger.info(
                "tradier.select_option(%s): no expirations within DTE window [%d,%d]",
                symbol,
                dte_min,
                dte_max,
            )
            return None

        candidates_in_window.sort()
        preferred = [c for c in candidates_in_window if c[0] >= 120]
        chosen_dte, chosen_exp = preferred[0] if preferred else candidates_in_window[-1]

        chain = self.get_chain(symbol, chosen_exp)
        if not chain:
            logger.info("tradier.select_option(%s): empty chain for expiration %s", symbol, chosen_exp)
            return None

        eligible: list[dict[str, Any]] = []
        for opt in chain:
            if opt.get("option_type") != "call":
                continue
            greeks = opt.get("greeks") or {}
            delta = greeks.get("delta")
            if delta is None:
                continue
            if not (delta_min <= delta <= delta_max):
                continue

            open_interest = opt.get("open_interest") or 0
            if open_interest < min_open_interest:
                continue

            bid = opt.get("bid")
            ask = opt.get("ask")
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            if mid <= 0:
                continue
            spread_pct = (ask - bid) / mid
            if spread_pct > max_spread_pct:
                continue

            iv = greeks.get("mid_iv") or greeks.get("smv_vol")

            eligible.append(
                {
                    "occ_symbol": opt.get("symbol"),
                    "strike": opt.get("strike"),
                    "expiration": chosen_exp,
                    "dte": chosen_dte,
                    "bid": bid,
                    "ask": ask,
                    "mid": round(mid, 4),
                    "delta": delta,
                    "open_interest": open_interest,
                    "spread_pct": round(spread_pct, 4),
                    "underlying_price": underlying_price,
                    "iv": iv,
                }
            )

        if not eligible:
            logger.info(
                "tradier.select_option(%s): no call in expiration %s met delta/OI/spread filters",
                symbol,
                chosen_exp,
            )
            return None

        eligible.sort(key=lambda o: (o["spread_pct"], -o["open_interest"]))
        best = eligible[0]
        logger.info(
            "tradier.select_option(%s): selected %s strike=%s dte=%s delta=%.2f oi=%s",
            symbol,
            best["occ_symbol"],
            best["strike"],
            best["dte"],
            best["delta"],
            best["open_interest"],
        )
        return best

    def select_short_leg(
        self,
        symbol: str,
        expiration: str,
        target_delta: float = 0.32,
        min_open_interest: int = 50,
        max_spread_pct: float = 0.15,
    ) -> dict[str, Any] | None:
        """Select a short call leg (for a debit call spread) on the given expiration.

        Among calls with delta <= `target_delta` (staying below the long
        leg's delta so the spread remains a net debit), sufficient open
        interest, and a tradable bid/ask within the spread cap, pick the one
        whose delta is closest to `target_delta`. Fault-tolerant: any
        failure, empty chain, or no matching call returns None.
        """
        try:
            chain = self.get_chain(symbol, expiration)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "tradier.select_short_leg(%s, %s): chain fetch failed: %s", symbol, expiration, exc
            )
            return None

        if not chain:
            return None

        eligible: list[dict[str, Any]] = []
        for opt in chain:
            if opt.get("option_type") != "call":
                continue
            greeks = opt.get("greeks") or {}
            delta = greeks.get("delta")
            if delta is None or delta > target_delta:
                continue

            open_interest = opt.get("open_interest") or 0
            if open_interest < min_open_interest:
                continue

            bid = opt.get("bid")
            ask = opt.get("ask")
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            if mid <= 0:
                continue
            spread_pct = (ask - bid) / mid
            if spread_pct > max_spread_pct:
                continue

            eligible.append(
                {
                    "occ_symbol": opt.get("symbol"),
                    "strike": opt.get("strike"),
                    "mid": round(mid, 4),
                    "delta": delta,
                    "open_interest": open_interest,
                    "spread_pct": round(spread_pct, 4),
                }
            )

        if not eligible:
            logger.info(
                "tradier.select_short_leg(%s, %s): no call met delta/OI/spread filters",
                symbol,
                expiration,
            )
            return None

        eligible.sort(key=lambda o: abs(o["delta"] - target_delta))
        best = eligible[0]
        logger.info(
            "tradier.select_short_leg(%s, %s): selected %s strike=%s delta=%.2f oi=%s",
            symbol,
            expiration,
            best["occ_symbol"],
            best["strike"],
            best["delta"],
            best["open_interest"],
        )
        return best

    def get_next_earnings_date(self, symbol: str) -> str | None:
        """Best-effort next earnings date via Tradier's beta fundamentals calendar.

        Many Tradier accounts do not have access to this beta endpoint at
        all (403/404/empty), and the response shape varies by account tier.
        This method is therefore deliberately defensive: it recursively
        scans the parsed JSON for any date-like value attached to an
        "earnings" marker and returns the earliest upcoming one. On any
        error, missing access, or unrecognized shape it returns None (logged
        at debug/info, never raises) — this makes the earnings-trap gate
        opt-in: inactive when the data is unavailable, never crashes the
        pipeline.
        """
        data = self._get("/beta/markets/fundamentals/calendars", params={"symbols": symbol})
        if not data:
            logger.debug(
                "tradier.get_next_earnings_date(%s): empty/unavailable response (endpoint may not be entitled)",
                symbol,
            )
            return None

        today = date.today()
        candidates: list[date] = []

        def _looks_like_earnings(node: dict[str, Any]) -> bool:
            for key in ("event", "event_type", "type", "name", "description"):
                value = node.get(key)
                if isinstance(value, str) and "earning" in value.lower():
                    return True
                if key == "event_type" and isinstance(value, int) and value == 8:
                    return True
            return False

        def _parse_date(value: str) -> date | None:
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                return None

        def _walk(node: Any, earnings_context: bool) -> None:
            if isinstance(node, dict):
                node_is_earnings = earnings_context or _looks_like_earnings(node)
                for key, value in node.items():
                    if (
                        node_is_earnings
                        and isinstance(value, str)
                        and key.lower() in ("date", "begin_date", "begindate", "event_date", "eventdate")
                    ):
                        parsed = _parse_date(value)
                        if parsed and parsed >= today:
                            candidates.append(parsed)
                    _walk(value, node_is_earnings)
            elif isinstance(node, list):
                for item in node:
                    _walk(item, earnings_context)

        try:
            _walk(data, False)
        except Exception as exc:  # noqa: BLE001
            logger.debug("tradier.get_next_earnings_date(%s): parse failed: %s", symbol, exc)
            return None

        if not candidates:
            logger.info(
                "tradier.get_next_earnings_date(%s): no upcoming earnings event found "
                "(endpoint may be unavailable for this account)",
                symbol,
            )
            return None

        return min(candidates).isoformat()
