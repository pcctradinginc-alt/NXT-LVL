"""Black-Scholes call pricing (stdlib only) for rough option re-valuation.

Used by the reward evaluator to answer a simple yes/no question: "given the
underlying's price now and the option's remaining time-to-expiry, would the
originally selected call still be worth more than the entry mid?" This is a
theoretical estimate, not a live market quote (which would require another
Tradier options-chain call for a strike that may no longer be near-the-money).
"""

from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (stdlib only, no scipy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T_years: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price.

    S: underlying spot price
    K: strike
    T_years: time to expiry in years (must be > 0)
    r: risk-free rate (annualized, e.g. 0.04)
    sigma: implied volatility (annualized, e.g. 0.5 for 50%)
    """
    if S <= 0 or K <= 0:
        return 0.0
    if T_years <= 0:
        # At/after expiry: intrinsic value only.
        return max(0.0, S - K)
    if sigma <= 0:
        # No volatility: discounted intrinsic value.
        return max(0.0, S - K * math.exp(-r * T_years))

    sqrt_t = math.sqrt(T_years)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    call = S * _norm_cdf(d1) - K * math.exp(-r * T_years) * _norm_cdf(d2)
    return max(0.0, call)


def bs_call_delta(S: float, K: float, T_years: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta = N(d1).

    Degenerate cases (at/after expiry, or zero volatility) collapse to the
    intrinsic-value indicator: 1.0 if the option would finish in the money
    (S > K), else 0.0.
    """
    if T_years <= 0:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        return 1.0 if S > K else 0.0
    if S <= 0 or K <= 0:
        return 0.0

    sqrt_t = math.sqrt(T_years)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * sqrt_t)
    return _norm_cdf(d1)


def solve_strike_for_delta(
    S: float, target_delta: float, T_years: float, r: float, sigma: float
) -> float:
    """Binary-search the strike whose call delta is closest to `target_delta`.

    Delta is monotonically decreasing in strike (higher K -> lower delta), so
    a simple bisection over K in [0.3*S, 2.0*S] converges quickly. ~40
    iterations gives ample precision for strike granularity used elsewhere in
    this project.
    """
    low = 0.3 * S
    high = 2.0 * S
    for _ in range(40):
        mid = (low + high) / 2.0
        delta = bs_call_delta(S, mid, T_years, r, sigma)
        if delta > target_delta:
            # Delta too high -> strike too low -> search the upper half.
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def estimate_call_value(
    underlying_now: float,
    strike: float,
    dte_days_remaining: float,
    entry_iv: float | None = None,
    r: float = 0.04,
    sigma_fallback: float = 0.5,
) -> float:
    """Rough re-valuation of a call given current underlying price and remaining DTE.

    Falls back to `sigma_fallback` when no entry IV was recorded. Intended
    only for the reward evaluator's directional profitability check, not for
    precise pricing.
    """
    sigma = entry_iv if entry_iv and entry_iv > 0 else sigma_fallback
    t_years = max(0.0, dte_days_remaining) / 365.0
    return bs_call_price(underlying_now, strike, t_years, r, sigma)
