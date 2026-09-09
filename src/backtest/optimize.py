"""Walk-forward fold CALIBRATION + out-of-sample VALIDATION of the walk-forward
scoring (CONCEPT_PROFIT.md Phase C).

This is the honest gate between "we built some features" and "we bet money on
them". It never touches network data itself: it consumes the compact per-
(ticker, as_of) sample list that `walkforward.run_walkforward(...,
include_samples=True)` already produced (components + forward/option returns +
trend/regime gate flags), and re-derives scoring WEIGHTS purely from measured
Information Coefficients — no black-box optimizer, no parameter search, nothing
that could silently overfit.

The method (deliberately simple and explainable):

  1. `time_folds`  — split the samples into contiguous time blocks.
  2. For each validation fold k (1..n_folds-1): TRAIN only on the earlier
     folds [0..k-1] (compute each component's IC, turn IC into weights via
     `derive_weights`), then VALIDATE on fold k — data the training never saw —
     by re-scoring every fold-k sample under those weights and measuring
     IC(total) plus a top-vs-bottom-quartile option hit-rate.
  3. Aggregate the validation folds and apply explicit ADOPTION CRITERIA. If
     they pass, the calibrated weights showed out-of-sample edge; if not, the
     honest answer is `passed=False` — keep scoring in observation-only mode.

Weights come from `max(IC, 0)` only: a negative IC contributes ZERO weight, it
is NOT inverted. Inverting a negative in-sample IC is the classic way to fit
noise and get punished out-of-sample, which is exactly the failure mode Phase
A/B set out to avoid.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.backtest import walkforward
from src.backtest.calibrate import spearman
from src.config import DATA_DIR, load_settings
from src.options.tradier import TradierClient

logger = logging.getLogger(__name__)

CALIBRATION_PATH = DATA_DIR / "scoring_calibration.json"
# Experiment output (Hebel 1): calibration restricted to the regime-filtered
# subset (risk-on AND uptrend). Written to a SEPARATE file so it never
# overwrites the production calibration — adopting it would additionally
# require production to hard-gate live trades on the SAME trend_ok+regime
# filter, so it stays an explicit experiment until that consistency is wired.
FILTERED_CALIBRATION_PATH = DATA_DIR / "scoring_calibration_filtered.json"


def regime_filtered_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only samples taken in a confirmed risk-on regime AND uptrend.

    A sample qualifies iff BOTH `regime_risk_on` and `trend_ok` are exactly
    True (None — insufficient trailing history for the gate — is treated as
    'not confirmed', i.e. excluded, matching walkforward.filtered_buckets).
    The hypothesis (CONCEPT_PROFIT.md Hebel 1): factors may only carry edge in
    the right regime, which mixing all regimes together washes out.
    """
    return [
        s for s in samples
        if s.get("regime_risk_on") is True and s.get("trend_ok") is True
    ]

# The reconstructable, weightable score components (same nine
# walkforward.score_universe_asof now emits — divergence, theme_momentum,
# breadth, momentum_12_1, plus the CONCEPT_PROFIT.md Phase B/C candidate
# factors reversal_1m/low_vol/high_52w/rs (src.analysis.factors, pure
# cross-sectional price factors reconstructable from cached Tradier daily
# history) and revenue_growth (SEC EDGAR XBRL, see
# walkforward._revenue_yoy_asof)). trend_ok / regime_risk_on are GATES, not
# weighted components, so they are intentionally NOT in this list.
COMPONENTS: tuple[str, ...] = (
    "divergence",
    "theme_momentum",
    "breadth",
    "momentum_12_1",
    "reversal_1m",
    "low_vol",
    "high_52w",
    "rs",
    "revenue_growth",
)

NEUTRAL_SCORE = 50.0

# Crude round-trip cost haircut applied to every OPTION return (never to
# stock returns) before any option-track statistic is computed: 5% of
# notional premium, meant to stand in for bid/ask spread + commission on a
# single-leg option round-trip. This is NOT a real transaction-cost model —
# real spreads vary hugely by strike/expiry/liquidity — it exists so the
# option track is not judged on frictionless, unrealistically-clean fills.
# Overridable via `--option-cost`.
DEFAULT_OPTION_COST_HAIRCUT = 0.05

# Number of block-bootstrap resamples for the uncertainty estimates (change
# 4). Resampling is over whole `as_of` dates (never individual rows) with a
# fixed seed for reproducibility. Overridable via `--bootstrap`.
DEFAULT_BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 20240301

# Assumed within-cluster correlation used by the `effective_n` design-effect
# estimate (see `effective_sample_size`). This is a deliberately simple,
# defensible-but-not-precise stand-in for "how much do samples sharing the
# same as_of date move together" — real cross-sectional correlation varies by
# regime and is not separately measured here.
ASSUMED_WITHIN_CLUSTER_CORRELATION = 0.5

# Default adoption criteria (all must hold over the validation folds). Passed
# through `run_calibration(..., adoption=...)` can override any of these.
#
# Expectancy, not hit-rate, is the primary option-track gate: option payoffs
# are asymmetric (a call can lose ~100% of premium or gain several hundred
# percent), so a 45%-hit system can be profitable and a 60%-hit system can
# lose money. Hit-rate is still computed and reported (see "option_track" in
# `run_calibration`'s output) for sanity/transparency, but it is intentionally
# NOT part of this pass/fail conjunction — see the "hit-rate demoted" note
# `run_calibration` adds to its `notes` list for the reasoning.
DEFAULT_ADOPTION: dict[str, float] = {
    "min_median_val_ic_total": 0.0,  # median out-of-sample IC(total) must be > this
    "min_median_top_q_option_expectancy": 0.0,  # median top-quartile NET option expectancy must be > this
    "min_top_beats_bottom_fraction": 0.5,  # >= this fraction of folds must have top_q > bottom_q option hit
}


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def time_folds(samples: list[dict[str, Any]], n_folds: int) -> list[list[dict[str, Any]]]:
    """Split `samples` into `n_folds` contiguous, time-ordered blocks of ~equal count.

    Samples are sorted by (as_of, ticker) first — so every fold is a whole
    time slice and no fold ever leaks a later date into an earlier one — then
    chopped into `n_folds` chunks whose sizes differ by at most 1 (the first
    `n % n_folds` folds get one extra). If `len(samples) < n_folds` some
    trailing folds are empty (rather than raising); `run_calibration` skips any
    fold with an empty train or validation slice.
    """
    if n_folds < 1:
        n_folds = 1
    ordered = sorted(samples, key=lambda s: (str(s.get("as_of") or ""), str(s.get("ticker") or "")))
    n = len(ordered)
    base, rem = divmod(n, n_folds)
    folds: list[list[dict[str, Any]]] = []
    idx = 0
    for i in range(n_folds):
        size = base + (1 if i < rem else 0)
        folds.append(ordered[idx : idx + size])
        idx += size
    return folds


# ---------------------------------------------------------------------------
# Purged walk-forward CV with embargo (Lopez de Prado)
# ---------------------------------------------------------------------------


def _parse_as_of(value: Any) -> date | None:
    """Parse a sample's `as_of` field, robust to it being a `str` (the shape
    walkforward.run_walkforward actually emits, `date.isoformat()`) or an
    already-a-`date`/`datetime` value (defensive, e.g. hand-built test
    fixtures or a JSON round-trip that used a date-aware decoder).

    Returns None when `value` is neither, or is a string that does not parse
    as YYYY-MM-DD (only the first 10 characters are considered, so an
    ISO-8601 timestamp with a time component still parses). None is an honest
    "could not determine this sample's date" — callers must not guess.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def purge_train(
    train: list[dict[str, Any]],
    val_start: date,
    horizon_days: int,
    embargo_days: int,
) -> tuple[list[dict[str, Any]], int]:
    """Purge + embargo (Lopez de Prado, *Advances in Financial Machine
    Learning*): drop training samples whose LABEL WINDOW could overlap the
    validation fold, not just samples dated after it.

    A training sample's `fwd`/`opt` label is determined by prices over the
    `horizon_days` days AFTER its `as_of` date. Plain contiguous time-block
    splitting (`time_folds`) only guarantees `as_of` itself precedes the
    validation fold — it does NOT guarantee the label window does. If a
    training sample's 90-day-forward outcome window reaches into (or close
    to) the validation fold, the two folds are measuring correlated,
    overlapping price history and any "out-of-sample" edge is partly an
    artifact of that overlap. So:

        drop a training sample iff  as_of + horizon_days + embargo_days >= val_start

    `embargo_days` is an extra buffer past the label window itself, guarding
    against slower-moving shared effects (regime, theme momentum) that
    correlate samples dated close together even once their labels don't
    literally overlap; the caller (`run_calibration`) defaults it to
    `horizon_days`, i.e. a buffer as wide as the label window itself.

    A training sample with an unparseable `as_of` (see `_parse_as_of`) is
    ALSO dropped — the conservative choice: we cannot verify its label window
    doesn't reach into the validation fold, so it is excluded from training
    rather than risk silent leakage. This can only shrink `n_train`, never
    inflate it, which is the safe direction for the honest side of this gate.

    Returns `(purged_train, n_purged)`.
    """
    kept: list[dict[str, Any]] = []
    n_purged = 0
    cutoff = timedelta(days=horizon_days + embargo_days)
    for s in train:
        as_of = _parse_as_of(s.get("as_of"))
        if as_of is None or as_of + cutoff >= val_start:
            n_purged += 1
            continue
        kept.append(s)
    return kept, n_purged


# ---------------------------------------------------------------------------
# IC -> weights
# ---------------------------------------------------------------------------


def component_ics(samples: list[dict[str, Any]], horizon: str = "90") -> dict[str, float | None]:
    """Spearman IC of each component's value vs. that sample's `fwd[horizon]`.

    Computed over every sample whose `fwd[horizon]` is available (not None).
    A component's IC is None when there are fewer than 3 usable pairs or the
    component (or the forward return) has zero variance — `spearman` already
    returns None in those cases, and None is preserved here as an honest
    "not measurable", never coerced to 0.0.
    """
    ics: dict[str, float | None] = {}
    valid = [s for s in samples if (s.get("fwd") or {}).get(horizon) is not None]
    ys = [float(s["fwd"][horizon]) for s in valid]
    for comp in COMPONENTS:
        xs = [float((s.get("components") or {}).get(comp, NEUTRAL_SCORE)) for s in valid]
        ic = spearman(xs, ys) if len(valid) >= 3 else None
        ics[comp] = round(ic, 4) if ic is not None else None
    return ics


def is_degenerate(ics: dict[str, float | None]) -> bool:
    """True when no component has a positive IC (so `derive_weights` falls back
    to equal weights). This is exactly the "no measurable edge in any
    component" case — the caller uses it to know the returned weights are the
    neutral equal-weight fallback rather than a signal-driven allocation.
    """
    return not any(isinstance(v, (int, float)) and v > 0 for v in ics.values())


def derive_weights(
    ics: dict[str, float | None], w_min: float = 0.0, w_max: float = 0.6
) -> dict[str, float]:
    """Turn component ICs into normalized weights, edge-only and overfit-guarded.

    Rules (CONCEPT_PROFIT.md Phase C):
      - raw weight ∝ max(IC, 0): a None or negative IC contributes ZERO (it is
        NOT inverted — inverting an in-sample negative is how you fit noise).
      - the proportional weights are clipped to [w_min, w_max] (a cap on any
        single component's dominance — an overfitting guard) and renormalized
        to sum to 1.
      - if NO component has a positive IC (`is_degenerate(ics)` is True), fall
        back to EQUAL weights across all components and let the caller detect
        that via `is_degenerate(ics)`.

    Always returns a weight for every key in `ics` (0.0 for the non-positive
    ones), and the returned weights sum to 1 (within float error) whenever
    `ics` is non-empty.
    """
    keys = list(ics.keys())
    if not keys:
        return {}

    if is_degenerate(ics):
        equal = 1.0 / len(keys)
        return {k: equal for k in keys}

    raw = {k: max(float(v), 0.0) if isinstance(v, (int, float)) else 0.0 for k, v in ics.items()}
    total_raw = sum(raw.values())
    # total_raw > 0 is guaranteed here (not degenerate), but guard anyway.
    prop = {k: (v / total_raw if total_raw > 0 else 0.0) for k, v in raw.items()}
    clipped = {k: min(max(v, w_min), w_max) for k, v in prop.items()}
    total_clipped = sum(clipped.values())
    if total_clipped <= 0:
        equal = 1.0 / len(keys)
        return {k: equal for k in keys}
    return {k: v / total_clipped for k, v in clipped.items()}


def score_with_weights(sample: dict[str, Any], weights: dict[str, float]) -> float:
    """Weighted sum of `sample`'s components under `weights`.

    A component named in `weights` but missing from the sample's components
    contributes the NEUTRAL score (50) rather than 0 — a missing factor should
    not drag the total down, it should simply be uninformative.
    """
    components = sample.get("components") or {}
    return sum(weight * float(components.get(comp, NEUTRAL_SCORE)) for comp, weight in weights.items())


# ---------------------------------------------------------------------------
# Calibration + out-of-sample validation
# ---------------------------------------------------------------------------


def _quartile_split(
    scored: list[tuple[float, float | None]],
) -> tuple[list[float] | None, list[float] | None]:
    """Split (total_score, value) pairs into top-quartile and bottom-quartile
    VALUE lists, ranked by total_score. Only pairs with a non-None value
    count. Returns (None, None) when there are fewer than 4 usable pairs (too
    few to form a meaningful quartile split) or either quartile ends up
    empty. Used for both the stock track (`value` = fwd return) and the
    option track (`value` = raw, pre-haircut option return).
    """
    valid = [(t, v) for (t, v) in scored if v is not None]
    if len(valid) < 4:
        return None, None
    ordered = sorted(valid, key=lambda pair: pair[0])
    n = len(ordered)
    bottom: list[float] = []
    top: list[float] = []
    for i, (_total, val) in enumerate(ordered):
        frac = (i + 1) / n
        if frac <= 0.25:
            bottom.append(val)
        elif frac > 0.75:
            top.append(val)
    if not top or not bottom:
        return None, None
    return top, bottom


def _hit_rate(values: list[float]) -> float:
    return sum(1 for v in values if v > 0) / len(values)


def _downside_deviation(returns: list[float], mar: float = 0.0) -> float | None:
    """Sortino-style downside (semi-)deviation of `returns` below `mar`.

    sqrt(mean of squared shortfalls below `mar`, i.e. min(0, r - mar)**2,
    computed over ALL of `returns` — not just the negative ones, which is the
    standard Sortino denominator (returns at/above `mar` contribute a
    shortfall of exactly 0). Returns None — rather than 0 or a fabricated
    ratio — when there are fewer than 2 returns, or when NONE of them fall
    below `mar` (there is no measured downside to divide by).
    """
    if len(returns) < 2:
        return None
    shortfalls = [min(0.0, r - mar) for r in returns]
    if not any(s < 0 for s in shortfalls):
        return None
    mean_sq = sum(s * s for s in shortfalls) / len(shortfalls)
    dd = math.sqrt(mean_sq)
    return dd if dd > 0 else None


def _sortino(returns: list[float]) -> float | None:
    """Mean return / downside deviation. None whenever `_downside_deviation`
    is None (too few samples, or no downside at all) — never divides by a
    fabricated or zero denominator.
    """
    dd = _downside_deviation(returns)
    if dd is None:
        return None
    return statistics.mean(returns) / dd


def _median(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return statistics.median(nums)


def _round_or_none(v: float | None, ndigits: int = 4) -> float | None:
    return round(v, ndigits) if isinstance(v, (int, float)) else None


def option_track_stats(
    scored_opt: list[tuple[float, float | None]], cost_haircut: float
) -> dict[str, float | None]:
    """Top/bottom-quartile OPTION-track statistics from (total_score,
    raw_opt_return) pairs, net of `cost_haircut` (subtracted from every
    option return before any statistic below is computed — see
    `DEFAULT_OPTION_COST_HAIRCUT`).

    Returns a dict with (all None-safe, see `_quartile_split`):
      top_q_option_hit, bottom_q_option_hit   — fraction of net returns > 0
      top_q_option_expectancy                 — mean net return (primary criterion)
      top_q_option_median                     — median net return ("typical" trade P/L)
      top_q_option_sortino                    — mean / downside-deviation of net returns
      n_top, n_bottom                         — quartile sizes actually used
    """
    top, bottom = _quartile_split(scored_opt)
    if top is None or bottom is None:
        return {
            "top_q_option_hit": None,
            "bottom_q_option_hit": None,
            "top_q_option_expectancy": None,
            "top_q_option_median": None,
            "top_q_option_sortino": None,
            "n_top": 0,
            "n_bottom": 0,
        }
    top_net = [v - cost_haircut for v in top]
    bottom_net = [v - cost_haircut for v in bottom]
    return {
        "top_q_option_hit": _hit_rate(top_net),
        "bottom_q_option_hit": _hit_rate(bottom_net),
        "top_q_option_expectancy": statistics.mean(top_net),
        "top_q_option_median": statistics.median(top_net),
        "top_q_option_sortino": _sortino(top_net),
        "n_top": len(top_net),
        "n_bottom": len(bottom_net),
    }


def stock_track_stats(scored_fwd: list[tuple[float, float | None]]) -> dict[str, float | None]:
    """Top/bottom-quartile STOCK-track statistics from (total_score,
    fwd_return) pairs — plain forward returns, no cost haircut (there is no
    "trade cost" for simply holding the stock in this research context; the
    haircut models OPTION round-trip friction specifically).

    Returns: top_q_stock_mean, bottom_q_stock_mean, stock_top_minus_bottom
    (all None when the quartile split is unavailable, see `_quartile_split`).
    """
    top, bottom = _quartile_split(scored_fwd)
    if top is None or bottom is None:
        return {
            "top_q_stock_mean": None,
            "bottom_q_stock_mean": None,
            "stock_top_minus_bottom": None,
        }
    top_mean = statistics.mean(top)
    bottom_mean = statistics.mean(bottom)
    return {
        "top_q_stock_mean": top_mean,
        "bottom_q_stock_mean": bottom_mean,
        "stock_top_minus_bottom": top_mean - bottom_mean,
    }


# ---------------------------------------------------------------------------
# Effective sample size + block bootstrap (change 4)
# ---------------------------------------------------------------------------


def cluster_by_as_of(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group `samples` by their raw `as_of` string/value (stringified), so
    every sample dated the same day lands in the same cluster regardless of
    whether `as_of` is a str or a date. Order of the returned dict follows
    first-seen `as_of`.
    """
    clusters: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        key = str(s.get("as_of"))
        clusters.setdefault(key, []).append(s)
    return clusters


def effective_sample_size(
    samples: list[dict[str, Any]], rho: float = ASSUMED_WITHIN_CLUSTER_CORRELATION
) -> dict[str, float | int]:
    """Cluster `samples` by `as_of` date and estimate how many INDEPENDENT
    observations they are really worth.

    ~60 tickers sampled on the same as_of date share the same market regime,
    theme-momentum readings, and (for overlapping horizons) largely the same
    forward price history — they are not 60 independent draws. The standard
    "design effect" correction for clustered data is:

        effective_n = n / (1 + (avg_cluster_size - 1) * rho)

    where `rho` is the assumed average within-cluster correlation (constant
    `ASSUMED_WITHIN_CLUSTER_CORRELATION`, default 0.5 — a deliberately
    round, moderate guess, NOT a measured value; real within-cluster
    correlation varies by regime and is not separately estimated here). This
    is a standard, well-known formula (Kish's design effect) but its
    precision is only as good as the `rho` guess — treat `effective_n` as an
    order-of-magnitude sanity check on `n_samples`, never as an exact count.

    Returns {"n_clusters", "avg_cluster_size", "effective_n"}. `effective_n`
    is always <= `n_samples` (equal only when there is exactly one sample per
    cluster, i.e. rho's effect vanishes).
    """
    n = len(samples)
    clusters = cluster_by_as_of(samples)
    n_clusters = len(clusters)
    if n_clusters == 0:
        return {"n_clusters": 0, "avg_cluster_size": 0.0, "effective_n": 0.0}
    avg_cluster_size = n / n_clusters
    design_effect = 1 + (avg_cluster_size - 1) * rho
    effective_n = n / design_effect if design_effect > 0 else float(n)
    return {
        "n_clusters": n_clusters,
        "avg_cluster_size": round(avg_cluster_size, 2),
        "effective_n": round(effective_n, 1),
    }


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0, 100]) of `values`. Assumes
    `values` is non-empty; sorts a copy.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def block_bootstrap_ci(
    samples: list[dict[str, Any]],
    statistic_fn: Any,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> tuple[float | None, float | None] | None:
    """Block bootstrap: resample whole `as_of` DATES with replacement (never
    individual rows — that would pretend same-day samples are independent,
    the exact problem `effective_sample_size` documents), rebuild a pooled
    sample list from the resampled dates, and recompute `statistic_fn(pooled)`
    `n_resamples` times. Returns the (`lo_pct`, `hi_pct`) percentile
    confidence interval of the resampled statistic values, e.g. a (5th,
    95th) band.

    Deterministic for a given `seed` (default `BOOTSTRAP_SEED`) — same input
    samples + same seed always produce the same CI, so results are
    reproducible across runs.

    Returns None if there are no clusters, or if `statistic_fn` returned a
    usable (non-None) value on fewer than 2 resamples (too little signal for
    a meaningful percentile band).
    """
    clusters = cluster_by_as_of(samples)
    keys = list(clusters.keys())
    if not keys:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        chosen = rng.choices(keys, k=len(keys))
        pooled = [s for key in chosen for s in clusters[key]]
        stat = statistic_fn(pooled)
        if isinstance(stat, (int, float)):
            values.append(float(stat))
    if len(values) < 2:
        return None
    return round(_percentile(values, lo_pct), 4), round(_percentile(values, hi_pct), 4)


def run_calibration(
    samples: list[dict[str, Any]],
    n_folds: int = 4,
    horizon: str = "90",
    adoption: dict[str, float] | None = None,
    embargo_days: int | None = None,
    option_cost_haircut: float = DEFAULT_OPTION_COST_HAIRCUT,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Walk-forward fold calibration with PURGED, EMBARGOED out-of-sample
    validation.

    For each validation fold k in 1..n_folds-1: TRAIN on folds [0..k-1] MINUS
    any sample purged by `purge_train` (its label window would otherwise
    overlap fold k, or fall inside the `embargo_days` buffer past it — see
    `purge_train`'s docstring), then VALIDATE on fold k (never used in
    training, and now genuinely non-overlapping) by re-scoring each fold-k
    sample under the purged-training-derived weights.

    Two tracks are measured and reported SEPARATELY (change 3):
      - stock_track: IC of total score vs. the actual stock `fwd[horizon]`
        return, plus a top-vs-bottom-quartile stock return spread. This is
        the PRIMARY evidence of any selection edge — real prices, no model.
      - option_track: expectancy/median/Sortino/hit-rate of the top quartile
        of `opt[horizon]` (net of `option_cost_haircut`). The option return
        is a Black-Scholes-modeled overlay (realized vol as an IV proxy, no
        skew/crush/spreads/liquidity) — research-grade at best, and MUST NOT
        be trusted without stock-selection edge existing first.

    `embargo_days` defaults to `horizon_days` (derived from the `horizon`
    string) when None. `adoption` overrides any key of `DEFAULT_ADOPTION`
    (expectancy-primary, hit-rate reported-only — see that constant's
    docstring).

    Returns (backward-compatible top-level validation keys `median_val_ic_total`
    / `median_top_q_option_hit` / `top_beats_bottom_fraction` / `passed` are
    preserved for `src/analysis/calibration.py` and any other existing reader
    of `validation["passed"]` / `weights_final`):
      {
        "weights_final": derive_weights over ALL samples (production weights —
                          only meaningful if `passed`),
        "folds": [ {fold_index, n_train_raw, n_train, n_purged, val_start,
                    n_val, train_ics, weights, val_ic_total,
                    top_q_option_hit, bottom_q_option_hit,
                    top_q_option_expectancy, top_q_option_median,
                    top_q_option_sortino, top_q_stock_mean,
                    bottom_q_stock_mean, stock_top_minus_bottom}, ...],
        "validation": {
            median_val_ic_total, median_top_q_option_hit,
            top_beats_bottom_fraction, passed,   # backward-compatible
            "stock_track": {...}, "option_track": {...}, "uncertainty": {...},
        },
        "n_samples", "n_folds", "horizon", "embargo_days", "option_cost_haircut",
        "adoption_criteria", "generated_at", "notes": [...],
      }
    """
    crit = {**DEFAULT_ADOPTION, **(adoption or {})}
    horizon_days = int(horizon)
    embargo = horizon_days if embargo_days is None else embargo_days
    folds = time_folds(samples, n_folds)

    fold_reports: list[dict[str, Any]] = []
    val_ics: list[float | None] = []
    top_hits: list[float | None] = []
    top_expectancies: list[float | None] = []
    top_medians: list[float | None] = []
    top_sortinos: list[float | None] = []
    stock_spreads: list[float | None] = []
    top_q_stock_means: list[float | None] = []
    top_beats_bottom_flags: list[bool] = []
    # Folds that had training data before purging but none after it. Tracked
    # explicitly because an aggressive embargo can silently erase a fold, and
    # a fold that vanished from `folds` while `n_folds` still claims 4 would
    # overstate how much out-of-sample evidence this verdict actually rests on.
    folds_emptied_by_purge: list[int] = []

    for k in range(1, len(folds)):
        train_raw = [s for j in range(k) for s in folds[j]]
        val = folds[k]
        if not train_raw or not val:
            continue

        val_dates = [d for d in (_parse_as_of(s.get("as_of")) for s in val) if d is not None]
        val_start = min(val_dates) if val_dates else None
        if val_start is not None:
            train, n_purged = purge_train(train_raw, val_start, horizon_days, embargo)
        else:
            # No parseable as_of anywhere in the validation fold: we cannot
            # tell where fold k starts, so we cannot safely purge — training
            # proceeds unpurged rather than guessing. This is a degenerate
            # input case (malformed samples), not the expected path.
            train, n_purged = train_raw, 0
        if not train:
            folds_emptied_by_purge.append(k)
            continue

        train_ics = component_ics(train, horizon)
        weights = derive_weights(train_ics)

        scored_fwd: list[tuple[float, float | None]] = []
        scored_opt: list[tuple[float, float | None]] = []
        for s in val:
            total = score_with_weights(s, weights)
            scored_fwd.append((total, (s.get("fwd") or {}).get(horizon)))
            scored_opt.append((total, (s.get("opt") or {}).get(horizon)))

        fwd_pairs = [(t, f) for (t, f) in scored_fwd if f is not None]
        val_ic = (
            spearman([t for t, _ in fwd_pairs], [f for _, f in fwd_pairs])
            if len(fwd_pairs) >= 3
            else None
        )
        val_ic = round(val_ic, 4) if val_ic is not None else None

        stock_stats = stock_track_stats(scored_fwd)
        opt_stats = option_track_stats(scored_opt, option_cost_haircut)

        val_ics.append(val_ic)
        top_hits.append(opt_stats["top_q_option_hit"])
        top_expectancies.append(opt_stats["top_q_option_expectancy"])
        top_medians.append(opt_stats["top_q_option_median"])
        top_sortinos.append(opt_stats["top_q_option_sortino"])
        stock_spreads.append(stock_stats["stock_top_minus_bottom"])
        top_q_stock_means.append(stock_stats["top_q_stock_mean"])
        if opt_stats["top_q_option_hit"] is not None and opt_stats["bottom_q_option_hit"] is not None:
            top_beats_bottom_flags.append(opt_stats["top_q_option_hit"] > opt_stats["bottom_q_option_hit"])

        fold_reports.append(
            {
                "fold_index": k,
                "n_train_raw": len(train_raw),
                "n_train": len(train),
                "n_purged": n_purged,
                "val_start": val_start.isoformat() if val_start is not None else None,
                "n_val": len(val),
                "train_ics": train_ics,
                "weights": {c: round(w, 4) for c, w in weights.items()},
                "val_ic_total": val_ic,
                "top_q_option_hit": _round_or_none(opt_stats["top_q_option_hit"]),
                "bottom_q_option_hit": _round_or_none(opt_stats["bottom_q_option_hit"]),
                "top_q_option_expectancy": _round_or_none(opt_stats["top_q_option_expectancy"]),
                "top_q_option_median": _round_or_none(opt_stats["top_q_option_median"]),
                "top_q_option_sortino": _round_or_none(opt_stats["top_q_option_sortino"]),
                "top_q_stock_mean": _round_or_none(stock_stats["top_q_stock_mean"]),
                "bottom_q_stock_mean": _round_or_none(stock_stats["bottom_q_stock_mean"]),
                "stock_top_minus_bottom": _round_or_none(stock_stats["stock_top_minus_bottom"]),
                "degenerate_train": is_degenerate(train_ics),
            }
        )

    median_val_ic = _median(val_ics)
    median_top_hit = _median(top_hits)
    median_top_expectancy = _median(top_expectancies)
    median_top_median_pl = _median(top_medians)
    median_top_sortino = _median(top_sortinos)
    median_stock_spread = _median(stock_spreads)
    median_top_q_stock_mean = _median(top_q_stock_means)
    top_beats_bottom_fraction = (
        sum(1 for f in top_beats_bottom_flags if f) / len(top_beats_bottom_flags)
        if top_beats_bottom_flags
        else 0.0
    )

    passed = bool(
        median_val_ic is not None
        and median_val_ic > crit["min_median_val_ic_total"]
        and median_top_expectancy is not None
        and median_top_expectancy > crit["min_median_top_q_option_expectancy"]
        and top_beats_bottom_fraction >= crit["min_top_beats_bottom_fraction"]
    )

    final_ics = component_ics(samples, horizon)
    weights_final = derive_weights(final_ics)
    n = len(samples)

    # --- effective sample size + block bootstrap (change 4) ---------------
    eff = effective_sample_size(samples)

    def _pooled_ic(pooled: list[dict[str, Any]]) -> float | None:
        scored = [(score_with_weights(s, weights_final), (s.get("fwd") or {}).get(horizon)) for s in pooled]
        pairs = [(t, f) for (t, f) in scored if f is not None]
        if len(pairs) < 3:
            return None
        return spearman([t for t, _ in pairs], [f for _, f in pairs])

    def _pooled_top_q_expectancy(pooled: list[dict[str, Any]]) -> float | None:
        scored = [(score_with_weights(s, weights_final), (s.get("opt") or {}).get(horizon)) for s in pooled]
        stats = option_track_stats(scored, option_cost_haircut)
        return stats["top_q_option_expectancy"]

    pooled_ic_ci = block_bootstrap_ci(samples, _pooled_ic, n_resamples=bootstrap_resamples)
    top_q_expectancy_ci = block_bootstrap_ci(samples, _pooled_top_q_expectancy, n_resamples=bootstrap_resamples)

    notes = [
        "Weights are derived from max(IC, 0) only — a negative or unmeasurable "
        "(None) component IC gets ZERO weight and is NEVER inverted, which "
        "would fit in-sample noise and fail out-of-sample.",
        "Validation folds are strictly out-of-sample AND purged+embargoed: "
        "fold k's weights are trained only on earlier folds, with any "
        "training sample whose label window could overlap fold k (or fall "
        f"within the {embargo}-day embargo past it) dropped by `purge_train` "
        "before training — see each fold's n_train_raw vs n_train/n_purged.",
        "The STOCK track (real forward returns) is the PRIMARY evidence of "
        "selection edge. The OPTION track is a Black-Scholes-modeled overlay "
        "(realized volatility as an IV proxy — no skew, no vol crush, no real "
        "spreads or historical liquidity) and must NOT be trusted on its own: "
        "stock-selection edge has to exist first before any long-call-vs-"
        "call-spread optimization built on top of it is even meaningful.",
        "Hit-rate is demoted to reported-only, NOT part of the pass/fail "
        "conjunction: option payoffs are asymmetric (a losing call loses up "
        "to 100% of premium, a winner can gain hundreds of percent), so a "
        "45%-hit system can be profitable and a 60%-hit system can lose "
        "money. Net expectancy (mean net return per trade, after "
        f"option_cost_haircut={option_cost_haircut}) is the primary option-"
        "track criterion instead.",
        f"passed={passed} reflects the adoption criteria "
        f"(median_val_ic_total > {crit['min_median_val_ic_total']}, "
        f"median_top_q_option_expectancy > {crit['min_median_top_q_option_expectancy']}, "
        f"top_beats_bottom_fraction >= {crit['min_top_beats_bottom_fraction']}). "
        "passed=False means the calibrated scoring did NOT show out-of-sample "
        "edge — keep it in observation-only mode, do not gate live trades on it.",
        "weights_final is derived over ALL samples for production use and is "
        "only meaningful when passed=True; when passed=False it is reported "
        "for transparency, not for adoption.",
        f"n_samples={n} pools {eff['n_clusters']} distinct as_of dates (avg "
        f"{eff['avg_cluster_size']} samples/date); the design-effect estimate "
        f"(Kish's formula, assumed within-cluster correlation="
        f"{ASSUMED_WITHIN_CLUSTER_CORRELATION}) puts effective_n at only "
        f"~{eff['effective_n']} — treat n_samples alone as an OVERSTATEMENT of "
        "independent evidence.",
    ]
    if is_degenerate(final_ics):
        notes.append(
            "No component had a positive IC over the full sample — weights_final "
            "fell back to EQUAL weights (a neutral non-signal), which is itself "
            "evidence of no measured edge."
        )
    if n < 100:
        notes.append(
            f"Small sample (n_samples={n}): fold ICs and option stats carry "
            "wide sampling error — treat any 'passed' here as provisional until "
            "n grows (CONCEPT_PROFIT.md Phase A targets n ~1500-2500)."
        )
    if folds_emptied_by_purge:
        notes.append(
            f"Purging emptied the training set of {len(folds_emptied_by_purge)} "
            f"fold(s) {folds_emptied_by_purge} entirely, so they contributed NO "
            f"validation evidence: this verdict rests on {len(fold_reports)} "
            f"usable fold(s), not n_folds={n_folds}. A wider cadence or fewer "
            "folds would leave more training history intact past the "
            f"{embargo}-day embargo."
        )

    return {
        "weights_final": {c: round(w, 4) for c, w in weights_final.items()},
        "folds": fold_reports,
        "n_usable_folds": len(fold_reports),
        "folds_emptied_by_purge": folds_emptied_by_purge,
        "validation": {
            # --- backward-compatible top-level keys ---
            "median_val_ic_total": round(median_val_ic, 4) if median_val_ic is not None else None,
            "median_top_q_option_hit": _round_or_none(median_top_hit),
            "top_beats_bottom_fraction": round(top_beats_bottom_fraction, 4),
            "passed": passed,
            # --- change 3: stock vs. option tracks reported separately ---
            "stock_track": {
                "median_val_ic_total": round(median_val_ic, 4) if median_val_ic is not None else None,
                "median_top_q_stock_mean": _round_or_none(median_top_q_stock_mean),
                "median_stock_top_minus_bottom": _round_or_none(median_stock_spread),
                "note": "Primary evidence of selection edge — real forward stock returns, no model.",
            },
            "option_track": {
                "median_top_q_option_hit": _round_or_none(median_top_hit),
                "median_top_q_option_expectancy": _round_or_none(median_top_expectancy),
                "median_top_q_option_median_pl": _round_or_none(median_top_median_pl),
                "median_top_q_option_sortino": _round_or_none(median_top_sortino),
                "top_beats_bottom_fraction": round(top_beats_bottom_fraction, 4),
                "option_cost_haircut": option_cost_haircut,
                "note": "Modeled overlay (Black-Scholes + realized-vol IV proxy) — research-grade, not to be trusted standalone.",
            },
            # --- change 4: effective sample size + bootstrap CIs ---
            "uncertainty": {
                "n_clusters": eff["n_clusters"],
                "avg_cluster_size": eff["avg_cluster_size"],
                "effective_n": eff["effective_n"],
                "assumed_within_cluster_correlation": ASSUMED_WITHIN_CLUSTER_CORRELATION,
                "bootstrap": {
                    "n_resamples": bootstrap_resamples,
                    "seed": BOOTSTRAP_SEED,
                    "pooled_ic_ci": list(pooled_ic_ci) if pooled_ic_ci is not None else None,
                    "top_q_option_expectancy_ci": (
                        list(top_q_expectancy_ci) if top_q_expectancy_ci is not None else None
                    ),
                },
            },
        },
        "n_samples": n,
        "n_folds": n_folds,
        "horizon": horizon,
        "embargo_days": embargo,
        "option_cost_haircut": option_cost_haircut,
        "adoption_criteria": crit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


HEADER_LINE = (
    "Scoring calibration — walk-forward fold train/validate over the "
    "walk-forward samples. Reports whether the IC-calibrated scoring showed "
    "OUT-OF-SAMPLE edge; passed=false means keep scoring in observation-only "
    "mode (no validated edge — do not gate live trades on it)."
)


def _fmt(v: Any) -> str:
    return f"{v:+.3f}" if isinstance(v, (int, float)) else "n/a"


def _fmt_pct(v: Any) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "n/a"


def _print_report(calib: dict[str, Any]) -> None:
    print(HEADER_LINE)
    print()
    print(
        f"n_samples={calib['n_samples']}  n_folds={calib['n_folds']}  horizon={calib['horizon']}d  "
        f"embargo_days={calib.get('embargo_days')}  option_cost_haircut={calib.get('option_cost_haircut')}"
    )
    print()

    print("Per-fold train ICs -> weights, then out-of-sample validation (purged+embargoed):")
    if not calib["folds"]:
        print("  (no usable folds — too few samples to train+validate)")
    for f in calib["folds"]:
        ics = f["train_ics"]
        weights = f["weights"]
        ic_str = ", ".join(f"{c}={_fmt(ics.get(c))}" for c in COMPONENTS)
        w_str = ", ".join(f"{c}={weights.get(c, 0.0):.2f}" for c in COMPONENTS)
        print(
            f"  fold {f['fold_index']}: n_train_raw={f.get('n_train_raw', f['n_train']):>4} "
            f"n_purged={f.get('n_purged', 0):>4} n_train={f['n_train']:>4} n_val={f['n_val']:>4} "
            f"val_start={f.get('val_start')}"
            f"{'  [degenerate train]' if f.get('degenerate_train') else ''}"
        )
        print(f"    train IC:  {ic_str}")
        print(f"    weights:   {w_str}")
        print(
            f"    stock:     val_ic_total={_fmt(f['val_ic_total'])}  "
            f"top_q_stock_mean={_fmt_pct(f.get('top_q_stock_mean'))}  "
            f"top_minus_bottom={_fmt_pct(f.get('stock_top_minus_bottom'))}"
        )
        print(
            f"    option:    top_q_hit={_fmt_pct(f['top_q_option_hit'])}  "
            f"bottom_q_hit={_fmt_pct(f['bottom_q_option_hit'])}  "
            f"top_q_expectancy={_fmt_pct(f.get('top_q_option_expectancy'))}  "
            f"top_q_median_pl={_fmt_pct(f.get('top_q_option_median'))}  "
            f"top_q_sortino={_fmt(f.get('top_q_option_sortino'))}"
        )
    print()

    v = calib["validation"]
    st = v.get("stock_track", {})
    ot = v.get("option_track", {})
    unc = v.get("uncertainty", {})
    print("Validation verdict (aggregated over folds):")
    print("  STOCK track (primary evidence of selection edge):")
    print(f"    median val IC(total):              {_fmt(st.get('median_val_ic_total'))}")
    print(f"    median top-quartile stock return:  {_fmt_pct(st.get('median_top_q_stock_mean'))}")
    print(f"    median top-minus-bottom spread:    {_fmt_pct(st.get('median_stock_top_minus_bottom'))}")
    print("  OPTION track (modeled overlay — do not trust standalone):")
    print(f"    median top-quartile hit-rate (reported only): {_fmt_pct(ot.get('median_top_q_option_hit'))}")
    print(f"    median top-quartile net expectancy:           {_fmt_pct(ot.get('median_top_q_option_expectancy'))}")
    print(f"    median top-quartile net median P/L:           {_fmt_pct(ot.get('median_top_q_option_median_pl'))}")
    print(f"    median top-quartile Sortino:                  {_fmt(ot.get('median_top_q_option_sortino'))}")
    print(f"    folds top-quartile beats bottom-quartile:     {_fmt_pct(v['top_beats_bottom_fraction'])}")
    print("  UNCERTAINTY (clustering + block bootstrap):")
    print(
        f"    n_clusters={unc.get('n_clusters')}  avg_cluster_size={unc.get('avg_cluster_size')}  "
        f"effective_n≈{unc.get('effective_n')} (n_samples={calib['n_samples']})"
    )
    boot = unc.get("bootstrap", {})
    print(
        f"    bootstrap ({boot.get('n_resamples')} resamples, seed={boot.get('seed')}): "
        f"pooled IC 5-95% CI={boot.get('pooled_ic_ci')}  "
        f"top_q_option_expectancy 5-95% CI={boot.get('top_q_option_expectancy_ci')}"
    )
    crit = calib["adoption_criteria"]
    print(
        f"  adoption criteria: median_val_ic > {crit['min_median_val_ic_total']}, "
        f"median_top_q_option_expectancy > {crit['min_median_top_q_option_expectancy']}, "
        f"top_beats_bottom >= {crit['min_top_beats_bottom_fraction']}"
    )
    print(f"  ==> PASSED: {v['passed']}")
    print()

    wf = calib["weights_final"]
    print("Final weights (over all samples — adopt only if PASSED):")
    print("  " + ", ".join(f"{c}={wf.get(c, 0.0):.3f}" for c in COMPONENTS))
    print()

    print("Notes / caveats:")
    for note in calib["notes"]:
        print(f"  - {note}")


def _print_summary(calib: dict[str, Any]) -> None:
    v = calib["validation"]
    st = v.get("stock_track", {})
    ot = v.get("option_track", {})
    unc = v.get("uncertainty", {})
    print("=== CALIBRATION SUMMARY ===")
    print(f"n_samples: {calib['n_samples']}  n_folds: {calib['n_folds']}  horizon: {calib['horizon']}d")
    if calib.get("folds_emptied_by_purge"):
        print(
            f"usable folds: {calib.get('n_usable_folds')} "
            f"({len(calib['folds_emptied_by_purge'])} emptied by purge/embargo)"
        )
    print(f"effective_n (clustered): {unc.get('effective_n')}  n_clusters: {unc.get('n_clusters')}")
    print(f"median val IC(total) [stock track]: {_fmt(st.get('median_val_ic_total'))}")
    print(f"median top-quartile stock return: {_fmt_pct(st.get('median_top_q_stock_mean'))}")
    print(f"median top-quartile option net expectancy [option track]: {_fmt_pct(ot.get('median_top_q_option_expectancy'))}")
    print(f"median top-quartile option hit-rate (reported only): {_fmt_pct(ot.get('median_top_q_option_hit'))}")
    print(f"top-beats-bottom fraction: {_fmt_pct(v['top_beats_bottom_fraction'])}")
    print(f"PASSED (out-of-sample edge validated): {v['passed']}")
    wf = calib["weights_final"]
    print("final weights: " + ", ".join(f"{c}={wf.get(c, 0.0):.3f}" for c in COMPONENTS))
    print("=== END SUMMARY ===")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NXT LVL — walk-forward scoring calibration + out-of-sample validation (CONCEPT_PROFIT.md Phase C)"
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: 3 years before --end)")
    parser.add_argument(
        "--end", type=str, default=None, help="End date YYYY-MM-DD (default: today minus the longest horizon)"
    )
    parser.add_argument(
        "--cadence", type=int, default=30, dest="cadence_days", help="Days between as-of samples (default 30)"
    )
    parser.add_argument("--folds", type=int, default=4, help="Number of time folds (default 4)")
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        dest="embargo_days",
        help=(
            "Extra purge buffer (days) past a training sample's label window before the "
            "validation fold start (default: same as the horizon, e.g. 90)"
        ),
    )
    parser.add_argument(
        "--option-cost",
        type=float,
        default=DEFAULT_OPTION_COST_HAIRCUT,
        dest="option_cost_haircut",
        help=f"Round-trip option cost haircut, fraction of notional premium (default {DEFAULT_OPTION_COST_HAIRCUT})",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        dest="bootstrap_resamples",
        help=f"Number of block-bootstrap resamples for the uncertainty CIs (default {DEFAULT_BOOTSTRAP_RESAMPLES})",
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic synthetic samples, no network required")
    parser.add_argument(
        "--filtered",
        action="store_true",
        help=(
            "Hebel 1 experiment: calibrate ONLY on the regime-filtered subset "
            "(risk-on AND uptrend). Writes to scoring_calibration_filtered.json, "
            "never the production file."
        ),
    )
    return parser.parse_args(argv)


def _load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]] | None, int]:
    """Produce the walk-forward samples for calibration.

    Returns (samples, exit_code): on the live path with no TRADIER_API_KEY,
    returns (None, 1) after printing a clear message; otherwise (samples, 0).
    """
    horizons = walkforward.DEFAULT_HORIZONS
    end = (
        walkforward._parse_date(args.end)
        if args.end
        else date.today() - timedelta(days=max(horizons))
    )
    start = walkforward._parse_date(args.start) if args.start else end - timedelta(days=365 * 3)

    if args.mock:
        price_series = walkforward._generate_mock_price_series(start, end)
        tickers = list(walkforward.MOCK_TICKERS.keys())
        themes = walkforward.MOCK_THEMES
        report = walkforward.run_walkforward(
            None,
            tickers,
            themes,
            start=start,
            end=end,
            cadence_days=args.cadence_days,
            horizons=horizons,
            benchmarks=walkforward.DEFAULT_BENCHMARKS,
            mock=True,
            price_series=price_series,
            include_samples=True,
        )
        return report.get("samples", []), 0

    settings = load_settings()
    if not settings.tradier_api_key:
        print(
            "TRADIER_API_KEY is not set. Live calibration needs a Tradier API key to fetch "
            "historical prices for the walk-forward samples. Set TRADIER_API_KEY, or run "
            "with --mock for an offline synthetic-data run."
        )
        return None, 1
    tradier = TradierClient(settings.tradier_api_key, settings.tradier_env)
    tickers = sorted(
        settings.watchlist_tickers()
        | {str(t).upper() for theme in settings.themes for t in (theme.get("tickers") or [])}
    )
    themes = settings.themes
    if not tickers or not themes:
        print("No tickers/themes found in config.yaml (stages[].tickers / themes[]).")
        return None, 1
    report = walkforward.run_walkforward(
        tradier,
        tickers,
        themes,
        start=start,
        end=end,
        cadence_days=args.cadence_days,
        horizons=horizons,
        benchmarks=walkforward.DEFAULT_BENCHMARKS,
        mock=False,
        include_samples=True,
    )
    return report.get("samples", []), 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    args = parse_args(argv)

    samples, code = _load_samples(args)
    if samples is None:
        return code

    out_path = CALIBRATION_PATH
    if args.filtered:
        n_before = len(samples)
        samples = regime_filtered_samples(samples)
        out_path = FILTERED_CALIBRATION_PATH
        print(
            f"[Hebel 1] regime-filtered calibration: {len(samples)}/{n_before} samples "
            "kept (regime_risk_on AND trend_ok both True)\n"
        )

    calib = run_calibration(
        samples,
        n_folds=args.folds,
        horizon="90",
        embargo_days=args.embargo_days,
        option_cost_haircut=args.option_cost_haircut,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    if args.filtered:
        calib["filter"] = "regime_risk_on AND trend_ok (both True)"
        calib.setdefault("notes", []).append(
            "EXPERIMENT (Hebel 1): calibrated ONLY on risk-on + uptrend samples. "
            "Adopting a PASSED result here would require production to HARD-GATE "
            "live trades on the same trend_ok+regime filter (production currently "
            "hard-gates regime but treats trend_ok as a soft risk flag)."
        )

    _print_report(calib)
    _print_summary(calib)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"\nCalibration written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
