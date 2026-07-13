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

# The reconstructable, weightable score components (same four
# walkforward.score_universe_asof now emits — divergence, theme_momentum,
# breadth, momentum_12_1). trend_ok / regime_risk_on are GATES, not weighted
# components, so they are intentionally NOT in this list.
COMPONENTS: tuple[str, ...] = ("divergence", "theme_momentum", "breadth", "momentum_12_1")

NEUTRAL_SCORE = 50.0

# Default adoption criteria (all must hold over the validation folds). Passed
# through `run_calibration(..., adoption=...)` can override any of these.
DEFAULT_ADOPTION: dict[str, float] = {
    "min_median_val_ic_total": 0.0,      # median out-of-sample IC(total) must be > this
    "min_median_top_q_option_hit": 0.55,  # median top-quartile option hit-rate must be >= this
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


def _quartile_option_hits(
    scored: list[tuple[float, float | None]],
) -> tuple[float | None, float | None]:
    """Top- and bottom-quartile option hit-rates from (total_score, opt_return)
    pairs. Ranks by total_score; the top quartile is the highest-scoring 25%,
    the bottom the lowest 25%. Only pairs with a non-None option return count.
    Returns (None, None) when there are fewer than 4 usable pairs (too few to
    form a meaningful quartile split) or either quartile ends up empty.
    """
    valid = [(t, o) for (t, o) in scored if o is not None]
    if len(valid) < 4:
        return None, None
    ordered = sorted(valid, key=lambda pair: pair[0])
    n = len(ordered)
    bottom: list[float] = []
    top: list[float] = []
    for i, (_total, opt) in enumerate(ordered):
        frac = (i + 1) / n
        if frac <= 0.25:
            bottom.append(opt)
        elif frac > 0.75:
            top.append(opt)
    if not top or not bottom:
        return None, None
    top_hit = sum(1 for o in top if o > 0) / len(top)
    bottom_hit = sum(1 for o in bottom if o > 0) / len(bottom)
    return top_hit, bottom_hit


def _median(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return statistics.median(nums)


def run_calibration(
    samples: list[dict[str, Any]],
    n_folds: int = 4,
    horizon: str = "90",
    adoption: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Walk-forward fold calibration with strict out-of-sample validation.

    For each validation fold k in 1..n_folds-1: TRAIN on folds [0..k-1]
    (component_ics -> derive_weights), then VALIDATE on fold k (never used in
    training) by re-scoring each fold-k sample under those weights and
    measuring IC(total) vs fwd[horizon] and a top-vs-bottom-quartile option
    hit-rate (via opt[horizon], None-safe). Aggregates the validation folds
    and applies the adoption criteria (defaults `DEFAULT_ADOPTION`, overridable
    via `adoption`).

    Returns:
      {
        "weights_final": derive_weights over ALL samples (production weights —
                          only meaningful if `passed`),
        "folds": [ {fold_index, n_train, n_val, train_ics, weights,
                    val_ic_total, top_q_option_hit, bottom_q_option_hit}, ...],
        "validation": {median_val_ic_total, median_top_q_option_hit,
                       top_beats_bottom_fraction, passed},
        "n_samples", "n_folds", "horizon", "adoption_criteria",
        "generated_at", "notes": [...],
      }
    """
    crit = {**DEFAULT_ADOPTION, **(adoption or {})}
    folds = time_folds(samples, n_folds)

    fold_reports: list[dict[str, Any]] = []
    val_ics: list[float | None] = []
    top_hits: list[float | None] = []
    top_beats_bottom_flags: list[bool] = []

    for k in range(1, len(folds)):
        train = [s for j in range(k) for s in folds[j]]
        val = folds[k]
        if not train or not val:
            continue

        train_ics = component_ics(train, horizon)
        weights = derive_weights(train_ics)

        scored_fwd: list[tuple[float, float | None]] = []
        for s in val:
            total = score_with_weights(s, weights)
            fwd = (s.get("fwd") or {}).get(horizon)
            scored_fwd.append((total, fwd))

        fwd_pairs = [(t, f) for (t, f) in scored_fwd if f is not None]
        val_ic = (
            spearman([t for t, _ in fwd_pairs], [f for _, f in fwd_pairs])
            if len(fwd_pairs) >= 3
            else None
        )
        val_ic = round(val_ic, 4) if val_ic is not None else None

        scored_opt = [(score_with_weights(s, weights), (s.get("opt") or {}).get(horizon)) for s in val]
        top_hit, bottom_hit = _quartile_option_hits(scored_opt)

        val_ics.append(val_ic)
        top_hits.append(top_hit)
        if top_hit is not None and bottom_hit is not None:
            top_beats_bottom_flags.append(top_hit > bottom_hit)

        fold_reports.append(
            {
                "fold_index": k,
                "n_train": len(train),
                "n_val": len(val),
                "train_ics": train_ics,
                "weights": {c: round(w, 4) for c, w in weights.items()},
                "val_ic_total": val_ic,
                "top_q_option_hit": round(top_hit, 4) if top_hit is not None else None,
                "bottom_q_option_hit": round(bottom_hit, 4) if bottom_hit is not None else None,
                "degenerate_train": is_degenerate(train_ics),
            }
        )

    median_val_ic = _median(val_ics)
    median_top_hit = _median(top_hits)
    top_beats_bottom_fraction = (
        sum(1 for f in top_beats_bottom_flags if f) / len(top_beats_bottom_flags)
        if top_beats_bottom_flags
        else 0.0
    )

    passed = bool(
        median_val_ic is not None
        and median_val_ic > crit["min_median_val_ic_total"]
        and median_top_hit is not None
        and median_top_hit >= crit["min_median_top_q_option_hit"]
        and top_beats_bottom_fraction >= crit["min_top_beats_bottom_fraction"]
    )

    final_ics = component_ics(samples, horizon)
    weights_final = derive_weights(final_ics)

    notes = [
        "Weights are derived from max(IC, 0) only — a negative or unmeasurable "
        "(None) component IC gets ZERO weight and is NEVER inverted, which "
        "would fit in-sample noise and fail out-of-sample.",
        "Validation folds are strictly out-of-sample: fold k's weights are "
        "trained ONLY on the earlier folds [0..k-1] and never see fold k.",
        f"passed={passed} reflects the adoption criteria "
        f"(median_val_ic_total > {crit['min_median_val_ic_total']}, "
        f"median_top_q_option_hit >= {crit['min_median_top_q_option_hit']}, "
        f"top_beats_bottom_fraction >= {crit['min_top_beats_bottom_fraction']}). "
        "passed=False means the calibrated scoring did NOT show out-of-sample "
        "edge — keep it in observation-only mode, do not gate live trades on it.",
        "weights_final is derived over ALL samples for production use and is "
        "only meaningful when passed=True; when passed=False it is reported "
        "for transparency, not for adoption.",
    ]
    if is_degenerate(final_ics):
        notes.append(
            "No component had a positive IC over the full sample — weights_final "
            "fell back to EQUAL weights (a neutral non-signal), which is itself "
            "evidence of no measured edge."
        )
    n = len(samples)
    if n < 100:
        notes.append(
            f"Small sample (n_samples={n}): fold ICs and option hit-rates carry "
            "wide sampling error — treat any 'passed' here as provisional until "
            "n grows (CONCEPT_PROFIT.md Phase A targets n ~1500-2500)."
        )

    return {
        "weights_final": {c: round(w, 4) for c, w in weights_final.items()},
        "folds": fold_reports,
        "validation": {
            "median_val_ic_total": round(median_val_ic, 4) if median_val_ic is not None else None,
            "median_top_q_option_hit": round(median_top_hit, 4) if median_top_hit is not None else None,
            "top_beats_bottom_fraction": round(top_beats_bottom_fraction, 4),
            "passed": passed,
        },
        "n_samples": n,
        "n_folds": n_folds,
        "horizon": horizon,
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
        f"n_samples={calib['n_samples']}  n_folds={calib['n_folds']}  horizon={calib['horizon']}d"
    )
    print()

    print("Per-fold train ICs -> weights, then out-of-sample validation:")
    if not calib["folds"]:
        print("  (no usable folds — too few samples to train+validate)")
    for f in calib["folds"]:
        ics = f["train_ics"]
        weights = f["weights"]
        ic_str = ", ".join(f"{c}={_fmt(ics.get(c))}" for c in COMPONENTS)
        w_str = ", ".join(f"{c}={weights.get(c, 0.0):.2f}" for c in COMPONENTS)
        print(
            f"  fold {f['fold_index']}: n_train={f['n_train']:>4} n_val={f['n_val']:>4}"
            f"{'  [degenerate train]' if f.get('degenerate_train') else ''}"
        )
        print(f"    train IC:  {ic_str}")
        print(f"    weights:   {w_str}")
        print(
            f"    validate:  val_ic_total={_fmt(f['val_ic_total'])}  "
            f"top_q_option_hit={_fmt_pct(f['top_q_option_hit'])}  "
            f"bottom_q_option_hit={_fmt_pct(f['bottom_q_option_hit'])}"
        )
    print()

    v = calib["validation"]
    print("Validation verdict (aggregated over folds):")
    print(f"  median val IC(total):        {_fmt(v['median_val_ic_total'])}")
    print(f"  median top-quartile opt hit: {_fmt_pct(v['median_top_q_option_hit'])}")
    print(f"  folds top-quartile beats bottom-quartile: {_fmt_pct(v['top_beats_bottom_fraction'])}")
    crit = calib["adoption_criteria"]
    print(
        f"  adoption criteria: median_val_ic > {crit['min_median_val_ic_total']}, "
        f"median_top_q_opt_hit >= {crit['min_median_top_q_option_hit']}, "
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
    print("=== CALIBRATION SUMMARY ===")
    print(f"n_samples: {calib['n_samples']}  n_folds: {calib['n_folds']}  horizon: {calib['horizon']}d")
    print(f"median val IC(total): {_fmt(v['median_val_ic_total'])}")
    print(f"median top-quartile option hit: {_fmt_pct(v['median_top_q_option_hit'])}")
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
    parser.add_argument("--mock", action="store_true", help="Use deterministic synthetic samples, no network required")
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

    calib = run_calibration(samples, n_folds=args.folds, horizon="90")

    _print_report(calib)
    _print_summary(calib)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"\nCalibration written to {CALIBRATION_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
