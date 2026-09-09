"""Offline tests for the fold calibration + out-of-sample validation optimizer
(src/backtest/optimize.py).

No network: all tests build synthetic compact sample lists (the same shape
walkforward.run_walkforward(..., include_samples=True) emits) directly.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from src.backtest import optimize


def _mk_sample(as_of: str, ticker: str, components: dict, fwd90, opt90) -> dict:
    """Build one compact sample in the shape optimize.py consumes."""
    return {
        "as_of": as_of,
        "ticker": ticker,
        "components": {
            "divergence": 50.0,
            "theme_momentum": 50.0,
            "breadth": 50.0,
            "momentum_12_1": 50.0,
            **components,
        },
        "trend_ok": True,
        "regime_risk_on": True,
        "fwd": {"90": fwd90},
        "opt": {"90": opt90},
    }


# ---------------------------------------------------------------------------
# time_folds
# ---------------------------------------------------------------------------


def test_time_folds():
    base = date(2024, 1, 1)
    samples = [
        {"as_of": (base + timedelta(days=i)).isoformat(), "ticker": "T", "components": {}, "fwd": {}, "opt": {}}
        for i in range(10)
    ]
    # Feed them shuffled to prove time_folds sorts internally.
    shuffled = list(samples)
    random.Random(1).shuffle(shuffled)

    folds = optimize.time_folds(shuffled, 3)

    assert len(folds) == 3
    # ~equal sizes, differing by at most 1, first folds get the extra.
    assert [len(f) for f in folds] == [4, 3, 3]

    # Flattened folds are globally time-ordered.
    flat = [s for f in folds for s in f]
    assert [s["as_of"] for s in flat] == sorted(s["as_of"] for s in samples)

    # Contiguous, non-overlapping in time: each fold ends strictly before the next begins.
    for earlier, later in zip(folds, folds[1:]):
        assert max(s["as_of"] for s in earlier) < min(s["as_of"] for s in later)


def test_time_folds_more_folds_than_samples():
    samples = [{"as_of": "2024-01-01", "ticker": "T", "components": {}, "fwd": {}, "opt": {}}]
    folds = optimize.time_folds(samples, 4)
    assert len(folds) == 4
    assert [len(f) for f in folds] == [1, 0, 0, 0]  # no crash, trailing folds empty


# ---------------------------------------------------------------------------
# component_ics + derive_weights
# ---------------------------------------------------------------------------


def test_component_ics_and_derive_weights():
    """momentum_12_1 positively correlated with fwd, breadth negatively →
    momentum gets positive weight, breadth gets 0 (negatives are NOT inverted),
    weights sum to 1.
    """
    base = date(2024, 1, 1)
    samples = []
    for i in range(10):
        samples.append(
            _mk_sample(
                (base + timedelta(days=i)).isoformat(),
                "T",
                {"momentum_12_1": float(i), "breadth": float(10 - i)},
                fwd90=float(i),
                opt90=None,
            )
        )

    ics = optimize.component_ics(samples, "90")
    assert ics["momentum_12_1"] is not None and ics["momentum_12_1"] > 0.9  # near +1
    assert ics["breadth"] is not None and ics["breadth"] < -0.9  # near -1
    # constant components have zero variance -> None (honest, not 0.0)
    assert ics["divergence"] is None
    assert ics["theme_momentum"] is None

    weights = optimize.derive_weights(ics)
    assert weights["momentum_12_1"] > 0.0
    assert weights["breadth"] == 0.0  # negative IC -> zero weight, never inverted
    assert weights["divergence"] == 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert not optimize.is_degenerate(ics)


def test_derive_weights_all_negative_degenerate():
    """All-negative (or None) ICs → degenerate flag True + equal weights."""
    ics = {"divergence": -0.1, "theme_momentum": -0.2, "breadth": -0.05, "momentum_12_1": None}

    assert optimize.is_degenerate(ics) is True

    weights = optimize.derive_weights(ics)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    # equal weights across all four components
    for w in weights.values():
        assert abs(w - 0.25) < 1e-9


def test_derive_weights_clip_and_normalize():
    """Two positive ICs of equal magnitude split 50/50, sum to 1, others 0."""
    ics = {"divergence": None, "theme_momentum": 0.2, "breadth": -0.1, "momentum_12_1": 0.2}
    weights = optimize.derive_weights(ics)
    assert abs(weights["theme_momentum"] - 0.5) < 1e-9
    assert abs(weights["momentum_12_1"] - 0.5) < 1e-9
    assert weights["breadth"] == 0.0
    assert weights["divergence"] == 0.0
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_score_with_weights_missing_component_is_neutral():
    weights = {"momentum_12_1": 1.0}
    # component present
    s1 = {"components": {"momentum_12_1": 80.0}}
    assert optimize.score_with_weights(s1, weights) == 80.0
    # component missing -> neutral 50, not 0
    s2 = {"components": {}}
    assert optimize.score_with_weights(s2, weights) == optimize.NEUTRAL_SCORE


# ---------------------------------------------------------------------------
# run_calibration
# ---------------------------------------------------------------------------


def test_run_calibration_passes_on_edge():
    """A component (momentum_12_1) that genuinely predicts BOTH the forward
    return and the option outcome should validate out-of-sample: passed=True
    and the final weights concentrate on that component.

    Samples are spaced 30 days apart (like the production cadence) rather
    than daily: with the default embargo (= horizon = 90 days), a purge
    window of horizon+embargo=180 days would swallow an entire fold if the
    samples were only a few days apart. 30-day spacing over 80 samples makes
    each fold ~600 days wide, so purging removes only the most recent ~180
    days of each training fold (see the dedicated purge_train tests below for
    the boundary behavior in isolation) while leaving the genuine signal
    intact in what remains.
    """
    base = date(2024, 1, 1)
    samples = []
    for j in range(80):
        m = (j % 10) * 10.0  # 0,10,...,90 — a full momentum spread inside each fold
        samples.append(
            _mk_sample(
                (base + timedelta(days=30 * j)).isoformat(),
                "T",
                {"momentum_12_1": m},
                fwd90=m,  # forward return rank-tracks momentum -> IC +1
                opt90=0.5 if m >= 50 else -0.5,  # high momentum -> option winner
            )
        )

    calib = optimize.run_calibration(samples, n_folds=4, horizon="90")

    assert calib["validation"]["passed"] is True
    assert calib["validation"]["median_val_ic_total"] > 0.0
    # Purging actually removed training rows in at least one fold.
    assert any(f["n_purged"] > 0 for f in calib["folds"])
    assert all(f["n_train"] <= f["n_train_raw"] for f in calib["folds"])

    ot = calib["validation"]["option_track"]
    st = calib["validation"]["stock_track"]
    assert ot["median_top_q_option_expectancy"] > 0.0  # primary criterion
    assert ot["median_top_q_option_hit"] >= 0.55  # still true here, reported only
    assert calib["validation"]["top_beats_bottom_fraction"] >= 0.5
    assert st["median_top_q_stock_mean"] > 0.0

    wf = calib["weights_final"]
    # weights concentrate on the genuinely-predictive component.
    assert wf["momentum_12_1"] == max(wf.values())
    assert wf["momentum_12_1"] > 0.9
    assert wf["breadth"] == 0.0
    assert wf["divergence"] == 0.0


def test_run_calibration_generalizes_to_expanded_components():
    """CONCEPT_PROFIT.md Phase B/C: run_calibration must work over the
    EXPANDED COMPONENTS set (9, not just the original 4) — a new candidate
    factor (low_vol) that genuinely predicts forward returns should earn
    weight and validate out-of-sample, while a same-shaped but genuinely
    RANDOM new candidate factor (high_52w) should end up with ~0 final
    weight — proving derive_weights' max(IC, 0)-only rule (never invert a
    negative IC) generalizes to the new factors, not just the original four.
    """
    base = date(2024, 1, 1)
    rng = random.Random(99)
    samples = []
    for j in range(80):
        lv = (j % 10) * 10.0  # 0,10,...,90 — a full spread inside each fold
        noise = rng.uniform(0, 100)  # unrelated to forward return
        samples.append(
            _mk_sample(
                (base + timedelta(days=30 * j)).isoformat(),
                "T",
                {"low_vol": lv, "high_52w": noise},
                fwd90=lv,  # forward return rank-tracks low_vol -> IC +1
                opt90=0.5 if lv >= 50 else -0.5,
            )
        )

    calib = optimize.run_calibration(samples, n_folds=4, horizon="90")

    assert calib["validation"]["passed"] is True
    assert calib["validation"]["median_val_ic_total"] > 0.0

    wf = calib["weights_final"]
    assert wf["low_vol"] == max(wf.values())
    assert wf["low_vol"] > 0.9
    assert wf["high_52w"] == 0.0  # negative full-sample IC -> zero weight, never inverted
    # the untouched original components stay at their honest "unmeasurable" state
    assert wf["divergence"] == 0.0
    assert wf["momentum_12_1"] == 0.0


def test_run_calibration_fails_on_noise():
    """Random, no-signal samples must NOT pass (honest 'no edge found')."""
    rng = random.Random(42)
    base = date(2024, 1, 1)
    samples = []
    for j in range(48):
        samples.append(
            _mk_sample(
                (base + timedelta(days=j)).isoformat(),
                "T",
                {
                    "momentum_12_1": rng.uniform(0, 100),
                    "divergence": rng.uniform(0, 100),
                    "theme_momentum": rng.uniform(0, 100),
                    "breadth": rng.uniform(0, 100),
                },
                fwd90=rng.uniform(-10, 10),
                opt90=rng.choice([-0.5, 0.5]),
            )
        )

    calib = optimize.run_calibration(samples, n_folds=4, horizon="90")
    assert calib["validation"]["passed"] is False


def test_run_calibration_report_shape():
    """The returned dict has the documented Phase C contract, including the
    backward-compatible top-level validation keys (src/analysis/calibration.py
    reads validation['passed'] and weights_final) plus the new stock_track /
    option_track / uncertainty sub-blocks.
    """
    base = date(2024, 1, 1)
    samples = [
        _mk_sample((base + timedelta(days=j)).isoformat(), "T", {"momentum_12_1": float(j % 10 * 10)}, float(j), 0.1)
        for j in range(20)
    ]
    calib = optimize.run_calibration(samples, n_folds=4, horizon="90")

    # Backward-compatible keys must still be present (existing consumers).
    assert set(calib["validation"].keys()) >= {
        "median_val_ic_total",
        "median_top_q_option_hit",
        "top_beats_bottom_fraction",
        "passed",
        "stock_track",
        "option_track",
        "uncertainty",
    }
    assert set(calib["validation"]["stock_track"].keys()) >= {
        "median_val_ic_total",
        "median_top_q_stock_mean",
        "median_stock_top_minus_bottom",
    }
    assert set(calib["validation"]["option_track"].keys()) >= {
        "median_top_q_option_hit",
        "median_top_q_option_expectancy",
        "median_top_q_option_median_pl",
        "median_top_q_option_sortino",
        "top_beats_bottom_fraction",
        "option_cost_haircut",
    }
    assert set(calib["validation"]["uncertainty"].keys()) >= {
        "n_clusters",
        "avg_cluster_size",
        "effective_n",
        "bootstrap",
    }
    assert "weights_final" in calib
    assert "folds" in calib and isinstance(calib["folds"], list)
    assert calib["n_samples"] == 20
    assert "generated_at" in calib
    assert isinstance(calib["notes"], list) and calib["notes"]
    for f in calib["folds"]:
        assert set(f.keys()) >= {
            "fold_index",
            "n_train_raw",
            "n_train",
            "n_purged",
            "val_start",
            "n_val",
            "train_ics",
            "weights",
            "val_ic_total",
            "top_q_option_hit",
            "bottom_q_option_hit",
            "top_q_option_expectancy",
            "top_q_option_median",
            "top_q_option_sortino",
            "top_q_stock_mean",
            "bottom_q_stock_mean",
            "stock_top_minus_bottom",
        }


def test_regime_filtered_samples():
    from src.backtest import optimize as opt
    samples = [
        {"ticker": "A", "regime_risk_on": True, "trend_ok": True},    # keep
        {"ticker": "B", "regime_risk_on": True, "trend_ok": False},   # drop (downtrend)
        {"ticker": "C", "regime_risk_on": False, "trend_ok": True},   # drop (risk-off)
        {"ticker": "D", "regime_risk_on": None, "trend_ok": True},    # drop (unconfirmed)
        {"ticker": "E", "regime_risk_on": True, "trend_ok": None},    # drop (unconfirmed)
        {"ticker": "F", "regime_risk_on": True, "trend_ok": True},    # keep
    ]
    kept = opt.regime_filtered_samples(samples)
    assert [s["ticker"] for s in kept] == ["A", "F"]
    assert opt.regime_filtered_samples([]) == []


# ---------------------------------------------------------------------------
# purge_train / embargo (change 1)
# ---------------------------------------------------------------------------


def _mk_train_row(as_of) -> dict:
    return {"as_of": as_of, "ticker": "T", "components": {}, "fwd": {}, "opt": {}}


def test_purge_train_removes_overlapping_samples():
    """A training sample whose label window (as_of + horizon_days) reaches
    into the embargo zone before val_start must be dropped; one dated early
    enough to clear the whole horizon+embargo buffer must be kept.
    """
    base = date(2024, 1, 1)
    val_start = base + timedelta(days=200)
    train = [_mk_train_row((base + timedelta(days=d)).isoformat()) for d in range(200)]

    kept, n_purged = optimize.purge_train(train, val_start, horizon_days=90, embargo_days=90)

    # cutoff = horizon_days + embargo_days = 180 -> drop iff as_of + 180 >= val_start,
    # i.e. as_of (as an offset from base) >= 200 - 180 = 20.
    kept_offsets = {(date.fromisoformat(s["as_of"]) - base).days for s in kept}
    assert kept_offsets == set(range(20))  # offsets 0..19 survive
    assert n_purged == 200 - 20
    assert len(kept) + n_purged == len(train)


def test_purge_train_embargo_boundary():
    """Exact boundary: as_of + horizon_days + embargo_days == val_start is
    DROPPED (the purge condition is `>=`, not `>`); one day earlier survives.
    """
    base = date(2024, 1, 1)
    val_start = base + timedelta(days=180)
    on_boundary = _mk_train_row(base.isoformat())  # 0 + 90 + 90 == 180 -> drop
    one_day_earlier = _mk_train_row((base - timedelta(days=1)).isoformat())  # -1+180=179 < 180 -> keep

    kept, n_purged = optimize.purge_train(
        [on_boundary, one_day_earlier], val_start, horizon_days=90, embargo_days=90
    )
    assert n_purged == 1
    assert kept == [one_day_earlier]


def test_purge_train_zero_embargo_uses_horizon_only():
    """With embargo_days=0, only the label window itself (horizon_days)
    matters: a sample whose 90-day-forward window just touches val_start is
    dropped, one dated a single extra day earlier is kept.
    """
    base = date(2024, 1, 1)
    val_start = base + timedelta(days=90)
    touching = _mk_train_row(base.isoformat())  # 0 + 90 == 90 -> drop
    clears = _mk_train_row((base - timedelta(days=1)).isoformat())  # -1+90=89 < 90 -> keep

    kept, n_purged = optimize.purge_train([touching, clears], val_start, horizon_days=90, embargo_days=0)
    assert n_purged == 1
    assert kept == [clears]


def test_purge_train_unparseable_as_of_is_dropped():
    """Conservative choice: a training sample whose as_of cannot be parsed
    is dropped rather than risk (unverifiable) leakage into the validation
    fold — see purge_train's docstring.
    """
    val_start = date(2024, 6, 1)
    bad_rows = [
        _mk_train_row(None),
        _mk_train_row("not-a-date"),
        _mk_train_row(""),
    ]
    kept, n_purged = optimize.purge_train(bad_rows, val_start, horizon_days=90, embargo_days=90)
    assert kept == []
    assert n_purged == len(bad_rows)

    # A date object (not a str) still parses fine and is purge-evaluated normally.
    far_past = _mk_train_row(date(2020, 1, 1))
    kept2, n_purged2 = optimize.purge_train([far_past], val_start, horizon_days=90, embargo_days=90)
    assert kept2 == [far_past]
    assert n_purged2 == 0


# ---------------------------------------------------------------------------
# option_track_stats: expectancy/median/Sortino with cost haircut (change 2)
# ---------------------------------------------------------------------------


def test_option_track_stats_expectancy_with_cost_haircut():
    """Top quartile all winners, bottom quartile all losers; the cost
    haircut is subtracted from every option return before expectancy/median/
    hit-rate are computed.
    """
    # 8 pairs ranked by score 0..7: bottom quartile (score 0-1) loses, top
    # quartile (score 6-7) wins.
    scored = [(float(i), 0.5 if i >= 6 else (-0.5 if i < 2 else 0.1)) for i in range(8)]
    stats = optimize.option_track_stats(scored, cost_haircut=0.05)

    assert stats["n_top"] == 2 and stats["n_bottom"] == 2
    assert abs(stats["top_q_option_expectancy"] - 0.45) < 1e-9  # 0.5 - 0.05
    assert abs(stats["top_q_option_median"] - 0.45) < 1e-9
    assert stats["top_q_option_hit"] == 1.0
    assert abs(stats["bottom_q_option_hit"] - 0.0) < 1e-9  # -0.5-0.05 = -0.55, never > 0


def test_option_track_stats_too_few_pairs_returns_none():
    scored = [(1.0, 0.1), (2.0, 0.2), (3.0, None)]  # < 4 usable pairs
    stats = optimize.option_track_stats(scored, cost_haircut=0.05)
    assert stats == {
        "top_q_option_hit": None,
        "bottom_q_option_hit": None,
        "top_q_option_expectancy": None,
        "top_q_option_median": None,
        "top_q_option_sortino": None,
        "n_top": 0,
        "n_bottom": 0,
    }


def test_sortino_none_when_no_downside_or_too_few():
    # All returns >= 0 -> no downside deviation -> None, not a fabricated ratio.
    assert optimize._sortino([0.1, 0.2, 0.05, 0.3]) is None
    # A single return -> too few samples -> None.
    assert optimize._sortino([0.1]) is None
    # Mixed returns -> a real, finite ratio.
    ratio = optimize._sortino([0.2, -0.1, 0.1, -0.05])
    assert ratio is not None and isinstance(ratio, float)


def test_stock_track_stats_no_cost_haircut_applied():
    """Stock returns are never haircut (there is no round-trip option cost
    for simply holding the underlying in this research context)."""
    scored = [(float(i), 0.1 if i >= 6 else (-0.1 if i < 2 else 0.0)) for i in range(8)]
    stats = optimize.stock_track_stats(scored)
    assert abs(stats["top_q_stock_mean"] - 0.1) < 1e-9
    assert abs(stats["bottom_q_stock_mean"] - (-0.1)) < 1e-9
    assert abs(stats["stock_top_minus_bottom"] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Effective sample size + block bootstrap (change 4)
# ---------------------------------------------------------------------------


def test_effective_sample_size_shrinks_with_large_clusters():
    """Many samples sharing few as_of dates -> effective_n well below
    n_samples; one sample per as_of date -> effective_n == n_samples (the
    design effect vanishes)."""
    base = date(2024, 1, 1)
    clustered = [
        {"as_of": (base + timedelta(days=30 * (i // 20))).isoformat()} for i in range(100)
    ]  # 5 distinct dates, 20 samples each
    eff = optimize.effective_sample_size(clustered)
    assert eff["n_clusters"] == 5
    assert eff["avg_cluster_size"] == 20.0
    assert eff["effective_n"] < len(clustered)
    assert eff["effective_n"] < 20  # heavily deflated vs. the naive n=100

    unclustered = [{"as_of": (base + timedelta(days=i)).isoformat()} for i in range(100)]
    eff2 = optimize.effective_sample_size(unclustered)
    assert eff2["n_clusters"] == 100
    assert abs(eff2["effective_n"] - 100) < 1e-6


def test_block_bootstrap_ci_deterministic_under_fixed_seed():
    """Same samples + same seed -> identical CI across repeated calls (and
    across two independently-built but equal sample lists)."""
    base = date(2024, 1, 1)
    samples = [
        {"as_of": (base + timedelta(days=30 * (i // 5))).isoformat(), "score": float(i % 7)}
        for i in range(60)
    ]

    def stat(pooled):
        vals = [s["score"] for s in pooled]
        return sum(vals) / len(vals) if vals else None

    ci_a = optimize.block_bootstrap_ci(samples, stat, n_resamples=50, seed=123)
    ci_b = optimize.block_bootstrap_ci(samples, stat, n_resamples=50, seed=123)
    assert ci_a == ci_b
    assert ci_a is not None
    lo, hi = ci_a
    assert lo <= hi

    # A different seed is allowed to (and generally will) differ.
    ci_c = optimize.block_bootstrap_ci(samples, stat, n_resamples=50, seed=999)
    assert ci_c is not None


def test_block_bootstrap_ci_resamples_whole_dates_not_rows():
    """Every resample must draw whole as_of clusters -- never split a date's
    rows across in/out of a resample."""
    base = date(2024, 1, 1)
    samples = [
        {"as_of": (base + timedelta(days=30 * (i // 5))).isoformat(), "score": float(i)}
        for i in range(20)
    ]  # 4 distinct dates, 5 rows each

    seen_sizes = set()

    def stat(pooled):
        seen_sizes.add(len(pooled))
        return len(pooled)

    optimize.block_bootstrap_ci(samples, stat, n_resamples=30, seed=7)
    # Every resampled pool must be a multiple of the cluster size (5 rows).
    assert seen_sizes and all(size % 5 == 0 for size in seen_sizes)


def test_block_bootstrap_ci_none_when_no_clusters():
    assert optimize.block_bootstrap_ci([], lambda pooled: 1.0) is None
