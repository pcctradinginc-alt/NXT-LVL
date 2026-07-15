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
    """
    base = date(2024, 1, 1)
    samples = []
    for j in range(40):
        m = (j % 10) * 10.0  # 0,10,...,90 — a full momentum spread inside each fold of 10
        samples.append(
            _mk_sample(
                (base + timedelta(days=j)).isoformat(),
                "T",
                {"momentum_12_1": m},
                fwd90=m,  # forward return rank-tracks momentum -> IC +1
                opt90=0.5 if m >= 50 else -0.5,  # high momentum -> option winner
            )
        )

    calib = optimize.run_calibration(samples, n_folds=4, horizon="90")

    assert calib["validation"]["passed"] is True
    assert calib["validation"]["median_val_ic_total"] > 0.0
    assert calib["validation"]["median_top_q_option_hit"] >= 0.55
    assert calib["validation"]["top_beats_bottom_fraction"] >= 0.5

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
    for j in range(40):
        lv = (j % 10) * 10.0  # 0,10,...,90 — a full spread inside each fold of 10
        noise = rng.uniform(0, 100)  # unrelated to forward return
        samples.append(
            _mk_sample(
                (base + timedelta(days=j)).isoformat(),
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
    """The returned dict has the documented Phase C contract."""
    base = date(2024, 1, 1)
    samples = [
        _mk_sample((base + timedelta(days=j)).isoformat(), "T", {"momentum_12_1": float(j % 10 * 10)}, float(j), 0.1)
        for j in range(20)
    ]
    calib = optimize.run_calibration(samples, n_folds=4, horizon="90")

    assert set(calib["validation"].keys()) == {
        "median_val_ic_total",
        "median_top_q_option_hit",
        "top_beats_bottom_fraction",
        "passed",
    }
    assert "weights_final" in calib
    assert "folds" in calib and isinstance(calib["folds"], list)
    assert calib["n_samples"] == 20
    assert "generated_at" in calib
    assert isinstance(calib["notes"], list) and calib["notes"]
    for f in calib["folds"]:
        assert set(f.keys()) >= {
            "fold_index",
            "n_train",
            "n_val",
            "train_ics",
            "weights",
            "val_ic_total",
            "top_q_option_hit",
            "bottom_q_option_hit",
        }
