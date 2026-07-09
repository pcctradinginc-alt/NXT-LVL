"""Offline tests for the backtest/validation tools (src/backtest/*).

No real network calls: price_backtest uses injected `price_series`, and
calibrate uses injected `forward_returns` / fake history files.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.analysis import options_math
from src.backtest import calibrate, price_backtest
from src.main import append_digest_history


# ---------------------------------------------------------------------------
# options_math: delta + strike solver
# ---------------------------------------------------------------------------

def test_bs_call_delta_monotonic():
    # Delta rises as S rises (deeper ITM), for a fixed strike.
    d_low_s = options_math.bs_call_delta(S=90.0, K=100.0, T_years=0.5, r=0.04, sigma=0.4)
    d_high_s = options_math.bs_call_delta(S=120.0, K=100.0, T_years=0.5, r=0.04, sigma=0.4)
    assert d_high_s > d_low_s

    # Delta falls as K rises, for a fixed S.
    d_low_k = options_math.bs_call_delta(S=100.0, K=80.0, T_years=0.5, r=0.04, sigma=0.4)
    d_high_k = options_math.bs_call_delta(S=100.0, K=140.0, T_years=0.5, r=0.04, sigma=0.4)
    assert d_low_k > d_high_k

    # ITM delta > 0.5 > OTM delta.
    itm = options_math.bs_call_delta(S=120.0, K=100.0, T_years=0.5, r=0.04, sigma=0.4)
    otm = options_math.bs_call_delta(S=80.0, K=100.0, T_years=0.5, r=0.04, sigma=0.4)
    assert itm > 0.5 > otm


def test_bs_call_delta_degenerate_cases():
    assert options_math.bs_call_delta(S=110.0, K=100.0, T_years=0.0, r=0.04, sigma=0.4) == 1.0
    assert options_math.bs_call_delta(S=90.0, K=100.0, T_years=0.0, r=0.04, sigma=0.4) == 0.0
    assert options_math.bs_call_delta(S=110.0, K=100.0, T_years=0.5, r=0.04, sigma=0.0) == 1.0
    assert options_math.bs_call_delta(S=90.0, K=100.0, T_years=0.5, r=0.04, sigma=0.0) == 0.0


def test_solve_strike_for_delta():
    S = 100.0
    T = 120 / 365.0
    r = 0.04
    sigma = 0.45
    for target in (0.3, 0.5, 0.6, 0.7, 0.85):
        strike = options_math.solve_strike_for_delta(S, target, T, r, sigma)
        achieved_delta = options_math.bs_call_delta(S, strike, T, r, sigma)
        assert achieved_delta == pytest.approx(target, abs=0.02)


# ---------------------------------------------------------------------------
# price_backtest: mock run
# ---------------------------------------------------------------------------

def _build_synthetic_series(seed_prices: dict[str, float], n_days: int = 420) -> dict[str, dict[str, float]]:
    """Build a small deterministic multi-ticker daily price series.

    Ticker A trends up mildly (small trailing moves -> favorable divergence
    bucket), ticker B ramps hard partway through (large trailing moves at
    some entry points -> unfavorable divergence bucket), ticker C is roughly
    flat/choppy. Purely deterministic (no RNG) so the test is reproducible
    without depending on price_backtest's own mock generator.
    """
    start = date.today() - timedelta(days=n_days + 30)
    series: dict[str, dict[str, float]] = {}
    for ticker, base in seed_prices.items():
        bars: dict[str, float] = {}
        price = base
        for i in range(n_days):
            d = start + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            if ticker == "SYN_UP":
                price *= 1.0006  # slow steady drift
            elif ticker == "SYN_RUN":
                # Flat for the first 2/3, then a sharp run-up.
                price *= 1.0025 if i > n_days * 0.66 else 1.0001
            else:  # SYN_FLAT
                price *= 1.0001 if i % 2 == 0 else 0.9999
            bars[d.isoformat()] = round(price, 4)
        series[ticker] = bars
    return series


def test_price_backtest_mock_runs():
    price_series = _build_synthetic_series({"SYN_UP": 100.0, "SYN_RUN": 50.0, "SYN_FLAT": 200.0})

    report = price_backtest.run_backtest(
        None,
        ["SYN_UP", "SYN_RUN", "SYN_FLAT"],
        years=1,
        horizon_days=60,
        entry_dte=90,
        target_delta=0.60,
        cadence_days=20,
        price_series=price_series,
    )

    assert report["n_samples"] > 0
    assert "params" in report
    assert "overall" in report
    assert "by_divergence_bucket" in report

    overall = report["overall"]
    assert overall["n"] == report["n_samples"]
    assert 0.0 <= overall["hit_rate"] <= 1.0

    for bucket, stats in report["by_divergence_bucket"].items():
        assert stats["n"] >= 0
        if stats["n"]:
            assert 0.0 <= stats["hit_rate"] <= 1.0


def test_price_backtest_skips_tickers_with_too_little_data():
    price_series = {"THIN": {date.today().isoformat(): 100.0}}
    report = price_backtest.run_backtest(
        None, ["THIN"], years=1, horizon_days=60, price_series=price_series
    )
    assert report["n_samples"] == 0
    assert "THIN" in report["params"]["tickers_skipped"]


# ---------------------------------------------------------------------------
# calibrate: spearman
# ---------------------------------------------------------------------------

def test_spearman_known_values():
    xs = [1, 2, 3, 4, 5]
    ys_monotonic = [10, 20, 30, 40, 50]
    ys_reversed = [50, 40, 30, 20, 10]
    ys_constant = [7, 7, 7, 7, 7]

    assert calibrate.spearman(xs, ys_monotonic) == pytest.approx(1.0)
    assert calibrate.spearman(xs, ys_reversed) == pytest.approx(-1.0)
    assert calibrate.spearman(xs, ys_constant) is None
    assert calibrate.spearman([1, 2], [1, 2]) is None  # n < 3


def test_spearman_handles_ties():
    xs = [1, 1, 2, 3]
    ys = [1, 2, 3, 3]
    value = calibrate.spearman(xs, ys)
    assert value is not None
    assert -1.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# calibrate: run_calibration
# ---------------------------------------------------------------------------

def _write_history_line(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record))
        fh.write("\n")


def test_calibrate_insufficient_history(tmp_path: Path):
    history_path = tmp_path / "digest_history.jsonl"
    old_date = (date.today() - timedelta(days=200)).isoformat()
    _write_history_line(
        history_path,
        {
            "date": old_date,
            "current_stage": 2,
            "next_stage": 3,
            "top_pick": "AAA",
            "candidates": [
                {"ticker": "AAA", "total_score": 80.0, "scores": {"breadth": 70, "momentum": 60}, "source_count": 3}
            ],
            "all_theme_scores": {},
            "emergent_themes": [],
        },
    )

    class NoForwardTradier:
        def get_history(self, symbol, start, interval="daily"):
            return []

    report = calibrate.run_calibration(NoForwardTradier(), history_path, horizon_days=90)
    assert report["status"] == "insufficient_history"
    assert report["n"] < report["needed"]


def test_calibrate_computes_ic(tmp_path: Path):
    history_path = tmp_path / "digest_history.jsonl"
    old_date = (date.today() - timedelta(days=200)).isoformat()

    forward_returns = {}
    n_candidates = 32
    for i in range(n_candidates):
        ticker = f"T{i}"
        # Construct momentum score to correlate perfectly (rank-wise) with
        # forward return; other components are noise/constant.
        momentum = float(i)  # 0..31, monotonically increasing
        forward_ret = float(i) * 0.01  # also monotonically increasing -> IC should be ~1.0
        candidate = {
            "ticker": ticker,
            "total_score": 50.0,
            "scores": {
                "breadth": 50.0,
                "momentum": momentum,
                "stage_fit": 50.0,
                "divergence": 50.0,
                "option_quality": 50.0,
                "emergence": 50.0,
            },
            "source_count": 2,
        }
        _write_history_line(
            history_path,
            {
                "date": old_date,
                "current_stage": 2,
                "next_stage": 3,
                "top_pick": ticker if i == 0 else None,
                "candidates": [candidate],
                "all_theme_scores": {},
                "emergent_themes": [],
            },
        )
        forward_returns[(ticker, old_date)] = forward_ret

    report = calibrate.run_calibration(
        None, history_path, horizon_days=90, forward_returns=forward_returns
    )

    assert report["status"] == "ok"
    assert report["n"] >= 30
    assert report["ic"]["momentum"] is not None
    assert report["ic"]["momentum"] > 0.95  # perfectly monotonic -> IC close to 1.0
    # Constant components have zero variance -> undefined IC (None), never a
    # fabricated 0.0 or false-precision value.
    assert report["ic"]["breadth"] is None


def test_calibrate_respects_horizon_not_yet_elapsed(tmp_path: Path):
    history_path = tmp_path / "digest_history.jsonl"
    recent_date = (date.today() - timedelta(days=5)).isoformat()  # horizon (90d) not elapsed yet
    _write_history_line(
        history_path,
        {
            "date": recent_date,
            "current_stage": 2,
            "next_stage": 3,
            "top_pick": "AAA",
            "candidates": [
                {"ticker": "AAA", "total_score": 80.0, "scores": {"breadth": 70}, "source_count": 3}
            ],
            "all_theme_scores": {},
            "emergent_themes": [],
        },
    )
    report = calibrate.run_calibration(
        None, history_path, horizon_days=90, forward_returns={("AAA", recent_date): 0.05}
    )
    assert report["status"] == "insufficient_history"
    assert report["n"] == 0


# ---------------------------------------------------------------------------
# append_digest_history
# ---------------------------------------------------------------------------

def test_append_digest_history(tmp_path: Path):
    path = tmp_path / "digest_history.jsonl"
    top_pick = {
        "ticker": "VRT",
        "total_score": 82.5,
        "scores": {
            "breadth": 80.0,
            "momentum": 70.0,
            "stage_fit": 100.0,
            "divergence": 70.0,
            "option_quality": 60.0,
            "emergence": 55.0,
        },
        "source_count": 3,
    }
    scored_candidates = [top_pick]

    append_digest_history(
        path,
        current_stage=2,
        next_stage=3,
        top_pick=top_pick,
        scored_candidates=scored_candidates,
        all_theme_scores={"dc_cooling": 78.0},
        emergent_themes=[{"theme_id": "dc_cooling", "name": "Data Center Cooling"}],
    )

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["current_stage"] == 2
    assert record["next_stage"] == 3
    assert record["top_pick"] == "VRT"
    assert record["candidates"][0]["ticker"] == "VRT"
    assert record["candidates"][0]["scores"]["momentum"] == 70.0
    assert record["all_theme_scores"] == {"dc_cooling": 78.0}
    assert record["emergent_themes"] == ["dc_cooling"]
    assert "date" in record


def test_append_digest_history_handles_no_top_pick(tmp_path: Path):
    path = tmp_path / "digest_history.jsonl"
    append_digest_history(
        path,
        current_stage=1,
        next_stage=None,
        top_pick=None,
        scored_candidates=[],
        all_theme_scores=None,
        emergent_themes=None,
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["top_pick"] is None
    assert record["candidates"] == []


def test_append_digest_history_never_raises(tmp_path: Path):
    # Passing a path whose parent cannot be created (e.g. pointing through a
    # file) must not raise — the helper is fault-tolerant by design.
    bad_parent = tmp_path / "not_a_dir"
    bad_parent.write_text("i am a file, not a directory")
    bad_path = bad_parent / "digest_history.jsonl"
    # Should not raise.
    append_digest_history(
        bad_path,
        current_stage=None,
        next_stage=None,
        top_pick=None,
        scored_candidates=[],
        all_theme_scores=None,
        emergent_themes=None,
    )
