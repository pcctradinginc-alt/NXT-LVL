"""Offline tests for the walk-forward out-of-sample backtest (src/backtest/walkforward.py).

No real network calls: reconstruct_digest is exercised via its `mock_data`
short-circuit, and forward_return / run_walkforward are exercised with an
injected `price_series` dict instead of a TradierClient.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from src.backtest import walkforward


# ---------------------------------------------------------------------------
# spearman (reused from calibrate.py) — quick sanity check on the import
# ---------------------------------------------------------------------------


def test_spearman():
    xs = [1, 2, 3, 4, 5]
    ys_monotonic = [10, 20, 30, 40, 50]
    ys_reversed = [50, 40, 30, 20, 10]
    ys_constant = [7, 7, 7, 7, 7]

    assert walkforward.spearman(xs, ys_monotonic) > 0.99
    assert walkforward.spearman(xs, ys_reversed) < -0.99
    assert walkforward.spearman(xs, ys_constant) is None


# ---------------------------------------------------------------------------
# reconstruct_digest
# ---------------------------------------------------------------------------


def test_reconstruct_digest_mock():
    mock_data = {
        "theme_a": {
            "edgar_fts": 4,
            "arxiv": 7,
            "hn_buzz": 12,
            "jobs": 3,
            "baseline": {"edgar_fts": 2, "arxiv": 5, "hn_buzz": 8, "jobs": 1},
        },
        "theme_b": {
            "edgar_fts": 0,
            "arxiv": 0,
            "hn_buzz": 0,
            "jobs": 0,
            "baseline": {"edgar_fts": 0, "arxiv": 0, "hn_buzz": 0, "jobs": 0},
        },
    }
    themes = [{"id": "theme_a", "keywords": ["foo"]}, {"id": "theme_b", "keywords": ["bar"]}]

    result = walkforward.reconstruct_digest(date(2024, 1, 1), themes, mock_data=mock_data)

    # No network call was made: mock_data is returned verbatim.
    assert result == mock_data
    assert set(result.keys()) == {"theme_a", "theme_b"}
    for theme_block in result.values():
        for source in ("edgar_fts", "arxiv", "hn_buzz", "jobs"):
            assert source in theme_block
        assert "baseline" in theme_block
        for source in ("edgar_fts", "arxiv", "hn_buzz", "jobs"):
            assert source in theme_block["baseline"]


# ---------------------------------------------------------------------------
# forward_return — no look-ahead
# ---------------------------------------------------------------------------


def test_forward_return_no_lookahead():
    base = date(2024, 1, 1)

    # Rising price -> positive forward return.
    price_series_up = {
        "AAA": {
            base.isoformat(): 100.0,
            (base + timedelta(days=1)).isoformat(): 101.0,
            (base + timedelta(days=30)).isoformat(): 110.0,
        }
    }
    ret_up = walkforward.forward_return("AAA", base, 30, None, price_series=price_series_up)
    assert ret_up is not None
    assert ret_up > 0

    # Falling price -> negative forward return.
    price_series_down = {
        "BBB": {
            base.isoformat(): 100.0,
            (base + timedelta(days=30)).isoformat(): 90.0,
        }
    }
    ret_down = walkforward.forward_return("BBB", base, 30, None, price_series=price_series_down)
    assert ret_down is not None
    assert ret_down < 0

    # No data anywhere near the target exit date -> None (no fabricated look-ahead-free guess).
    price_series_short = {
        "CCC": {
            base.isoformat(): 100.0,
            (base + timedelta(days=2)).isoformat(): 101.0,
        }
    }
    ret_none = walkforward.forward_return("CCC", base, 30, None, price_series=price_series_short)
    assert ret_none is None

    # Unknown ticker entirely -> None, never raises.
    ret_missing = walkforward.forward_return("ZZZ", base, 30, None, price_series=price_series_up)
    assert ret_missing is None


def test_forward_return_snaps_backward_only():
    # Exit target lands on a weekend/no-data day; nearest available close must
    # be the one BEFORE it (backward snap), not one that comes after.
    base = date(2024, 1, 1)  # Monday
    price_series = {
        "AAA": {
            base.isoformat(): 100.0,
            (base + timedelta(days=28)).isoformat(): 105.0,  # last close before target
            (base + timedelta(days=33)).isoformat(): 999.0,  # AFTER target; must not be used
        }
    }
    ret = walkforward.forward_return("AAA", base, 30, None, price_series=price_series, max_snap_days=7)
    assert ret is not None
    # Should reflect the 105.0 close (28 days out), not the 999.0 one after the target.
    assert ret == round((105.0 - 100.0) / 100.0 * 100.0, 4)


# ---------------------------------------------------------------------------
# run_walkforward — full offline run
# ---------------------------------------------------------------------------


def _build_price_series(tickers: list[str], start: date, end: date, seed: int) -> dict[str, dict[str, float]]:
    """Small deterministic synthetic GBM series for a handful of tickers, covering
    well beyond [start, end] so every as-of date has forward data at every horizon.
    """
    rng = random.Random(seed)
    dt = 1.0 / 252
    series_start = start - timedelta(days=15)
    n_days = (end - series_start).days + 250

    series: dict[str, dict[str, float]] = {}
    for i, ticker in enumerate(tickers):
        drift = 0.05 + 0.08 * i  # differentiate tickers so scores/returns aren't degenerate
        vol = 0.30
        price = 100.0
        bars: dict[str, float] = {}
        current = series_start
        for _ in range(n_days):
            if current.weekday() < 5:
                z = rng.gauss(0.0, 1.0)
                price *= math.exp((drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z)
                bars[current.isoformat()] = round(price, 4)
            current += timedelta(days=1)
        series[ticker] = bars
    return series


def _build_mock_digests(theme_ids: list[str], start: date, end: date) -> dict[str, dict]:
    """A digest entry for every calendar day in [start, end] so run_walkforward's
    as-of grid (whatever its exact cadence-derived dates are) always finds a match.
    """
    digests: dict[str, dict] = {}
    d = start
    while d <= end:
        day_index = (d - start).days
        digests[d.isoformat()] = {
            theme_ids[0]: {
                "edgar_fts": 2 + day_index // 10,
                "arxiv": 3 + day_index // 8,
                "hn_buzz": 5 + day_index // 5,
                "jobs": 1 + day_index // 15,
                "baseline": {"edgar_fts": 2, "arxiv": 3, "hn_buzz": 5, "jobs": 1},
            },
            theme_ids[1]: {
                "edgar_fts": 3,
                "arxiv": 3,
                "hn_buzz": 4,
                "jobs": 2,
                "baseline": {"edgar_fts": 3, "arxiv": 3, "hn_buzz": 4, "jobs": 2},
            },
        }
        d += timedelta(days=1)
    return digests


def test_run_walkforward_mock():
    tickers = ["TA", "TB"]
    benchmarks = ("SPY",)
    themes = [
        {"id": "th1", "keywords": ["mockkeyword1"], "tickers": ["TA"]},
        {"id": "th2", "keywords": ["mockkeyword2"], "tickers": ["TB"]},
    ]

    start = date(2023, 1, 2)
    end = date(2023, 6, 1)

    price_series = _build_price_series(list(tickers) + list(benchmarks), start, end, seed=7)
    mock_digests = _build_mock_digests(["th1", "th2"], start, end)

    report = walkforward.run_walkforward(
        None,
        tickers,
        themes,
        start=start,
        end=end,
        cadence_days=30,
        horizons=(30, 60, 90),
        benchmarks=benchmarks,
        price_series=price_series,
        mock_digests=mock_digests,
    )

    assert report["n_samples"] > 0
    assert report["params"]["mock"] is False  # mock flag wasn't set; mock_digests took priority
    assert report["params"]["tickers"] == tickers

    assert "ic_by_component" in report
    for h_key in ("30", "60", "90"):
        assert h_key in report["ic_by_component"]
        row = report["ic_by_component"][h_key]
        assert "n" in row
        for component in ("divergence", "theme_momentum", "breadth", "total"):
            assert component in row
            assert row[component] is None or -1.0 <= row[component] <= 1.0

    assert "buckets" in report
    for h_key, buckets in report["buckets"].items():
        assert set(buckets.keys()) == {"Q1", "Q2", "Q3", "Q4"}
        total_n = 0
        for label, b in buckets.items():
            assert "n" in b
            assert b["n"] >= 0
            if b["n"]:
                assert 0.0 <= b["hit_rate"] <= 1.0
            total_n += b["n"]
        # every valid (ticker, as_of) sample for this horizon lands in exactly one
        # quartile bucket, so the bucket counts must sum to the IC table's n.
        assert total_n == report["ic_by_component"][h_key]["n"]

    assert "lead_lag" in report
    for source in ("edgar_fts", "arxiv", "hn_buzz", "jobs"):
        assert source in report["lead_lag"]
        info = report["lead_lag"][source]
        assert "ic_by_horizon" in info
        assert "strongest_horizon_days" in info

    assert "temporal_ordering" in report
    assert "n_winners" in report["temporal_ordering"]
    assert "pct_leading_before_price_move" in report["temporal_ordering"]

    notes = report["notes"]
    assert isinstance(notes, list) and notes
    assert any("github" in n.lower() for n in notes)
    assert any("look-ahead" in n.lower() for n in notes)


def test_run_walkforward_mock_flag_generates_synthetic_digests():
    """When mock=True and no mock_digests are supplied at all, run_walkforward
    must still complete offline using its own deterministic synthetic digests.
    """
    tickers = ["WFA", "WFB"]
    start = date(2023, 1, 2)
    end = date(2023, 4, 1)
    price_series = _build_price_series(tickers + ["SPY"], start, end, seed=3)
    themes = [
        {"id": "t1", "keywords": ["kw1"], "tickers": ["WFA"]},
        {"id": "t2", "keywords": ["kw2"], "tickers": ["WFB"]},
    ]

    report = walkforward.run_walkforward(
        None,
        tickers,
        themes,
        start=start,
        end=end,
        cadence_days=30,
        horizons=(30, 60),
        benchmarks=("SPY",),
        mock=True,
        price_series=price_series,
    )

    assert report["n_samples"] > 0
    assert report["params"]["mock"] is True


def test_walkforward_includes_samples_when_requested():
    """CONCEPT_PROFIT.md Phase C: run_walkforward(..., include_samples=True)
    exposes report["samples"] (compact per-(ticker, as_of) records) so the
    optimizer can recalibrate WITHOUT re-fetching; each record carries the new
    momentum_12_1 component plus the trend_ok/regime_risk_on gate flags. Also
    checks that the report gained ic_by_component["*"]["momentum_12_1"] and a
    filtered_buckets block, and that samples stay OFF by default.
    """
    tickers = ["TA", "TB"]
    benchmarks = ("SPY",)
    themes = [
        {"id": "th1", "keywords": ["mockkeyword1"], "tickers": ["TA"]},
        {"id": "th2", "keywords": ["mockkeyword2"], "tickers": ["TB"]},
    ]
    start = date(2023, 1, 2)
    end = date(2023, 6, 1)
    price_series = _build_price_series(list(tickers) + list(benchmarks), start, end, seed=11)
    mock_digests = _build_mock_digests(["th1", "th2"], start, end)

    common = dict(
        start=start,
        end=end,
        cadence_days=30,
        horizons=(30, 60, 90),
        benchmarks=benchmarks,
        price_series=price_series,
        mock_digests=mock_digests,
    )

    # Off by default.
    report_default = walkforward.run_walkforward(None, tickers, themes, **common)
    assert "samples" not in report_default
    # momentum_12_1 is now an IC-measured component; filtered_buckets is present.
    assert "momentum_12_1" in report_default["ic_by_component"]["90"]
    assert "filtered_buckets" in report_default
    fb = report_default["filtered_buckets"]
    assert set(("horizon", "n_gated", "underlying", "option")) <= set(fb.keys())

    # On when requested.
    report = walkforward.run_walkforward(None, tickers, themes, include_samples=True, **common)
    assert "samples" in report
    samples = report["samples"]
    assert isinstance(samples, list) and len(samples) == report["n_samples"] > 0

    for s in samples:
        assert set(("as_of", "ticker", "components", "trend_ok", "regime_risk_on", "fwd", "opt")) <= set(s.keys())
        # the four weighted components, including the new momentum factor
        for comp in ("divergence", "theme_momentum", "breadth", "momentum_12_1"):
            assert comp in s["components"]
        # gate flags are True/False/None (never missing)
        assert s["trend_ok"] in (True, False, None)
        assert s["regime_risk_on"] in (True, False, None)
        # forward + option maps cover every requested horizon, values are numbers or None
        for h_key in ("30", "60", "90"):
            assert h_key in s["fwd"]
            assert s["fwd"][h_key] is None or isinstance(s["fwd"][h_key], (int, float))
            assert h_key in s["opt"]
            assert s["opt"][h_key] is None or isinstance(s["opt"][h_key], (int, float))


def test_run_walkforward_no_data_returns_empty_but_well_formed():
    """No tradier, no price_series, no mock_digests, mock=False: every price
    lookup degrades to empty/None gracefully rather than raising, and the
    report structure remains well-formed with n_samples == 0.
    """
    themes = [{"id": "t1", "keywords": ["kw"], "tickers": ["XYZ"]}]
    report = walkforward.run_walkforward(
        None,
        ["XYZ"],
        themes,
        start=date(2023, 1, 1),
        end=date(2023, 1, 1),
        cadence_days=30,
        horizons=(30,),
        benchmarks=("SPY",),
        mock_digests={"2023-01-01": {"t1": {"edgar_fts": 0, "arxiv": 0, "hn_buzz": 0, "jobs": 0, "baseline": {}}}},
    )
    assert report["n_samples"] == 0
    assert report["buckets"]["30"]["Q1"]["n"] == 0
    assert report["ic_by_component"]["30"]["total"] is None


# ---------------------------------------------------------------------------
# Phase A: monthly bucketing + cache + option P/L
# ---------------------------------------------------------------------------

def test_month_range():
    months = walkforward._month_range(date(2024, 11, 15), date(2025, 2, 3))
    assert months == ["2024-11", "2024-12", "2025-01", "2025-02"]
    # Reversed bounds are tolerated (swapped).
    assert walkforward._month_range(date(2025, 2, 3), date(2024, 11, 15)) == months
    # Single month.
    assert walkforward._month_range(date(2025, 5, 2), date(2025, 5, 28)) == ["2025-05"]


def test_bucketed_counts_uses_cache():
    calls = {"n": 0}

    def fake_source(themes, start, end):
        calls["n"] += 1
        return {themes[0]["id"]: 7}

    theme = {"id": "ai_inference"}
    cache: dict[str, int] = {}
    months = ["2024-11", "2024-12"]  # completed past months

    out1 = walkforward._bucketed_counts(fake_source, cache, "arxiv", theme, months)
    assert out1 == {"2024-11": 7, "2024-12": 7}
    assert calls["n"] == 2

    # Same completed months again -> served from cache, no new calls.
    out2 = walkforward._bucketed_counts(fake_source, cache, "arxiv", theme, months)
    assert out2 == out1
    assert calls["n"] == 2

    # The current (incomplete) month is never cached -> re-queried every time.
    cur = date.today().strftime("%Y-%m")
    walkforward._bucketed_counts(fake_source, cache, "arxiv", theme, [cur])
    n1 = calls["n"]
    walkforward._bucketed_counts(fake_source, cache, "arxiv", theme, [cur])
    assert calls["n"] == n1 + 1


def test_bucketed_counts_degraded_excluded():
    def none_source(themes, start, end):
        return {}  # no value for the theme -> degraded, excluded (not a false 0)

    degraded = [0]
    out = walkforward._bucketed_counts(
        none_source, {}, "arxiv", {"id": "ai_inference"}, ["2024-11", "2024-12"], degraded=degraded
    )
    assert out == {}
    assert degraded[0] == 2


def _synthetic_series(start_price: float, daily_ret: float, n: int) -> dict[str, float]:
    series: dict[str, float] = {}
    p = start_price
    cur = date(2025, 1, 1)
    for _ in range(n):
        series[cur.isoformat()] = round(p, 4)
        p *= (1.0 + daily_ret)
        cur = cur + timedelta(days=1)
    return series


def test_option_pl_direction():
    # ~63 days of history before as_of (for realized vol) + >90 days forward.
    as_of = date(2025, 1, 1) + timedelta(days=70)
    rising = {"RISE": _synthetic_series(100.0, 0.003, 220)}
    falling = {"FALL": _synthetic_series(100.0, -0.003, 220)}

    up = walkforward._compute_option_metrics("RISE", as_of, None, rising, (90,))
    down = walkforward._compute_option_metrics("FALL", as_of, None, falling, (90,))

    assert up.get("90") is not None
    assert up["90"]["return"] > 0 and up["90"]["hit"] is True
    assert down.get("90") is not None
    assert down["90"]["return"] < 0 and down["90"]["hit"] is False
