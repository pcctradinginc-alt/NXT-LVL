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
