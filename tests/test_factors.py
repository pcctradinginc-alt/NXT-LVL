"""Offline unit tests for src.analysis.factors (candidate price/fundamental
factor scores for the walk-forward calibration).

All tests use constructed price series and plain assertions -- no network,
no fixtures beyond simple Python lists.
"""

from __future__ import annotations

from src.analysis import factors


def _flat_series(n: int, price: float = 100.0) -> list[float]:
    return [price] * n


# ---------------------------------------------------------------------------
# reversal_1m_score
# ---------------------------------------------------------------------------


class TestReversal1m:
    def test_recent_loser_scores_higher_than_recent_winner(self):
        # Flat history, then a sharp drop in the last 21 trading days -> loser.
        loser = _flat_series(230, 100.0)
        loser += [100.0 * (1.0 - 0.30 * i / 21) for i in range(1, 22)]  # ~-30% over 21d

        # Flat history, then a sharp rise in the last 21 trading days -> winner.
        winner = _flat_series(230, 100.0)
        winner += [100.0 * (1.0 + 0.30 * i / 21) for i in range(1, 22)]  # ~+30% over 21d

        loser_score = factors.reversal_1m_score(loser)
        winner_score = factors.reversal_1m_score(winner)

        assert loser_score > winner_score
        assert loser_score == 85.0  # r <= -15
        assert winner_score == 25.0  # r > 15

    def test_short_series_is_neutral(self):
        assert factors.reversal_1m_score([100.0] * 10) == 50.0
        assert factors.reversal_1m_score([]) == 50.0
        assert factors.reversal_1m_score(None) == 50.0

    def test_flat_series_is_neutral_bucket(self):
        flat = _flat_series(30, 100.0)
        assert factors.reversal_1m_score(flat) == 55.0


# ---------------------------------------------------------------------------
# low_vol_score
# ---------------------------------------------------------------------------


class TestLowVol:
    def test_smooth_series_scores_higher_than_jagged_series(self):
        smooth = [100.0 + 0.01 * i for i in range(70)]  # near-zero daily vol

        jagged = []
        price = 100.0
        for i in range(70):
            price *= 1.08 if i % 2 == 0 else (1.0 / 1.08)
            jagged.append(price)

        smooth_score = factors.low_vol_score(smooth)
        jagged_score = factors.low_vol_score(jagged)

        assert smooth_score > jagged_score
        assert smooth_score == 80.0  # rv <= 0.25
        assert jagged_score == 20.0  # very high rv

    def test_insufficient_data_is_neutral(self):
        assert factors.low_vol_score([100.0] * 30) == 50.0
        assert factors.low_vol_score([]) == 50.0
        assert factors.low_vol_score(None) == 50.0


# ---------------------------------------------------------------------------
# high_52w_score
# ---------------------------------------------------------------------------


class TestHigh52w:
    def test_near_high_scores_higher_than_near_low(self):
        # Ends right at its own trailing high.
        near_high = [100.0 + i for i in range(260)]
        # Ends well below its trailing high (fell from a peak).
        near_low = [100.0 + i for i in range(200)] + [300.0 - i for i in range(60)]

        high_score = factors.high_52w_score(near_high)
        low_score = factors.high_52w_score(near_low)

        assert high_score > low_score
        assert high_score == 85.0  # ratio >= 0.95

    def test_short_series_is_neutral(self):
        assert factors.high_52w_score([100.0] * 10) == 50.0
        assert factors.high_52w_score([]) == 50.0
        assert factors.high_52w_score(None) == 50.0


# ---------------------------------------------------------------------------
# rs_score
# ---------------------------------------------------------------------------


class TestRelativeStrength:
    def test_ticker_beating_benchmark_scores_higher_than_lagging(self):
        benchmark = [100.0 + 0.2 * i for i in range(70)]  # modest steady climb

        beats = [100.0 + 1.0 * i for i in range(70)]  # much stronger climb
        lags = [100.0 - 0.3 * i for i in range(70)]  # declining

        beats_score = factors.rs_score(beats, benchmark)
        lags_score = factors.rs_score(lags, benchmark)

        assert beats_score > lags_score
        assert beats_score == 85.0
        assert lags_score == 25.0

    def test_insufficient_ticker_or_benchmark_is_neutral(self):
        long_series = [100.0 + i for i in range(70)]
        short_series = [100.0] * 10

        assert factors.rs_score(short_series, long_series) == 50.0
        assert factors.rs_score(long_series, short_series) == 50.0
        assert factors.rs_score(None, long_series) == 50.0
        assert factors.rs_score(long_series, None) == 50.0


# ---------------------------------------------------------------------------
# revenue_growth_score
# ---------------------------------------------------------------------------


class TestRevenueGrowth:
    def test_positive_growth_scores_higher_than_negative(self):
        assert factors.revenue_growth_score(50.0) > factors.revenue_growth_score(-10.0)
        assert factors.revenue_growth_score(50.0) == 85.0
        assert factors.revenue_growth_score(-10.0) == 25.0

    def test_buckets(self):
        assert factors.revenue_growth_score(45.0) == 85.0
        assert factors.revenue_growth_score(25.0) == 70.0
        assert factors.revenue_growth_score(10.0) == 55.0
        assert factors.revenue_growth_score(0.0) == 45.0
        assert factors.revenue_growth_score(-1.0) == 25.0

    def test_none_is_neutral(self):
        assert factors.revenue_growth_score(None) == 50.0

    def test_non_numeric_is_neutral(self):
        assert factors.revenue_growth_score("not-a-number") == 50.0
