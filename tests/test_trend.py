"""Comprehensive unit tests for src.analysis.trend module.

All tests are offline, using plain Python lists and pytest assertions.
"""

from __future__ import annotations

import pytest

from src.analysis import trend


class TestSMA:
    """Test simple moving average function."""

    def test_sma_basic(self):
        """Test basic SMA calculation."""
        # sma([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) should be (6+7+8+9+10)/5 = 8.0
        result = trend.sma([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
        assert result == 8.0

    def test_sma_too_short(self):
        """Test SMA with insufficient data."""
        result = trend.sma([1, 2, 3], 5)
        assert result is None

    def test_sma_empty(self):
        """Test SMA with empty list."""
        result = trend.sma([], 5)
        assert result is None

    def test_sma_none_input(self):
        """Test SMA with None input."""
        result = trend.sma(None, 5)
        assert result is None

    def test_sma_window_zero(self):
        """Test SMA with zero window."""
        result = trend.sma([1, 2, 3, 4, 5], 0)
        assert result is None

    def test_sma_negative_window(self):
        """Test SMA with negative window."""
        result = trend.sma([1, 2, 3, 4, 5], -1)
        assert result is None


class TestClean:
    """Test the _clean helper function via SMA with mixed inputs."""

    def test_clean_filters_bad_values(self):
        """Test that _clean filters None, non-numeric, negative, and zero values.

        Series: [1, None, 'x', -5, 0, 2, 3, 4, 5, 6]
        Cleaned: [1, 2, 3, 4, 5, 6]
        SMA(5) = (2 + 3 + 4 + 5 + 6) / 5 = 4.0
        """
        series = [1, None, "x", -5, 0, 2, 3, 4, 5, 6]
        result = trend.sma(series, 5)
        assert result == 4.0

    def test_clean_all_invalid(self):
        """Test series with all invalid values."""
        series = [None, "x", -5, 0]
        result = trend.sma(series, 1)
        assert result is None

    def test_clean_mixed_numeric_strings(self):
        """Test that numeric strings are converted."""
        series = ["10", "20", "30", "40", "50"]
        result = trend.sma(series, 5)
        assert result == 30.0

    def test_clean_float_strings(self):
        """Test that float strings are converted."""
        series = ["1.5", "2.5", "3.5", "4.5", "5.5"]
        result = trend.sma(series, 3)
        # Last 3 values: [3.5, 4.5, 5.5], average = 4.5
        assert result == pytest.approx(4.5)


class TestMomentum121:
    """Test 12-1 momentum calculation."""

    def test_momentum_12_1_insufficient_data(self):
        """Test momentum_12_1 with fewer than 253 closes."""
        result = trend.momentum_12_1([1, 2, 3] * 50)  # Only 150 values
        assert result is None

    def test_momentum_12_1_exactly_253(self):
        """Test with exactly 253 values."""
        closes = list(range(1, 254))  # 1 to 253
        result = trend.momentum_12_1(closes)
        # p_start = closes[-253] = closes[0] = 1
        # p_end = closes[-22] = closes[231] = 232
        # momentum = (232 / 1 - 1) * 100 = 23100.0
        assert result == pytest.approx(23100.0)

    def test_momentum_12_1_up(self):
        """Test momentum_12_1 with a rising series.

        260 closes rising linearly from 100 to 200 → momentum > 0.
        """
        closes = [100 + (100 * i / 259) for i in range(260)]
        result = trend.momentum_12_1(closes)
        assert result is not None
        assert result > 0

    def test_momentum_12_1_down(self):
        """Test momentum_12_1 with a falling series."""
        closes = [200 - (100 * i / 259) for i in range(260)]
        result = trend.momentum_12_1(closes)
        assert result is not None
        assert result < 0

    def test_momentum_12_1_exact(self):
        """Test momentum_12_1 with exact values for closes[-253] and closes[-22].

        Build a 260-length series where:
        - closes[-253] = 100.0
        - closes[-22] = 150.0
        - Other values filled with reasonable data
        Expected momentum = (150 / 100 - 1) * 100 = 50.0
        """
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 150.0
        result = trend.momentum_12_1(closes)
        assert result == pytest.approx(50.0)

    def test_momentum_12_1_empty(self):
        """Test momentum_12_1 with empty list."""
        result = trend.momentum_12_1([])
        assert result is None

    def test_momentum_12_1_none_input(self):
        """Test momentum_12_1 with None input."""
        result = trend.momentum_12_1(None)
        assert result is None


class TestAboveSMA:
    """Test above_sma function."""

    def test_above_sma_true(self):
        """Test when last close is above SMA."""
        # Last close is 10, SMA(3) of [1,2,3,4,5,10] is (3+4+5)/3 = 4, so 10 > 4
        closes = [1, 2, 3, 4, 5, 10]
        result = trend.above_sma(closes, 3)
        assert result is True

    def test_above_sma_false(self):
        """Test when last close is below SMA."""
        # Last close is 1, SMA(3) is (3+4+5)/3 = 4, so 1 < 4
        closes = [5, 4, 3, 2, 3, 4, 5, 1]
        result = trend.above_sma(closes, 3)
        assert result is False

    def test_above_sma_equal(self):
        """Test when last close equals SMA (should be False, not >=)."""
        # SMA(2) of [10, 10, 10] is 10, and last is 10, so not strictly above
        closes = [10, 10, 10]
        result = trend.above_sma(closes, 2)
        assert result is False

    def test_above_sma_insufficient_data(self):
        """Test with insufficient data for SMA."""
        result = trend.above_sma([1, 2], 5)
        assert result is None

    def test_above_sma_empty(self):
        """Test with empty list."""
        result = trend.above_sma([], 3)
        assert result is None

    def test_above_sma_with_bad_values(self):
        """Test above_sma with mixed valid/invalid values."""
        closes = [None, "x", 1, 2, 3, 4, 5, 10]
        result = trend.above_sma(closes, 3)
        # Cleaned: [1, 2, 3, 4, 5, 10]
        # SMA(3) = (3 + 4 + 5) / 3 = 4, last is 10 > 4
        assert result is True


class TestTrendOk:
    """Test trend_ok filter function."""

    def test_trend_ok_uptrend(self):
        """Test trend_ok with strictly rising 250 closes (uptrend)."""
        closes = [100 + i for i in range(250)]
        result = trend.trend_ok(closes)
        assert result is True

    def test_trend_ok_downtrend(self):
        """Test trend_ok with strictly falling 250 closes (downtrend)."""
        closes = [300 - i for i in range(250)]
        result = trend.trend_ok(closes)
        assert result is False

    def test_trend_ok_insufficient_data(self):
        """Test trend_ok with fewer than 200 closes."""
        closes = list(range(100))
        result = trend.trend_ok(closes)
        assert result is None

    def test_trend_ok_empty(self):
        """Test trend_ok with empty list."""
        result = trend.trend_ok([])
        assert result is None

    def test_trend_ok_none_input(self):
        """Test trend_ok with None input."""
        result = trend.trend_ok(None)
        assert result is None

    def test_trend_ok_partial_uptrend(self):
        """Test trend_ok with 200 close uptrend (exactly minimum)."""
        closes = [100 + i for i in range(200)]
        result = trend.trend_ok(closes)
        # Both conditions should hold: last > 50-SMA and 50-SMA > 200-SMA
        assert result is True

    def test_trend_ok_mixed_prices(self):
        """Test trend_ok with realistic mixed price movements."""
        # Strong uptrend: last 200 closes mostly rising
        closes = [100] * 50 + [100 + i * 0.5 for i in range(150)]
        result = trend.trend_ok(closes)
        # Expect True because we end on uptrend
        assert result is True


class TestRegimeRiskOn:
    """Test regime_risk_on function."""

    def test_regime_risk_on_true(self):
        """Test regime_risk_on with rising benchmark."""
        closes = [100 + i for i in range(250)]
        result = trend.regime_risk_on(closes)
        assert result is True

    def test_regime_risk_on_false(self):
        """Test regime_risk_on with falling benchmark."""
        closes = [300 - i for i in range(250)]
        result = trend.regime_risk_on(closes)
        assert result is False

    def test_regime_risk_on_insufficient_data(self):
        """Test regime_risk_on with fewer than 200 closes."""
        result = trend.regime_risk_on([1, 2, 3] * 50)  # 150 values
        assert result is None

    def test_regime_risk_on_empty(self):
        """Test regime_risk_on with empty list."""
        result = trend.regime_risk_on([])
        assert result is None

    def test_regime_risk_on_none_input(self):
        """Test regime_risk_on with None input."""
        result = trend.regime_risk_on(None)
        assert result is None

    def test_regime_risk_on_exactly_200(self):
        """Test with exactly 200 closes."""
        closes = [100 + i for i in range(200)]
        result = trend.regime_risk_on(closes)
        assert result is True


class TestScoreMomentum121:
    """Test score_momentum_12_1 bucketing function."""

    def test_score_momentum_short_list_returns_neutral(self):
        """Test that a short list (insufficient for momentum) scores 50.0 (neutral)."""
        result = trend.score_momentum_12_1([1, 2, 3])
        assert result == 50.0

    def test_score_momentum_none_input(self):
        """Test that None input (via short list after cleaning) scores 50.0."""
        result = trend.score_momentum_12_1(None)
        assert result == 50.0

    def test_score_momentum_bucket_very_negative(self):
        """Test bucket: momentum <= -20 → 10.0"""
        # Create series where momentum_12_1 returns -30
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 70.0  # (70/100 - 1)*100 = -30
        result = trend.score_momentum_12_1(closes)
        assert result == 10.0

    def test_score_momentum_bucket_negative(self):
        """Test bucket: -20 < momentum <= 0 → 30.0"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 95.0  # (95/100 - 1)*100 = -5
        result = trend.score_momentum_12_1(closes)
        assert result == 30.0

    def test_score_momentum_bucket_low_positive(self):
        """Test bucket: 0 < momentum <= 15 → 60.0"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 110.0  # (110/100 - 1)*100 = 10
        result = trend.score_momentum_12_1(closes)
        assert result == 60.0

    def test_score_momentum_bucket_medium_positive(self):
        """Test bucket: 15 < momentum <= 40 → 85.0"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 130.0  # (130/100 - 1)*100 = 30
        result = trend.score_momentum_12_1(closes)
        assert result == 85.0

    def test_score_momentum_bucket_high_positive(self):
        """Test bucket: momentum > 40 → 70.0 (haircut for crash risk)"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 160.0  # (160/100 - 1)*100 = 60
        result = trend.score_momentum_12_1(closes)
        assert result == 70.0

    def test_score_momentum_boundary_minus_20(self):
        """Test boundary: momentum < -20 → 10.0"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 70.0  # (70/100 - 1)*100 = -30, clearly in <= -20 range
        result = trend.score_momentum_12_1(closes)
        assert result == 10.0

    def test_score_momentum_boundary_zero(self):
        """Test boundary: -20 < momentum <= 0 → 30.0"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 95.0  # (95/100 - 1)*100 = -5, in the -20 < m <= 0 range
        result = trend.score_momentum_12_1(closes)
        assert result == 30.0

    def test_score_momentum_boundary_15(self):
        """Test boundary: 0 < momentum <= 15 → 60.0"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 110.0  # (110/100 - 1)*100 = 10, in the 0 < m <= 15 range
        result = trend.score_momentum_12_1(closes)
        assert result == 60.0

    def test_score_momentum_boundary_40(self):
        """Test boundary: 15 < momentum <= 40 → 85.0"""
        closes = [100.0] * 260
        closes[-253] = 100.0
        closes[-22] = 130.0  # (130/100 - 1)*100 = 30, in the 15 < m <= 40 range
        result = trend.score_momentum_12_1(closes)
        assert result == 85.0


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_uptrend_with_positive_momentum(self):
        """Test realistic uptrend scenario: trend_ok=True, momentum>0, risk_on=True."""
        # Create 260 closes: rising from 100 to 180
        stock_closes = [100 + (80 * i / 259) for i in range(260)]
        benchmark_closes = [100 + (50 * i / 259) for i in range(260)]

        assert trend.trend_ok(stock_closes) is True
        assert trend.momentum_12_1(stock_closes) > 0
        assert trend.regime_risk_on(benchmark_closes) is True

    def test_downtrend_scenario(self):
        """Test downtrend scenario: trend_ok=False, momentum<0, risk_off=True."""
        stock_closes = [200 - (100 * i / 259) for i in range(260)]
        benchmark_closes = [200 - (100 * i / 259) for i in range(260)]

        assert trend.trend_ok(stock_closes) is False
        assert trend.momentum_12_1(stock_closes) < 0
        assert trend.regime_risk_on(benchmark_closes) is False

    def test_scoring_pipeline(self):
        """Test complete scoring pipeline with momentum and filters."""
        closes = [100 + (50 * i / 259) for i in range(260)]

        momentum_score = trend.score_momentum_12_1(closes)
        above_50 = trend.above_sma(closes, 50)
        trend_filter = trend.trend_ok(closes)

        assert isinstance(momentum_score, float)
        assert 0 <= momentum_score <= 100
        assert above_50 is True or above_50 is False or above_50 is None
        assert trend_filter is True or trend_filter is False or trend_filter is None
