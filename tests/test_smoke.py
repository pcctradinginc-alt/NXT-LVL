"""Offline smoke tests — no real network calls.

Covers: scoring math/threshold behavior, tracking add/evaluate/stats with a
fake Tradier stub, email HTML building, LLM response validation, and option
selection filtering with fake chain data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("NXT_OFFLINE", "1")

from src.analysis import llm, options_math, scoring
from src.emergence import baseline as baseline_mod
from src.emergence import detector as emergence_detector
from src.mailer import build_email
from src.options.tradier import TradierClient
from src import tracking
from src.reward import engine as reward_engine
from src.reward import evaluator as reward_evaluator
from src.reward import weights as weights_mod


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

FIXTURE_DIGEST = {
    "edgar_capex": {
        "source": "edgar_capex",
        "companies": {"MSFT": {"yoy_growth_pct": 30.0}},
        "aggregate_capex_yoy_pct": 25.0,
    },
    "github_trends": {
        "source": "github_trends",
        "top_new_repos": [{"name": "foo/bar", "stars": 100}],
        "stage_heat": {3: 500, 4: 1000},
    },
    "jobs_hn": {
        "source": "jobs_hn",
        "stage_job_counts": {3: 40, 4: 10},
        "stage_job_mom_change": {3: 12, 4: -2},
        "total_comments": 500,
    },
    "arxiv_trends": {
        "source": "arxiv_trends",
        "stage_paper_counts": {3: 8, 4: 20},
        "sample_hot_titles": ["Some paper"],
    },
    "hn_buzz": {
        "source": "hn_buzz",
        "stage_buzz": {3: {"stories": 5, "points": 300}, 4: {"stories": 2, "points": 50}},
    },
}


def test_score_breadth_counts_only_active_sources():
    candidate = {
        "ticker": "VRT",
        "stage_id": 3,
        "source_evidence": ["edgar_capex", "github_trends", "jobs_hn", "hn_buzz", "arxiv_trends"],
    }
    breadth = scoring.score_breadth(candidate, FIXTURE_DIGEST)
    assert breadth == pytest.approx(100.0)


def test_score_breadth_partial_sources():
    # stage_id=1 has no per-stage activity in FIXTURE_DIGEST for github_trends,
    # jobs_hn, arxiv_trends, or hn_buzz (none of them have a stage-1 entry), so
    # only the explicit source_evidence ("edgar_capex", which is also credited
    # via its stage-agnostic aggregate figure) is counted -> 1/5 sources.
    candidate = {"ticker": "VRT", "stage_id": 1, "source_evidence": ["edgar_capex"]}
    breadth = scoring.score_breadth(candidate, FIXTURE_DIGEST)
    assert breadth == pytest.approx(20.0)


def test_score_stage_fit_exact_and_adjacent_and_none():
    assert scoring.score_stage_fit({"stage_id": 3}, next_stage=3) == 100.0
    assert scoring.score_stage_fit({"stage_id": 2}, next_stage=3) == 50.0
    assert scoring.score_stage_fit({"stage_id": 1}, next_stage=3) == 0.0


def test_score_divergence_buckets():
    assert scoring.score_divergence(None) == 50.0
    assert scoring.score_divergence(2.0) == 100.0
    assert scoring.score_divergence(10.0) == 70.0
    assert scoring.score_divergence(20.0) == 40.0
    assert scoring.score_divergence(50.0) == 10.0


def test_score_option_quality_neutral_when_missing():
    assert scoring.score_option_quality(None) == 50.0


def test_score_option_quality_good_liquidity_scores_high():
    option = {"bid": 4.9, "ask": 5.0, "mid": 4.95, "open_interest": 800}
    score = scoring.score_option_quality(option)
    assert score > 70


def test_conviction_multiplier_bounds():
    assert scoring.conviction_multiplier(0.0) == pytest.approx(0.8)
    assert scoring.conviction_multiplier(1.0) == pytest.approx(1.0)
    assert scoring.conviction_multiplier(None) == pytest.approx(0.9)


def test_score_candidates_sorted_and_weighted():
    candidates = [
        {
            "ticker": "VRT",
            "stage_id": 3,
            "thesis": "cooling demand",
            "source_evidence": ["edgar_capex", "github_trends", "jobs_hn"],
            "conviction": 0.9,
        },
        {
            "ticker": "XYZ",
            "stage_id": 1,
            "thesis": "irrelevant stage",
            "source_evidence": ["hn_buzz"],
            "conviction": 0.2,
        },
    ]
    scored = scoring.score_candidates(candidates, FIXTURE_DIGEST, next_stage=3)
    assert scored[0]["ticker"] == "VRT"
    assert scored[0]["total_score"] >= scored[1]["total_score"]
    # Divergence and option_quality are neutral (50) before option selection,
    # so with strong breadth/momentum/stage_fit this should still score well
    # above the weaker, wrong-stage, low-conviction competitor.
    assert scored[0]["total_score"] >= 60
    assert scored[1]["total_score"] < scored[0]["total_score"]


def test_divergence_affects_ranking():
    # Two otherwise-identical candidates, differing only in their perf_lookup
    # entry: a small 3-month move (<5% -> divergence 100) should outrank a
    # large one (>30% -> divergence 10), proving divergence (pre-gate, via
    # perf_lookup) actually influences ranking now.
    candidates = [
        {
            "ticker": "AAA",
            "stage_id": 3,
            "thesis": "small move",
            "source_evidence": ["edgar_capex", "github_trends"],
            "conviction": 0.7,
        },
        {
            "ticker": "BBB",
            "stage_id": 3,
            "thesis": "large move",
            "source_evidence": ["edgar_capex", "github_trends"],
            "conviction": 0.7,
        },
    ]
    perf_lookup = {"AAA": 2.0, "BBB": 40.0}
    scored = scoring.score_candidates(candidates, FIXTURE_DIGEST, next_stage=3, perf_lookup=perf_lookup)
    assert scored[0]["ticker"] == "AAA"
    assert scored[0]["scores"]["divergence"] == pytest.approx(100.0)
    assert scored[1]["scores"]["divergence"] == pytest.approx(10.0)
    assert scored[0]["total_score"] > scored[1]["total_score"]


def test_emergence_theme_mapping_boosts():
    # A candidate mapped to a hot emergent theme (score 80) should outrank an
    # identical candidate with no theme mapping (neutral 50), all else equal.
    all_theme_scores = {"enterprise_automation": 80.0}
    candidates = [
        {
            "ticker": "NOW",
            "stage_id": 3,
            "thesis": "themed",
            "source_evidence": ["edgar_capex", "github_trends"],
            "conviction": 0.7,
            "theme_id": "enterprise_automation",
        },
        {
            "ticker": "ZZZ",
            "stage_id": 3,
            "thesis": "unthemed",
            "source_evidence": ["edgar_capex", "github_trends"],
            "conviction": 0.7,
        },
    ]
    scored = scoring.score_candidates(
        candidates, FIXTURE_DIGEST, next_stage=3, all_theme_scores=all_theme_scores
    )
    assert scored[0]["ticker"] == "NOW"
    assert scored[0]["scores"]["emergence"] == pytest.approx(80.0)
    assert scored[1]["scores"]["emergence"] == pytest.approx(50.0)
    assert scored[0]["total_score"] > scored[1]["total_score"]


def test_breadth_credits_stage_active_sources():
    # source_evidence cites only jobs_hn, but stage_id=3 is active (non-zero)
    # in github_trends, arxiv_trends, hn_buzz, and edgar_capex too, per
    # FIXTURE_DIGEST -> those should be credited as well, not just the 1
    # explicitly-cited source.
    candidate = {"ticker": "VRT", "stage_id": 3, "source_evidence": ["jobs_hn"]}
    scored = scoring.score_candidate(candidate, FIXTURE_DIGEST, next_stage=3)
    assert scored["source_count"] >= 3
    assert scored["scores"]["breadth"] > 20.0
    credited = scoring._credited_sources(candidate, FIXTURE_DIGEST)
    assert credited == {"jobs_hn", "github_trends", "arxiv_trends", "hn_buzz", "edgar_capex"}


def test_score_candidate_with_good_option_and_divergence_clears_threshold():
    candidate = {
        "ticker": "VRT",
        "stage_id": 3,
        "thesis": "cooling demand",
        "source_evidence": ["edgar_capex", "github_trends", "jobs_hn", "hn_buzz"],
        "conviction": 0.9,
    }
    good_option = {"bid": 4.9, "ask": 5.0, "mid": 4.95, "open_interest": 800}
    scored = scoring.score_candidate(
        candidate,
        FIXTURE_DIGEST,
        next_stage=3,
        three_month_perf_pct=3.0,  # not yet priced in -> high divergence score
        option=good_option,
    )
    assert scored["total_score"] >= 70


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class FakeTradier:
    """Minimal Tradier stand-in for tracking tests."""

    def __init__(self, quotes: dict[str, dict]):
        self.quotes = quotes

    def get_quote(self, symbol: str):
        return self.quotes.get(symbol)


def test_add_signal_and_stats_roundtrip(tmp_path: Path):
    path = tmp_path / "signals.json"
    tracking.add_signal(
        ticker="VRT",
        score=85.0,
        thesis="test thesis",
        occ_symbol="VRT260116C00120000",
        strike=120.0,
        expiration="2026-01-16",
        entry_option_mid=5.0,
        entry_underlying=110.0,
        path=path,
    )
    signals = tracking.load_signals(path)
    assert len(signals) == 1
    assert signals[0]["status"] == "open"

    stats = tracking.stats(path=path)
    assert stats["open"] == 1
    assert stats["closed"] == 0
    assert stats["hit_rate"] is None


def test_evaluate_open_signals_closes_on_low_dte(tmp_path: Path):
    path = tmp_path / "signals.json"
    signals = [
        {
            "id": "abc123",
            "date": "2026-01-01",
            "ticker": "VRT",
            "occ_symbol": "VRT260601C00120000",
            "strike": 120.0,
            "expiration": "2026-07-20",  # close to "today" in the test env -> low DTE
            "entry_option_mid": 5.0,
            "entry_underlying": 110.0,
            "score": 85.0,
            "thesis": "test",
            "status": "open",
            "checkpoints": [],
            "result": None,
        }
    ]
    tracking.save_signals(signals, path)

    fake = FakeTradier({"VRT260601C00120000": {"bid": 6.0, "ask": 6.2}})
    updated = tracking.evaluate_open_signals(fake, close_after_trading_days=60, close_at_dte=40, path=path)

    sig = updated[0]
    assert len(sig["checkpoints"]) == 1
    # Expiration is well within 40 DTE of "today" per the fixture -> should close as a hit
    assert sig["status"] == "closed"
    assert sig["result"]["hit"] is True
    assert sig["result"]["pnl_pct"] > 0


def test_evaluate_open_signals_closes_when_unquotable(tmp_path: Path):
    path = tmp_path / "signals.json"
    signals = [
        {
            "id": "def456",
            "date": "2026-01-01",
            "ticker": "ZZZ",
            "occ_symbol": "ZZZ260101C00050000",
            "strike": 50.0,
            "expiration": "2027-01-01",
            "entry_option_mid": 2.0,
            "entry_underlying": 45.0,
            "score": 80.0,
            "thesis": "test",
            "status": "open",
            "checkpoints": [],
            "result": None,
        }
    ]
    tracking.save_signals(signals, path)

    fake = FakeTradier({})  # no quote available -> unquotable
    updated = tracking.evaluate_open_signals(fake, close_after_trading_days=60, close_at_dte=40, path=path)

    assert updated[0]["status"] == "closed"
    assert updated[0]["result"]["hit"] is False


def test_in_cooldown_true_within_window(tmp_path: Path):
    path = tmp_path / "signals.json"
    from datetime import date

    tracking.add_signal(
        ticker="VRT", score=80.0, thesis="x", path=path,
    )
    assert tracking.in_cooldown("VRT", days=14, path=path) is True
    assert tracking.in_cooldown("OTHER", days=14, path=path) is False


# ---------------------------------------------------------------------------
# Mailer
# ---------------------------------------------------------------------------

def test_build_email_with_top_pick_contains_ticker_and_hit_rate():
    result = {
        "stages_config": [{"id": 3, "name": "Energie / Kühlung / Netz"}],
        "current_stage": 2,
        "next_stage": 3,
        "stage_reasoning": "Test reasoning",
        "top_pick": {
            "ticker": "VRT",
            "total_score": 82.5,
            "thesis": "Cooling demand thesis",
            "option": {"strike": 120, "expiration": "2026-11-20", "mid": 5.25, "dte": 136, "delta": 0.65},
        },
        "top5": [
            {
                "ticker": "VRT",
                "total_score": 82.5,
                "scores": {"breadth": 80, "momentum": 70, "stage_fit": 100, "divergence": 70, "option_quality": 60},
            }
        ],
        "track_record": {"closed": 4, "open": 1, "hits": 3, "hit_rate": 75.0, "avg_pnl_pct": 12.3},
    }
    subject, html = build_email(result)
    assert "VRT" in subject
    assert "VRT" in html
    assert "75.0%" in html
    assert "Kühlung" in html or "Kühlung" in html


def test_build_email_no_signal():
    result = {
        "stages_config": [],
        "current_stage": None,
        "next_stage": None,
        "stage_reasoning": "",
        "top_pick": None,
        "top5": [],
        "track_record": {"closed": 0, "open": 0, "hits": 0, "hit_rate": None, "avg_pnl_pct": None},
    }
    subject, html = build_email(result)
    assert subject == "NXT LVL: Kein Signal heute"
    assert "Kein Kandidat" in html


# ---------------------------------------------------------------------------
# LLM response validation
# ---------------------------------------------------------------------------

def test_llm_validate_accepts_well_formed_payload():
    payload = {
        "current_stage": 2,
        "next_stage": 3,
        "reasoning": "ok",
        "candidates": [
            {
                "ticker": "VRT",
                "stage_id": 3,
                "thesis": "x",
                "source_evidence": ["edgar_capex"],
                "conviction": 0.7,
            }
        ],
    }
    validated = llm._validate(payload)
    assert validated["next_stage"] == 3


def test_llm_validate_rejects_missing_fields():
    with pytest.raises(llm.LLMResponseError):
        llm._validate({"current_stage": 1})


def test_llm_validate_rejects_non_dict():
    with pytest.raises(llm.LLMResponseError):
        llm._validate(["not", "a", "dict"])


def test_llm_dry_run_stub_matches_schema():
    stub = llm.dry_run_stub()
    validated = llm._validate(stub)
    assert validated["next_stage"] == 3
    assert len(validated["candidates"]) >= 3


# ---------------------------------------------------------------------------
# Option selection
# ---------------------------------------------------------------------------

FAKE_CHAIN = [
    {
        "symbol": "VRT260116C00110000",
        "option_type": "call",
        "strike": 110.0,
        "bid": 9.8,
        "ask": 10.0,
        "open_interest": 500,
        "greeks": {"delta": 0.65},
    },
    {
        "symbol": "VRT260116C00130000",
        "option_type": "call",
        "strike": 130.0,
        "bid": 3.0,
        "ask": 3.1,
        "open_interest": 50,  # too illiquid
        "greeks": {"delta": 0.62},
    },
    {
        "symbol": "VRT260116C00150000",
        "option_type": "call",
        "strike": 150.0,
        "bid": 1.0,
        "ask": 1.5,  # spread too wide
        "open_interest": 900,
        "greeks": {"delta": 0.61},
    },
    {
        "symbol": "VRT260116C00100000",
        "option_type": "call",
        "strike": 100.0,
        "bid": 12.0,
        "ask": 12.2,
        "open_interest": 700,
        "greeks": {"delta": 0.85},  # delta out of range
    },
    {
        "symbol": "VRT260116P00110000",
        "option_type": "put",
        "strike": 110.0,
        "bid": 5.0,
        "ask": 5.1,
        "open_interest": 700,
        "greeks": {"delta": -0.65},
    },
]


class SelectableTradierClient(TradierClient):
    """Subclass overriding network methods with fixtures for select_option tests."""

    def __init__(self, expirations, chain, quote=None):
        super().__init__(api_key="fake", env="prod")
        self._expirations = expirations
        self._chain = chain
        self._quote = quote

    def get_quote(self, symbol: str):
        return self._quote

    def get_expirations(self, symbol: str):
        return self._expirations

    def get_chain(self, symbol: str, expiration: str):
        return self._chain


def _future_date_str(days: int) -> str:
    from datetime import date, timedelta

    return (date.today() + timedelta(days=days)).isoformat()


def test_select_option_filters_delta_oi_spread():
    exp = _future_date_str(150)
    client = SelectableTradierClient(
        expirations=[exp],
        chain=FAKE_CHAIN,
        quote={"last": 115.0},
    )
    result = client.select_option("VRT")
    assert result is not None
    assert result["occ_symbol"] == "VRT260116C00110000"
    assert 0.60 <= result["delta"] <= 0.70
    assert result["open_interest"] >= 100


def test_select_option_returns_none_when_no_expiration_in_window():
    client = SelectableTradierClient(
        expirations=[_future_date_str(10)],  # too short
        chain=FAKE_CHAIN,
    )
    result = client.select_option("VRT")
    assert result is None


def test_select_option_returns_none_when_no_call_matches_filters():
    exp = _future_date_str(150)
    illiquid_chain = [c for c in FAKE_CHAIN if c["symbol"] != "VRT260116C00110000"]
    client = SelectableTradierClient(expirations=[exp], chain=illiquid_chain)
    result = client.select_option("VRT")
    assert result is None


def test_get_quote_handles_list_and_dict_shapes(monkeypatch):
    client = TradierClient(api_key="fake", env="prod")

    monkeypatch.setattr(client, "_get", lambda path, params=None: {"quotes": {"quote": {"symbol": "VRT", "last": 100.0}}})
    quote = client.get_quote("VRT")
    assert quote["symbol"] == "VRT"

    monkeypatch.setattr(
        client, "_get", lambda path, params=None: {"quotes": {"quote": [{"symbol": "VRT", "last": 100.0}]}}
    )
    quote_list = client.get_quote("VRT")
    assert quote_list["symbol"] == "VRT"


# ---------------------------------------------------------------------------
# Emergence & Reward Engine
# ---------------------------------------------------------------------------

EMERGENCE_THEMES = [
    {
        "id": "dc_cooling",
        "name": "Data Center Cooling",
        "keywords": ["liquid cooling", "immersion cooling"],
        "tickers": ["VRT", "MOD"],
    },
    {
        "id": "quiet_theme",
        "name": "Quiet Theme",
        "keywords": ["quiet keyword phrase"],
        "tickers": ["ZQX"],
    },
]

EMERGENCE_CFG = {
    "baseline_window": 30,
    "min_history_for_baseline": 3,
    "novelty_window_days": 90,
    "min_sources": 2,
    "theme_threshold": 60,
    "score_weights": {"frequency": 0.30, "acceleration": 0.35, "diversity": 0.20, "novelty": 0.15},
}


def test_emergence_detects_new_theme():
    # Baseline has a thin, low-frequency history for dc_cooling (well below
    # min_history_for_baseline) and no history at all for quiet_theme.
    baseline_obj = {
        "themes": {
            "dc_cooling": [
                {"date": "2026-04-01", "per_source_counts": {"edgar_fts": 1}, "frequency": 1, "source_diversity": 1},
                {"date": "2026-04-15", "per_source_counts": {"edgar_fts": 1}, "frequency": 1, "source_diversity": 1},
            ]
        }
    }

    digest = {
        "edgar_fts": {"source": "edgar_fts", "theme_counts": {"dc_cooling": 50, "quiet_theme": 0}},
        "github_trends": {
            "source": "github_trends",
            "top_new_repos": [
                {"name": "acme/liquid-cooling-cdu", "description": "immersion cooling for datacenters", "topics": ["cooling"]}
            ],
        },
        "arxiv_trends": {"source": "arxiv_trends", "sample_hot_titles": ["A study on liquid cooling for AI datacenters"]},
        "hn_buzz": {"source": "hn_buzz", "stage_buzz": {}},
    }

    result = emergence_detector.detect(
        digest, EMERGENCE_THEMES, EMERGENCE_CFG, baseline_obj, entity_aliases={}, megacap_exclude=[]
    )

    scores = result["all_theme_scores"]
    assert scores["dc_cooling"] >= EMERGENCE_CFG["theme_threshold"]
    assert scores["dc_cooling"] > scores["quiet_theme"]

    dc_theme = next(t for t in result["emergent_themes"] if t["theme_id"] == "dc_cooling")
    assert dc_theme["emergence_score"] >= 60
    assert dc_theme["acceleration_z"] > 0
    assert dc_theme["source_diversity"] >= 2

    # Baseline should have grown by one observation for dc_cooling.
    assert len(baseline_obj["themes"]["dc_cooling"]) == 3


def test_emergent_candidate_outside_watchlist():
    # MOD is not in this fixture's "watchlist" (we simulate the watchlist
    # check the same way main.py does: membership in stages[].tickers).
    watchlist_tickers = {"VRT"}  # MOD deliberately excluded

    baseline_obj = {"themes": {}}
    digest = {
        "edgar_fts": {"source": "edgar_fts", "theme_counts": {"dc_cooling": 40, "quiet_theme": 0}},
        "github_trends": {
            "source": "github_trends",
            "top_new_repos": [
                {"name": "someone/modine-immersion-cooling", "description": "Modine immersion cooling module", "topics": []}
            ],
        },
        "arxiv_trends": {"source": "arxiv_trends", "sample_hot_titles": ["Modine liquid cooling deployment study"]},
        "hn_buzz": {"source": "hn_buzz", "stage_buzz": {}},
    }
    entity_aliases = {"MOD": ["Modine"]}

    result = emergence_detector.detect(
        digest, EMERGENCE_THEMES, EMERGENCE_CFG, baseline_obj, entity_aliases=entity_aliases, megacap_exclude=[]
    )

    assert any(t["theme_id"] == "dc_cooling" for t in result["emergent_themes"])

    mod_candidate = next((c for c in result["emergent_candidates"] if c["ticker"] == "MOD"), None)
    assert mod_candidate is not None
    assert mod_candidate["theme_id"] == "dc_cooling"
    assert len(mod_candidate["sources"]) >= 2

    in_watchlist = mod_candidate["ticker"] in watchlist_tickers
    assert in_watchlist is False


class FakeHistoryTradier:
    """Fake Tradier stand-in providing deterministic price history bars."""

    def __init__(self, history_by_symbol: dict[str, list[dict]]):
        self.history_by_symbol = history_by_symbol

    def get_history(self, symbol: str, start: str, interval: str = "daily"):
        return self.history_by_symbol.get(symbol, [])


def _daily_bars(start_date: str, prices: list[float]) -> list[dict]:
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    bars = []
    for i, price in enumerate(prices):
        d = start + timedelta(days=i)
        bars.append({"date": d.isoformat(), "close": price})
    return bars


def test_retroactive_evaluation():
    from datetime import date, timedelta

    signal_date = (date.today() - timedelta(days=200)).isoformat()

    # Underlying steadily rises; benchmark stays flat -> positive alpha at every horizon.
    underlying_prices = [100.0 + i * 0.5 for i in range(210)]
    benchmark_prices = [50.0 for _ in range(210)]

    signal = {
        "id": "sig1",
        "date": signal_date,
        "ticker": "VRT",
        "score": 85.0,
        "thesis": "test",
        "status": "open",
        "checkpoints": [],
        "result": None,
        "option_idea": {"strike": 120.0, "mid": 5.0, "exp": (date.today() + timedelta(days=60)).isoformat(), "oi": 500, "spread": 0.03},
        "benchmark_symbol": "SPY",
        "horizon_evals": {},
    }

    fake = FakeHistoryTradier(
        {
            "VRT": _daily_bars(signal_date, underlying_prices),
            "SPY": _daily_bars(signal_date, benchmark_prices),
        }
    )

    reward_cfg = {"horizons": [30, 60, 90, 180], "benchmark_symbol": "SPY"}
    updated = reward_evaluator.evaluate_signals(
        [signal], fake, reward_cfg, current_emergence_scores=None, path="/tmp/does_not_matter_signals.json"
    )

    horizon_evals = updated[0]["horizon_evals"]
    for horizon in (30, 60, 90, 180):
        key = str(horizon)
        assert key in horizon_evals
        ev = horizon_evals[key]
        assert ev["alpha"] > 0
        assert ev["hit"] is True
        assert ev["max_drawdown"] >= 0


def test_weight_update_bounded_and_logged():
    weights_obj = {
        "feature_weights": {"breadth": 0.20, "momentum": 0.20, "stage_fit": 0.15, "divergence": 0.15, "option_quality": 0.10, "emergence": 0.20},
        "source_reliability": {"edgar_capex": 1.0},
        "history": [],
    }
    reward_cfg = {
        "weight_bounds": {"min": 0.05, "max": 0.45},
        "reliability_bounds": {"min": 0.5, "max": 1.5},
        "step_max": 0.02,
        "learning_rate": 0.04,
        "min_samples": 5,
    }
    ledgers = {
        "features": {
            # High win rate, n well above min_samples -> should be nudged up.
            "momentum": {"n": 10, "wins": 9, "sum_alpha": 50.0},
            # Below min_samples -> must be skipped with a documented reason.
            "stage_fit": {"n": 2, "wins": 2, "sum_alpha": 5.0},
        },
        "sources": {},
    }

    old_momentum = weights_obj["feature_weights"]["momentum"]
    updated = reward_engine.update_weights(weights_obj, ledgers, reward_cfg)

    new_momentum = updated["feature_weights"]["momentum"]
    # Nudge should be positive and bounded (before renormalization the raw
    # per-feature step is clipped to step_max; renormalization can shift it
    # further but must stay within the configured absolute bounds).
    assert new_momentum >= reward_cfg["weight_bounds"]["min"]
    assert new_momentum <= reward_cfg["weight_bounds"]["max"]

    total = sum(updated["feature_weights"].values())
    assert total == pytest.approx(1.0, abs=1e-3)

    history = updated["history"]
    momentum_entries = [h for h in history if h["target"] == "feature:momentum"]
    assert momentum_entries, "expected a logged history entry for momentum"
    assert "win_rate" in momentum_entries[0]["reason"] or "adjusted" in momentum_entries[0]["reason"]

    skip_entries = [h for h in history if h["target"] == "feature:stage_fit"]
    assert skip_entries, "expected a logged skip entry for stage_fit"
    assert "skipped" in skip_entries[0]["reason"]


def test_report_explains_candidate():
    result = {
        "stages_config": [{"id": 3, "name": "Energie / Kühlung / Netz"}],
        "current_stage": 2,
        "next_stage": 3,
        "stage_reasoning": "Test reasoning",
        "top_pick": {
            "ticker": "MOD",
            "total_score": 82.5,
            "thesis": "Emergent cooling theme thesis",
            "option": {"strike": 120, "expiration": "2026-11-20", "mid": 5.25, "dte": 136, "delta": 0.65},
            "discovery": {
                "via": "emergence",
                "theme_id": "dc_cooling",
                "theme": "Data Center Cooling",
                "drivers": {"frequency": 40, "acceleration_ratio": 5.0, "diversity": 3, "novelty": 0.9},
                "confirming_sources": ["edgar_fts", "github_trends", "arxiv_trends"],
            },
            "risks": ["Divergenz niedrig — der Titel ist bereits stark gelaufen."],
        },
        "top5": [],
        "track_record": {"closed": 4, "open": 1, "hits": 3, "hit_rate": 75.0, "avg_pnl_pct": 12.3},
        "emergent_themes": [
            {
                "theme_id": "dc_cooling",
                "name": "Data Center Cooling",
                "emergence_score": 78.4,
                "acceleration_z": 3.1,
                "source_diversity": 3,
                "novelty": 0.9,
                "drivers": {"frequency": 40, "acceleration_ratio": 5.0, "diversity": 3, "novelty": 0.9},
                "confirming_sources": ["edgar_fts", "github_trends", "arxiv_trends"],
            }
        ],
        "reward": {
            "feature_weights": {"breadth": 0.2, "momentum": 0.2, "emergence": 0.2},
            "source_reliability": {"edgar_capex": 1.0},
            "history": [{"date": "2026-07-01", "target": "feature:momentum", "old": 0.2, "new": 0.22, "reason": "adjusted"}],
            "hit_rate_by_horizon": {"30": 60.0, "90": 75.0},
        },
    }
    subject, html = build_email(result)

    assert "Warum" in html
    assert "Risiken" in html
    assert "edgar_fts" in html or "github_trends" in html
    assert "Data Center Cooling" in html


def test_options_math_call_price_monotonic():
    low = options_math.bs_call_price(S=90.0, K=100.0, T_years=0.5, r=0.04, sigma=0.5)
    high = options_math.bs_call_price(S=110.0, K=100.0, T_years=0.5, r=0.04, sigma=0.5)
    assert high > low

    value = options_math.estimate_call_value(underlying_now=120.0, strike=100.0, dte_days_remaining=90)
    assert value > 0
    # Deep ITM call should be worth at least its intrinsic value.
    assert value >= 20.0 - 1.0  # small tolerance for numerical rounding


def test_config_loads_themes_and_reward():
    from src.config import load_settings

    settings = load_settings()
    assert len(settings.themes) == 11
    assert settings.reward_config.get("benchmark_symbol") == "SPY"
    assert settings.emergence_config.get("theme_threshold") == 60
    assert "NVDA" in settings.megacap_exclude
