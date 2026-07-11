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

from src.analysis import insider, llm, options_math, phases, scoring, structures
from src.collectors import edgar_language
from src.emergence import baseline as baseline_mod
from src.emergence import detector as emergence_detector
from src.mailer import build_email
from src.options.tradier import TradierClient
from src import tracking
from src.main import _cluster_member_count, apply_claims_adjustment, compute_data_quality
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
    # Signed mapping (fix 5): downtrends must NOT be rewarded as if they were
    # "not run up". A falling knife (-20%) scores low, not high.
    assert scoring.score_divergence(-20.0) == 10.0
    assert scoring.score_divergence(2.0) == 100.0


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
        "greeks": {"delta": 0.65, "mid_iv": 0.42},
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


def test_select_option_carries_iv_from_greeks():
    # Fix 6: the selected option dict must surface the chain's implied
    # volatility (mid_iv, falling back to smv_vol) so the reward evaluator
    # can re-price the option with real IV instead of the 0.5 fallback.
    exp = _future_date_str(150)
    client = SelectableTradierClient(
        expirations=[exp],
        chain=FAKE_CHAIN,
        quote={"last": 115.0},
    )
    result = client.select_option("VRT")
    assert result is not None
    assert result["iv"] == pytest.approx(0.42)


def test_select_option_iv_falls_back_to_smv_vol():
    exp = _future_date_str(150)
    chain = [dict(FAKE_CHAIN[0])]
    chain[0] = dict(chain[0])
    chain[0]["greeks"] = {"delta": 0.65, "smv_vol": 0.37}
    client = SelectableTradierClient(expirations=[exp], chain=chain, quote={"last": 115.0})
    result = client.select_option("VRT")
    assert result is not None
    assert result["iv"] == pytest.approx(0.37)


def test_select_option_iv_none_when_greeks_missing():
    exp = _future_date_str(150)
    chain = [dict(FAKE_CHAIN[0])]
    chain[0] = dict(chain[0])
    chain[0]["greeks"] = {"delta": 0.65}
    client = SelectableTradierClient(expirations=[exp], chain=chain, quote={"last": 115.0})
    result = client.select_option("VRT")
    assert result is not None
    assert result["iv"] is None


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
    # Baseline has a warmed-up, low-frequency history for dc_cooling (>=
    # min_history_for_baseline observations) and no history for quiet_theme.
    baseline_obj = {
        "themes": {
            "dc_cooling": [
                {"date": "2026-03-15", "per_source_counts": {"edgar_fts": 1}, "frequency": 1, "source_diversity": 1},
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
    assert len(baseline_obj["themes"]["dc_cooling"]) == 4


def test_emergence_warmup_gate():
    # Weakness 8b: with no baseline history (cold start), even a very high
    # current frequency must NOT flag a theme as emergent — acceleration and
    # novelty are spurious until >= min_history_for_baseline observations exist.
    baseline_obj = {"themes": {}}
    digest = {
        "edgar_fts": {"source": "edgar_fts", "theme_counts": {"dc_cooling": 99, "quiet_theme": 0}},
        "github_trends": {
            "source": "github_trends",
            "top_new_repos": [
                {"name": "x/liquid-cooling-cdu", "description": "immersion cooling datacenters", "topics": ["cooling"]}
            ],
        },
        "arxiv_trends": {"source": "arxiv_trends", "sample_hot_titles": ["liquid cooling for AI datacenters"]},
        "hn_buzz": {"source": "hn_buzz", "stage_buzz": {}},
    }
    result = emergence_detector.detect(
        digest, EMERGENCE_THEMES, EMERGENCE_CFG, baseline_obj, entity_aliases={}, megacap_exclude=[]
    )
    # No theme may be flagged emergent during warm-up, hence no emergent candidates.
    assert result["emergent_themes"] == []
    assert result["emergent_candidates"] == []


def test_emergent_candidate_outside_watchlist():
    # MOD is not in this fixture's "watchlist" (we simulate the watchlist
    # check the same way main.py does: membership in stages[].tickers).
    watchlist_tickers = {"VRT"}  # MOD deliberately excluded

    baseline_obj = {
        "themes": {
            "dc_cooling": [
                {"date": "2026-03-15", "per_source_counts": {"edgar_fts": 1}, "frequency": 1, "source_diversity": 1},
                {"date": "2026-04-01", "per_source_counts": {"edgar_fts": 1}, "frequency": 1, "source_diversity": 1},
                {"date": "2026-04-15", "per_source_counts": {"edgar_fts": 1}, "frequency": 1, "source_diversity": 1},
            ]
        }
    }
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


def test_hit_is_option_based():
    from datetime import date, datetime, timedelta

    # Case A: option_idea present and BS re-check says it's still worth more
    # than entry mid -> hit True, hit_basis "option" (uses the same rising
    # underlying / option setup as test_retroactive_evaluation).
    signal_date = (date.today() - timedelta(days=200)).isoformat()
    underlying_prices = [100.0 + i * 0.5 for i in range(210)]
    benchmark_prices = [50.0 for _ in range(210)]

    profitable_signal = {
        "id": "sig_profitable",
        "date": signal_date,
        "ticker": "VRT",
        "score": 85.0,
        "thesis": "test",
        "status": "open",
        "checkpoints": [],
        "result": None,
        "option_idea": {
            "strike": 120.0,
            "mid": 5.0,
            "exp": (date.today() + timedelta(days=60)).isoformat(),
            "oi": 500,
            "spread": 0.03,
        },
        "benchmark_symbol": "SPY",
        "horizon_evals": {},
    }
    fake = FakeHistoryTradier(
        {
            "VRT": _daily_bars(signal_date, underlying_prices),
            "SPY": _daily_bars(signal_date, benchmark_prices),
        }
    )
    reward_cfg = {"horizons": [90], "benchmark_symbol": "SPY"}
    updated = reward_evaluator.evaluate_signals(
        [profitable_signal], fake, reward_cfg, current_emergence_scores=None,
        path="/tmp/does_not_matter_signals_a.json",
    )
    ev = updated[0]["horizon_evals"]["90"]
    assert ev["hit"] is True
    assert ev["hit_basis"] == "option"

    # Case B: the stock rose (absolute_return > 0) but the option itself
    # (deep OTM strike, high entry mid, little time left at the horizon) is
    # worth less than what was paid -> option_profitable False -> hit False,
    # even though the underlying went up. This is exactly the scenario the
    # old alpha/absolute_return-based `hit` definition got wrong.
    losing_signal_date = (date.today() - timedelta(days=100)).isoformat()
    mild_up_prices = [100.0 + i * 0.05 for i in range(110)]  # slow +5% drift
    flat_benchmark = [50.0 for _ in range(110)]
    losing_signal = {
        "id": "sig_losing",
        "date": losing_signal_date,
        "ticker": "ZZZ",
        "score": 70.0,
        "thesis": "test",
        "status": "open",
        "checkpoints": [],
        "result": None,
        "option_idea": {
            "strike": 150.0,  # deep OTM relative to ~100-105 underlying
            "mid": 3.0,
            "exp": (datetime.strptime(losing_signal_date, "%Y-%m-%d").date() + timedelta(days=95)).isoformat(),
            "iv": 0.3,
        },
        "benchmark_symbol": "SPY",
        "horizon_evals": {},
    }
    fake2 = FakeHistoryTradier(
        {
            "ZZZ": _daily_bars(losing_signal_date, mild_up_prices),
            "SPY": _daily_bars(losing_signal_date, flat_benchmark),
        }
    )
    updated2 = reward_evaluator.evaluate_signals(
        [losing_signal], fake2, reward_cfg, current_emergence_scores=None,
        path="/tmp/does_not_matter_signals_b.json",
    )
    ev2 = updated2[0]["horizon_evals"]["90"]
    assert ev2["abs_return"] > 0  # the stock did go up
    assert ev2["option_profitable"] is False  # but the option is worth less than paid
    assert ev2["hit"] is False
    assert ev2["hit_basis"] == "option"


# ---------------------------------------------------------------------------
# Reward engine: cumulative ledger + convergent weight recomputation
# ---------------------------------------------------------------------------

def _base_reward_cfg(**overrides):
    cfg = {
        "weight_bounds": {"min": 0.05, "max": 0.45},
        "reliability_bounds": {"min": 0.5, "max": 1.5},
        "step_max": 0.02,
        "learning_rate": 0.04,
        "min_samples": 5,
    }
    cfg.update(overrides)
    return cfg


def _signal_with_eval(sig_id, horizon, hit, alpha, feature_attribution, source_attribution,
                       score=50.0, data_quality_score=100.0):
    return {
        "id": sig_id,
        "date": "2026-01-01",
        "ticker": "XYZ",
        "score": score,
        "data_quality_score": data_quality_score,
        "feature_attribution": feature_attribution,
        "source_attribution": source_attribution,
        "horizon_evals": {
            str(horizon): {"hit": hit, "alpha": alpha, "abs_return": alpha, "hit_basis": "option"}
        },
    }


def test_reward_ledger_consume_once():
    weights_obj = {
        "feature_weights": {"momentum": 0.2},
        "source_reliability": {"edgar_capex": 1.0},
        "history": [],
        "ledger": {"features": {}, "sources": {}},
        "rewarded_evals": [],
    }
    signals = [
        _signal_with_eval("s1", 90, True, 5.0, {"momentum": 40.0}, ["edgar_capex"]),
        _signal_with_eval("s2", 90, False, -2.0, {"momentum": 30.0}, ["edgar_capex"]),
    ]

    first = reward_engine.accumulate_ledger(weights_obj, signals, primary_horizon=90, overheated_threshold=80.0)
    assert first == 2
    n_after_first = weights_obj["ledger"]["features"]["momentum"]["n"]

    second = reward_engine.accumulate_ledger(weights_obj, signals, primary_horizon=90, overheated_threshold=80.0)
    assert second == 0
    n_after_second = weights_obj["ledger"]["features"]["momentum"]["n"]
    assert n_after_second == n_after_first  # cumulative ledger did not double


def test_reward_weights_converge_not_drift():
    base_feature_weights = {
        "breadth": 0.20, "momentum": 0.20, "stage_fit": 0.15,
        "divergence": 0.15, "option_quality": 0.10, "emergence": 0.20,
    }
    base_reliability = {src: 1.0 for src in scoring.ALL_SOURCES}
    weights_obj = {
        "feature_weights": dict(base_feature_weights),
        "source_reliability": dict(base_reliability),
        "history": [],
        "ledger": {
            "features": {
                # High win rate, well above min_samples -> target pulls weight up.
                "momentum": {"n": 40.0, "wins": 36.0, "sum_reward": 200.0},
            },
            "sources": {},
        },
        "rewarded_evals": [],
    }
    reward_cfg = _base_reward_cfg()

    history_of_momentum = []
    for _ in range(50):
        weights_obj = reward_engine.recompute_weights(weights_obj, reward_cfg, base_feature_weights, base_reliability)
        history_of_momentum.append(weights_obj["feature_weights"]["momentum"])

    # Converges: stabilizes (last two iterations equal) rather than marching
    # to the bound purely from repeated calls on the same static ledger.
    assert history_of_momentum[-1] == pytest.approx(history_of_momentum[-2], abs=1e-9)
    assert history_of_momentum[-1] < reward_cfg["weight_bounds"]["max"]
    assert history_of_momentum[-1] > base_feature_weights["momentum"]

    momentum_history_entries = [h for h in weights_obj["history"] if h["target"] == "feature:momentum"]
    assert momentum_history_entries, "expected at least one logged history entry for momentum"


def test_reward_skips_when_below_min_samples():
    base_feature_weights = {
        "breadth": 0.20, "momentum": 0.20, "stage_fit": 0.15,
        "divergence": 0.15, "option_quality": 0.10, "emergence": 0.20,
    }
    base_reliability = {src: 1.0 for src in scoring.ALL_SOURCES}
    weights_obj = {
        "feature_weights": dict(base_feature_weights),
        "source_reliability": dict(base_reliability),
        "history": [],
        "ledger": {
            "features": {
                "stage_fit": {"n": 2.0, "wins": 2.0, "sum_reward": 10.0},  # n < min_samples (5)
            },
            "sources": {},
        },
        "rewarded_evals": [],
    }
    reward_cfg = _base_reward_cfg()

    updated = reward_engine.recompute_weights(weights_obj, reward_cfg, base_feature_weights, base_reliability)
    assert updated["feature_weights"]["stage_fit"] == pytest.approx(base_feature_weights["stage_fit"])


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


# ---------------------------------------------------------------------------
# Options structure-selection layer (options_math theta + structures.py)
# ---------------------------------------------------------------------------

def test_bs_call_theta_negative():
    # Theta (time decay) of an ATM call is negative: the option loses value as
    # time passes, all else equal.
    theta = options_math.bs_call_theta(S=100.0, K=100.0, T_years=0.5, r=0.04, sigma=0.5)
    assert theta < 0
    # Degenerate cases collapse to 0.0 rather than raising.
    assert options_math.bs_call_theta(S=100.0, K=100.0, T_years=0.0, r=0.04, sigma=0.5) == 0.0
    assert options_math.bs_call_theta(S=100.0, K=100.0, T_years=0.5, r=0.04, sigma=0.0) == 0.0


def test_realized_vol():
    # A gently oscillating price series yields a plausible (clipped) annualized
    # vol; a too-short series returns None.
    prices = [100.0 + (2.0 if i % 2 else -2.0) for i in range(80)]
    rvol = structures.realized_vol(prices)
    assert rvol is not None
    assert 0.05 <= rvol <= 3.0

    # Fewer than ~20 usable returns -> None.
    assert structures.realized_vol([100.0, 101.0, 102.0]) is None
    assert structures.realized_vol([]) is None


def test_iv_expensive():
    # IV 0.9 vs realized 0.4 at ratio 1.6 -> 0.9 > 0.64 -> expensive.
    assert structures.iv_expensive(0.9, 0.4, 1.6) is True
    # IV 0.5 vs realized 0.4 -> 0.5 <= 0.64 -> not expensive.
    assert structures.iv_expensive(0.5, 0.4, 1.6) is False
    # Missing inputs -> never expensive.
    assert structures.iv_expensive(None, 0.4, 1.6) is False
    assert structures.iv_expensive(0.9, None, 1.6) is False


def test_long_call_metrics_break_even():
    m = structures.long_call_metrics(
        underlying=100.0, strike=105.0, mid=4.0, delta=0.6, iv=0.5, dte_days=120
    )
    assert m["break_even"] == pytest.approx(109.0)  # strike + mid
    # (109/100 - 1) * 100 = 9.0%
    assert m["break_even_move_pct"] == pytest.approx(9.0)
    assert m["max_loss"] == pytest.approx(400.0)  # mid * 100
    assert m["theta_per_day"] is not None
    assert m["theta_per_day"] < 0


def test_call_spread_metrics():
    m = structures.call_spread_metrics(
        underlying=100.0, long_strike=100.0, long_mid=6.0, short_strike=110.0, short_mid=2.0, dte_days=120
    )
    assert m is not None
    assert m["net_debit"] == pytest.approx(4.0)  # 6 - 2
    assert m["width"] == pytest.approx(10.0)
    assert m["max_profit"] == pytest.approx(600.0)  # (10 - 4) * 100
    assert m["max_loss"] == pytest.approx(400.0)  # 4 * 100
    assert m["break_even"] == pytest.approx(104.0)  # long_strike + net_debit
    assert m["break_even_move_pct"] == pytest.approx(4.0)

    # Invalid: no short mid.
    assert structures.call_spread_metrics(100.0, 100.0, 6.0, 110.0, None, 120) is None
    # Invalid: short strike <= long strike.
    assert structures.call_spread_metrics(100.0, 100.0, 6.0, 100.0, 2.0, 120) is None
    # Invalid: non-positive net debit (short mid >= long mid).
    assert structures.call_spread_metrics(100.0, 100.0, 4.0, 110.0, 4.0, 120) is None


def test_choose_structure_branches():
    cfg = {"max_iv_realized_ratio": 1.6}
    long_call = {"strike": 100.0, "mid": 6.0, "delta": 0.65, "iv": 0.5, "dte": 120}
    short_leg = {"strike": 110.0, "mid": 2.0, "delta": 0.32}

    # Not expensive (IV 0.5 vs rvol 0.4 -> 0.5 <= 0.64) -> long_call.
    not_exp = structures.choose_structure(100.0, long_call, short_leg, 0.4, cfg)
    assert not_exp["structure"] == "long_call"
    assert not_exp["iv_expensive"] is False
    assert "im Rahmen" in not_exp["reason"]

    # Expensive (IV 0.9) + valid short leg -> call_spread.
    exp_call = dict(long_call, iv=0.9)
    spread = structures.choose_structure(100.0, exp_call, short_leg, 0.4, cfg)
    assert spread["structure"] == "call_spread"
    assert spread["iv_expensive"] is True
    assert "Call-Spread" in spread["reason"]

    # Expensive + no valid short leg -> stock.
    stock = structures.choose_structure(100.0, exp_call, None, 0.4, cfg)
    assert stock["structure"] == "stock"
    assert stock["iv_expensive"] is True
    assert "Aktie" in stock["reason"]


SHORT_LEG_CHAIN = [
    {
        "symbol": "VRT260116C00120000",
        "option_type": "call",
        "strike": 120.0,
        "bid": 5.0,
        "ask": 5.2,
        "open_interest": 400,
        "greeks": {"delta": 0.45},  # above target 0.32 -> excluded
    },
    {
        "symbol": "VRT260116C00135000",
        "option_type": "call",
        "strike": 135.0,
        "bid": 2.4,
        "ask": 2.6,
        "open_interest": 300,
        "greeks": {"delta": 0.31},  # closest to 0.32 and not above target
    },
    {
        "symbol": "VRT260116C00150000",
        "option_type": "call",
        "strike": 150.0,
        "bid": 1.0,
        "ask": 1.1,
        "open_interest": 200,
        "greeks": {"delta": 0.20},  # further from 0.32
    },
    {
        "symbol": "VRT260116C00160000",
        "option_type": "call",
        "strike": 160.0,
        "bid": 0.5,
        "ask": 0.6,
        "open_interest": 20,  # too illiquid
        "greeks": {"delta": 0.30},
    },
    {
        "symbol": "VRT260116P00135000",
        "option_type": "put",
        "strike": 135.0,
        "bid": 2.0,
        "ask": 2.1,
        "open_interest": 300,
        "greeks": {"delta": -0.33},
    },
]


def test_select_short_leg():
    exp = _future_date_str(150)
    client = SelectableTradierClient(expirations=[exp], chain=SHORT_LEG_CHAIN, quote={"last": 115.0})
    leg = client.select_short_leg("VRT", exp, target_delta=0.32)
    assert leg is not None
    # The 0.45-delta call is above target (excluded); the 0.31-delta call is the
    # closest one not above target 0.32 among eligible (OI, spread) candidates.
    assert leg["occ_symbol"] == "VRT260116C00135000"
    assert leg["delta"] == pytest.approx(0.31)
    assert leg["open_interest"] >= 50

    # No eligible short leg -> None (all too illiquid / above target).
    illiquid = [
        {
            "symbol": "VRT260116C00160000",
            "option_type": "call",
            "strike": 160.0,
            "bid": 0.5,
            "ask": 0.6,
            "open_interest": 10,
            "greeks": {"delta": 0.30},
        }
    ]
    client2 = SelectableTradierClient(expirations=[exp], chain=illiquid, quote={"last": 115.0})
    assert client2.select_short_leg("VRT", exp, target_delta=0.32) is None


# ---------------------------------------------------------------------------
# Decision gates & reporting (#11 sector benchmarks, #14 invalidation levels,
# #17 data-quality gate, #20 signal clustering)
# ---------------------------------------------------------------------------

def test_benchmark_for_maps_stage():
    from src.config import Settings, load_settings

    settings = load_settings()
    assert settings.benchmark_for(5) == "IGV"
    assert settings.benchmark_for(3) == "XLU"
    assert settings.benchmark_for(99) == "SPY"  # unmapped stage -> configured default

    # Handles str-keyed benchmark dicts (and str stage_id queries) too, not
    # just YAML's int-keyed parsing.
    str_keyed = Settings(raw={"benchmarks": {"5": "IGV", "3": "XLU", "default": "SPY"}})
    assert str_keyed.benchmark_for(5) == "IGV"
    assert str_keyed.benchmark_for("3") == "XLU"
    assert str_keyed.benchmark_for(None) == "SPY"


class RecordingHistoryTradier(FakeHistoryTradier):
    """FakeHistoryTradier that records which symbols get_history() was called with."""

    def __init__(self, history_by_symbol: dict[str, list[dict]]):
        super().__init__(history_by_symbol)
        self.requested_symbols: list[str] = []

    def get_history(self, symbol: str, start: str, interval: str = "daily"):
        self.requested_symbols.append(symbol)
        return super().get_history(symbol, start, interval)


def test_evaluator_uses_signal_benchmark():
    from datetime import date, timedelta

    signal_date = (date.today() - timedelta(days=100)).isoformat()
    underlying_prices = [100.0 + i * 0.3 for i in range(110)]
    soxx_prices = [50.0 + i * 0.1 for i in range(110)]

    signal = {
        "id": "sig_bench",
        "date": signal_date,
        "ticker": "MRVL",
        "score": 80.0,
        "thesis": "test",
        "status": "open",
        "checkpoints": [],
        "result": None,
        "benchmark_symbol": "SOXX",  # per-signal sector benchmark, not the global default
        "horizon_evals": {},
    }
    # Note: "SPY" is deliberately absent from history_by_symbol — if the
    # evaluator fell back to the reward_cfg global default instead of the
    # signal's own benchmark_symbol, this would fetch SPY (empty history)
    # instead of SOXX and the horizon would never evaluate.
    fake = RecordingHistoryTradier(
        {
            "MRVL": _daily_bars(signal_date, underlying_prices),
            "SOXX": _daily_bars(signal_date, soxx_prices),
        }
    )
    reward_cfg = {"horizons": [90], "benchmark_symbol": "SPY"}
    updated = reward_evaluator.evaluate_signals(
        [signal], fake, reward_cfg, current_emergence_scores=None,
        path="/tmp/does_not_matter_signals_bench.json",
    )
    assert "SOXX" in fake.requested_symbols
    assert "SPY" not in fake.requested_symbols
    assert "90" in updated[0]["horizon_evals"]


def test_data_quality_gate_blocks():
    from src.config import load_settings

    settings = load_settings()
    sparse_digest = {
        "edgar_capex": {"source": "edgar_capex", "companies": {}, "aggregate_capex_yoy_pct": None},
        "github_trends": {"source": "github_trends", "top_new_repos": [], "stage_heat": {}},
        "jobs_hn": {"source": "jobs_hn", "stage_job_counts": {}, "stage_job_mom_change": {}, "total_comments": 0},
        "arxiv_trends": {"source": "arxiv_trends", "stage_paper_counts": {}, "sample_hot_titles": []},
        "hn_buzz": {"source": "hn_buzz", "stage_buzz": {}},
        "edgar_fts": {"source": "edgar_fts", "theme_counts": {}},
    }
    dq = compute_data_quality(sparse_digest)
    assert dq < settings.min_data_quality


def test_cluster_gate_helper():
    scored = [
        {"ticker": "AAA", "stage_id": 3, "total_score": 80.0},
        {"ticker": "BBB", "stage_id": 5, "total_score": 70.0},
    ]
    # Only AAA shares stage_id=3 and clears the bar -> 1 member, below the
    # default min of 2 -> the clustering gate would fire.
    count = _cluster_member_count(scored, stage_id=3, theme_id=None, bar=45.0)
    assert count == 1

    scored_with_cluster = scored + [{"ticker": "CCC", "stage_id": 3, "total_score": 50.0}]
    count2 = _cluster_member_count(scored_with_cluster, stage_id=3, theme_id=None, bar=45.0)
    assert count2 == 2

    # A candidate below the score bar does not count toward the cluster.
    scored_weak = scored + [{"ticker": "DDD", "stage_id": 3, "total_score": 10.0}]
    count3 = _cluster_member_count(scored_weak, stage_id=3, theme_id=None, bar=45.0)
    assert count3 == 1

    # theme_id match also counts, independent of stage_id.
    themed = [
        {"ticker": "AAA", "stage_id": 3, "total_score": 80.0, "discovery": {"theme_id": "dc_cooling"}},
        {"ticker": "EEE", "stage_id": 7, "total_score": 60.0, "theme_id": "dc_cooling"},
    ]
    count4 = _cluster_member_count(themed, stage_id=3, theme_id="dc_cooling", bar=45.0)
    assert count4 == 2


def test_mailer_renders_no_signal_reason():
    result = {
        "stages_config": [],
        "current_stage": None,
        "next_stage": None,
        "stage_reasoning": "",
        "top_pick": None,
        "top5": [],
        "track_record": {"closed": 0, "open": 0, "hits": 0, "hit_rate": None, "avg_pnl_pct": None},
        "no_signal_reason": "Datenqualität zu niedrig (30.0/100, Minimum 45) — Signal unterdrückt.",
    }
    subject, html = build_email(result)
    assert "Datenqualität zu niedrig" in html


def test_mailer_renders_invalidation():
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
            "benchmark_symbol": "XLU",
            "invalidation": {
                "below_50dma": 105.5,
                "theme_score_drop": "Emergence-Score des Themas fällt in 2 Folgeläufen",
                "note": "These ungültig, wenn Underlying nachhaltig unter 50-Tage-Linie schließt.",
            },
        },
        "top5": [],
        "track_record": {"closed": 0, "open": 0, "hits": 0, "hit_rate": None, "avg_pnl_pct": None},
    }
    subject, html = build_email(result)
    assert "ungültig" in html
    assert "XLU" in html


# ---------------------------------------------------------------------------
# Probabilistic phase model (#5) & machine-checkable claims (#18)
# ---------------------------------------------------------------------------

def test_stage_distribution_sums_to_one():
    from src.config import load_settings

    settings = load_settings()
    result = phases.compute_stage_distribution(FIXTURE_DIGEST, settings.stages)

    # Rounded to 4 decimals for the injected digest -> allow a small tolerance.
    total_prob = sum(result["probabilities"].values())
    assert total_prob == pytest.approx(1.0, abs=1e-3)

    # FIXTURE_DIGEST's strongest per-stage signal (heat, jobs, arxiv, buzz,
    # capex) is concentrated on stage 3 -> it should be the most active stage.
    assert result["current_stage"] == 3
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["next_stage"] in result["probabilities"]


def test_stage_distribution_empty_digest():
    from src.config import load_settings

    settings = load_settings()
    result = phases.compute_stage_distribution({}, settings.stages)

    total_prob = sum(result["probabilities"].values())
    assert total_prob == pytest.approx(1.0, abs=1e-3)

    n = len(settings.stages)
    for prob in result["probabilities"].values():
        assert prob == pytest.approx(1.0 / n, abs=1e-3)

    assert result["confidence"] < 0.2  # no data at all -> low confidence
    assert result["current_stage"] is not None  # still picks a (arbitrary-tie) stage, never crashes


def test_stage_distribution_empty_stages_list_never_crashes():
    result = phases.compute_stage_distribution(FIXTURE_DIGEST, [])
    assert result["probabilities"] == {}
    assert result["current_stage"] is None
    assert result["next_stage"] is None
    assert result["confidence"] == 0.0


def test_verify_claims_fraction():
    strong_candidate = {
        "ticker": "VRT",
        "stage_id": 3,
        "claims": [
            {"source": "jobs_hn", "direction": "up", "reason": "hiring accelerating"},
            {"source": "hn_buzz", "direction": "high", "reason": "buzz elevated"},
            {"source": "edgar_capex", "direction": "up", "reason": "capex rising"},
        ],
    }
    cv_strong = phases.verify_claims(strong_candidate, FIXTURE_DIGEST, {})
    assert cv_strong["total"] == 3
    assert cv_strong["verified"] == 3
    assert cv_strong["fraction"] == pytest.approx(1.0)

    # stage_id=1 has zero per-stage activity anywhere in FIXTURE_DIGEST, so
    # claims about it should fail to verify -> a lower fraction than above.
    weak_candidate = {
        "ticker": "ZZZ",
        "stage_id": 1,
        "claims": [
            {"source": "jobs_hn", "direction": "up", "reason": "hiring accelerating"},
            {"source": "github_trends", "direction": "high", "reason": "repos active"},
        ],
    }
    cv_weak = phases.verify_claims(weak_candidate, FIXTURE_DIGEST, {})
    assert cv_weak["total"] == 2
    assert cv_weak["verified"] == 0
    assert cv_weak["fraction"] < cv_strong["fraction"]

    no_claims_candidate = {"ticker": "AAA", "stage_id": 3}
    cv_none = phases.verify_claims(no_claims_candidate, FIXTURE_DIGEST, {})
    assert cv_none["total"] == 0
    assert cv_none["fraction"] == pytest.approx(1.0)


def test_claims_factor_downweights():
    # Two identical candidates differing only in their claims: GOOD claims a
    # source that genuinely verifies against FIXTURE_DIGEST for stage 3
    # (edgar_capex/up: aggregate_capex_yoy_pct=25.0 > 0), BAD claims a source
    # that doesn't (arxiv_trends/high: stage 3's paper count 8 is below the
    # {3: 8, 4: 20} median of 14).
    good = {"ticker": "GOOD", "stage_id": 3, "total_score": 80.0,
            "claims": [{"source": "edgar_capex", "direction": "up"}]}
    bad = {"ticker": "BAD", "stage_id": 3, "total_score": 80.0,
           "claims": [{"source": "arxiv_trends", "direction": "high"}]}

    adjusted = apply_claims_adjustment([good, bad], FIXTURE_DIGEST, {}, claims_conviction_floor=0.5)

    good_after = next(c for c in adjusted if c["ticker"] == "GOOD")
    bad_after = next(c for c in adjusted if c["ticker"] == "BAD")

    assert good_after["claims_verified"]["fraction"] == pytest.approx(1.0)
    assert bad_after["claims_verified"]["fraction"] == pytest.approx(0.0)
    assert good_after["total_score"] == pytest.approx(80.0)  # factor 1.0 -> unchanged
    assert bad_after["total_score"] == pytest.approx(40.0)  # factor 0.5 -> halved
    assert bad_after["total_score"] < good_after["total_score"]
    # Re-sorted descending by the adjusted total_score.
    assert adjusted[0]["ticker"] == "GOOD"


def test_mailer_renders_stage_distribution():
    result = {
        "stages_config": [
            {"id": 2, "name": "Datacenter-Infrastruktur"},
            {"id": 3, "name": "Energie / Kühlung / Netz"},
        ],
        "current_stage": 3,
        "next_stage": 3,
        "stage_reasoning": "Capex bleibt erhöht, Energie-Engpässe dominieren zunehmend.",
        "llm_current_stage": 2,
        "llm_next_stage": 3,
        "stage_distribution": {
            "probabilities": {2: 0.3, 3: 0.7},
            "activity": {2: 0.2, 3: 0.6},
            "momentum": {2: 1.0, 3: 5.0},
            "current_stage": 3,
            "next_stage": 3,
            "confidence": 0.62,
        },
        "top_pick": None,
        "top5": [],
        "track_record": {"closed": 0, "open": 0, "hits": 0, "hit_rate": None, "avg_pnl_pct": None},
    }
    subject, html = build_email(result)
    assert "%" in html
    assert "Kühlung" in html
    assert "LLM-Einschätzung" in html


def test_mailer_renders_claims_verified():
    result = {
        "stages_config": [{"id": 3, "name": "Energie / Kühlung / Netz"}],
        "current_stage": 3,
        "next_stage": 3,
        "stage_reasoning": "Test reasoning",
        "top_pick": {
            "ticker": "VRT",
            "total_score": 82.5,
            "thesis": "Cooling demand thesis",
            "option": {"strike": 120, "expiration": "2026-11-20", "mid": 5.25, "dte": 136, "delta": 0.65},
            "claims_verified": {"verified": 2, "total": 3, "fraction": 0.667},
        },
        "top5": [],
        "track_record": {"closed": 0, "open": 0, "hits": 0, "hit_rate": None, "avg_pnl_pct": None},
    }
    subject, html = build_email(result)
    assert "Belege geprüft: 2/3" in html


def test_config_claims_conviction_floor_default():
    from src.config import load_settings

    settings = load_settings()
    assert settings.claims_conviction_floor == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SEC-filing language acceleration (#8) & deceleration filter free parts (#10)
# ---------------------------------------------------------------------------

def test_edgar_language_offline():
    # NXT_OFFLINE=1 is set at module import time (see top of file) -> must
    # short-circuit before any network call and return the empty structure.
    result = edgar_language.collect(["MSFT"], ["backlog"])
    assert result == {"source": "edgar_language", "companies": {}, "aggregate": {}}


def test_edgar_language_phrase_delta():
    latest_text = "we see strong backlog and ai demand growing. backlog backlog"
    prior_text = "backlog was modest last quarter"
    deltas = edgar_language._phrase_deltas(latest_text, prior_text, ["backlog", "ai demand", "liquid cooling"])

    assert deltas["backlog"] == {"latest": 3, "prior": 1, "delta": 2}
    assert deltas["ai demand"] == {"latest": 1, "prior": 0, "delta": 1}
    assert deltas["liquid cooling"] == {"latest": 0, "prior": 0, "delta": 0}


def test_edgar_language_html_to_text_strips_tags_and_entities():
    raw = "<html><body><p>Backlog &amp; capacity constrained</p></body></html>"
    text = edgar_language._html_to_text(raw)
    assert "<" not in text
    assert "backlog & capacity constrained" in text


def test_config_edgar_language_defaults():
    from src.config import load_settings

    settings = load_settings()
    cfg = settings.edgar_language_config
    assert cfg.get("enabled") is True
    assert "MSFT" in cfg.get("companies", [])
    assert "backlog" in cfg.get("phrases", [])


def test_relative_strength_decel():
    # Case 1: strong prior climb, then the last 21 days go flat -> short-window
    # RS rolls over below the long-window RS -> decelerating True.
    decel_underlying = [100 + i * (100 / 42) for i in range(43)] + [200.0] * 21
    flat_benchmark = [50.0] * 64

    result = insider.relative_strength_decel(decel_underlying, flat_benchmark)
    assert result is not None
    assert result["rs_21"] < result["rs_63"]
    assert result["decelerating"] is True

    # Case 2: weak drift over the full quarter, but the last 21 days accelerate
    # hard -> short-window RS clears the long-window RS -> decelerating False.
    accel_underlying = [250 - i * (100 / 42) for i in range(43)] + [150 + i * (150 / 21) for i in range(1, 22)]

    result2 = insider.relative_strength_decel(accel_underlying, flat_benchmark)
    assert result2 is not None
    assert result2["rs_21"] > result2["rs_63"]
    assert result2["decelerating"] is False

    # Case 3: too little history for the long window -> None.
    assert insider.relative_strength_decel([100.0, 101.0, 102.0], [50.0, 51.0, 52.0]) is None
    assert insider.relative_strength_decel(None, [50.0] * 64) is None
    assert insider.relative_strength_decel([100.0] * 64, []) is None


def test_insider_signal_shape(monkeypatch):
    from datetime import date

    # This test exercises the (mocked) network path, so temporarily lift the
    # module-wide NXT_OFFLINE=1 short-circuit for just this test.
    monkeypatch.delenv("NXT_OFFLINE", raising=False)

    fake_submissions = {
        "filings": {
            "recent": {
                "form": ["4", "10-Q", "4"],
                "accessionNumber": ["0001-24-000111", "0001-24-000112", "0001-24-000113"],
                "primaryDocument": ["form4_a.xml", "10q.htm", "form4_b.xml"],
                "filingDate": [date.today().isoformat()] * 3,
            }
        }
    }
    fake_sell_xml = (
        "<nonDerivativeTransaction><transactionCode>S</transactionCode>"
        "<transactionShares><value>1,000</value></transactionShares></nonDerivativeTransaction>"
    )

    monkeypatch.setattr(insider, "get_json", lambda url, **kw: fake_submissions)
    monkeypatch.setattr(insider, "get_text", lambda url, **kw: fake_sell_xml)

    result = insider.insider_selling_signal("XYZ", lambda t: 12345)
    assert result == {"recent_form4": 2, "net_sell_hint": "sell"}

    # No CIK resolvable -> None (fault-tolerant, not a hard error).
    assert insider.insider_selling_signal("XYZ", lambda t: None) is None

    # cik_resolver raising -> None, never propagates.
    def _raising_resolver(_ticker):
        raise RuntimeError("boom")

    assert insider.insider_selling_signal("XYZ", _raising_resolver) is None


def test_insider_signal_offline_short_circuits(monkeypatch):
    # NXT_OFFLINE=1 is already set process-wide for this test module; make sure
    # insider_selling_signal never even calls the resolver.
    calls = []

    def _resolver(t):
        calls.append(t)
        return 1

    assert insider.insider_selling_signal("XYZ", _resolver) is None
    assert calls == []


def test_mailer_renders_filing_language():
    result = {
        "stages_config": [],
        "current_stage": None,
        "next_stage": None,
        "stage_reasoning": "",
        "top_pick": None,
        "top5": [],
        "track_record": {"closed": 0, "open": 0, "hits": 0, "hit_rate": None, "avg_pnl_pct": None},
        "edgar_language": {
            "source": "edgar_language",
            "companies": {},
            "aggregate": {"liquid cooling": 12, "backlog": 5, "ai demand": -2},
        },
    }
    subject, html = build_email(result)
    assert "liquid cooling" in html
    assert "Vorquartal" in html
    assert "backlog" in html


def test_mailer_skips_filing_language_when_no_positive_delta():
    result = {
        "stages_config": [],
        "current_stage": None,
        "next_stage": None,
        "stage_reasoning": "",
        "top_pick": None,
        "top5": [],
        "track_record": {"closed": 0, "open": 0, "hits": 0, "hit_rate": None, "avg_pnl_pct": None},
        "edgar_language": {"source": "edgar_language", "companies": {}, "aggregate": {"backlog": 0, "ai demand": -1}},
    }
    subject, html = build_email(result)
    assert "Filing-Sprache" not in html
