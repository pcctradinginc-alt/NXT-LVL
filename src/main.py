"""Pipeline orchestration and CLI entry point.

Run with:
    python -m src.main               # live run (requires all secrets)
    python -m src.main --dry-run     # collectors run for real (free APIs),
                                      # LLM/Tradier/mail are stubbed/skipped

Env NXT_OFFLINE=1 makes every collector short-circuit to an empty result
immediately (no network calls at all) — used by CI smoke tests.

Exit code is always 0 for a clean run, including "no signal today". Only
hard failures (e.g. cannot even load config) result in a non-zero exit code.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from src import tracking
from src.analysis import llm, scoring
from src.collectors import arxiv_trends, edgar_capex, edgar_fts, github_trends, hn_buzz, jobs_hn
from src.config import DATA_DIR, Settings, load_settings
from src.emergence import baseline as baseline_mod
from src.emergence import detector as emergence_detector
from src.mailer import build_email, send
from src.options.tradier import TradierClient
from src.reward import engine as reward_engine
from src.reward import evaluator as reward_evaluator
from src.reward import weights as weights_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nxt_lvl.main")

LAST_DIGEST_PATH = DATA_DIR / "last_digest.json"
LAST_EMAIL_PATH = DATA_DIR / "last_email.html"
BASELINE_PATH = DATA_DIR / "baseline.json"
WEIGHTS_PATH = DATA_DIR / "weights.json"
DIGEST_HISTORY_PATH = DATA_DIR / "digest_history.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NXT LVL — AI Next-Stage Beneficiary Scanner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run collectors for real but stub the LLM, skip Tradier calls, "
            "and write the email to data/last_email.html instead of sending it."
        ),
    )
    return parser.parse_args(argv)


def run_collectors(settings: Settings) -> dict[str, Any]:
    """Run all five collectors sequentially, building the compact digest."""
    stage_keywords = settings.all_keywords()
    digest: dict[str, Any] = {}

    logger.info("Running collector: edgar_capex")
    try:
        digest["edgar_capex"] = edgar_capex.collect(settings.capex_companies)
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_capex collector crashed unexpectedly: %s", exc)
        digest["edgar_capex"] = {"source": "edgar_capex", "companies": {}, "aggregate_capex_yoy_pct": None}

    logger.info("Running collector: github_trends")
    try:
        digest["github_trends"] = github_trends.collect(stage_keywords)
    except Exception as exc:  # noqa: BLE001
        logger.warning("github_trends collector crashed unexpectedly: %s", exc)
        digest["github_trends"] = {"source": "github_trends", "top_new_repos": [], "stage_heat": {}}

    logger.info("Running collector: jobs_hn")
    try:
        digest["jobs_hn"] = jobs_hn.collect(stage_keywords)
    except Exception as exc:  # noqa: BLE001
        logger.warning("jobs_hn collector crashed unexpectedly: %s", exc)
        digest["jobs_hn"] = {"source": "jobs_hn", "stage_job_counts": {}, "stage_job_mom_change": {}, "total_comments": 0}

    logger.info("Running collector: arxiv_trends")
    try:
        digest["arxiv_trends"] = arxiv_trends.collect(stage_keywords)
    except Exception as exc:  # noqa: BLE001
        logger.warning("arxiv_trends collector crashed unexpectedly: %s", exc)
        digest["arxiv_trends"] = {"source": "arxiv_trends", "stage_paper_counts": {}, "sample_hot_titles": []}

    logger.info("Running collector: hn_buzz")
    try:
        digest["hn_buzz"] = hn_buzz.collect(stage_keywords)
    except Exception as exc:  # noqa: BLE001
        logger.warning("hn_buzz collector crashed unexpectedly: %s", exc)
        digest["hn_buzz"] = {"source": "hn_buzz", "stage_buzz": {}}

    logger.info("Running collector: edgar_fts")
    try:
        emergence_cfg = settings.emergence_config
        fts_cfg = emergence_cfg.get("edgar_fts", {}) or {}
        digest["edgar_fts"] = edgar_fts.collect(
            settings.themes,
            lookback_days=fts_cfg.get("lookback_days", 90),
            forms=fts_cfg.get("forms", "10-K,10-Q,8-K"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_fts collector crashed unexpectedly: %s", exc)
        digest["edgar_fts"] = {"source": "edgar_fts", "theme_counts": {}}

    return digest


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def append_digest_history(
    path: Path,
    *,
    current_stage: int | None,
    next_stage: int | None,
    top_pick: dict[str, Any] | None,
    scored_candidates: list[dict[str, Any]],
    all_theme_scores: dict[str, float] | None,
    emergent_themes: list[dict[str, Any]] | None,
    run_date: str | None = None,
) -> None:
    """Append one compact JSON line summarizing this run to `path`.

    This is the substrate for a real FORWARD backtest: a true retroactive
    backtest of the collector-driven signal logic is impossible (the free
    data sources are point-in-time and were never archived), but archiving
    each run's scores now lets src/backtest/calibrate.py measure, once
    enough history has accumulated, which scoring components actually
    predicted forward returns.

    Deliberately compact (a few KB per line, NOT the full raw digest) so the
    file stays cheap to commit daily. Fault-tolerant: any failure here is
    logged and swallowed, never allowed to break the pipeline. Runs in ALL
    modes, including dry-run, so offline tests exercise this path too.
    """
    try:
        record = {
            "date": run_date or date.today().isoformat(),
            "current_stage": current_stage,
            "next_stage": next_stage,
            "top_pick": top_pick.get("ticker") if top_pick else None,
            "candidates": [
                {
                    "ticker": c.get("ticker"),
                    "total_score": c.get("total_score"),
                    "scores": {
                        k: (c.get("scores") or {}).get(k)
                        for k in (
                            "breadth",
                            "momentum",
                            "stage_fit",
                            "divergence",
                            "option_quality",
                            "emergence",
                        )
                    },
                    "source_count": c.get("source_count"),
                }
                for c in scored_candidates
            ],
            "all_theme_scores": all_theme_scores or {},
            "emergent_themes": [t.get("theme_id") for t in (emergent_themes or []) if t.get("theme_id")],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("append_digest_history failed, continuing: %s", exc)


DIGEST_SOURCES = ["edgar_capex", "github_trends", "jobs_hn", "arxiv_trends", "hn_buzz", "edgar_fts"]


def compute_data_quality(digest: dict[str, Any]) -> float:
    """Deterministic 0-100 data-quality score from the digest.

    Combines: share of sources that returned any non-empty data, whether
    edgar_capex has a usable aggregate figure, and HN "who is hiring" comment
    volume (a proxy for whether that collector was rate-limited/degraded).
    """
    active = 0
    for source in DIGEST_SOURCES:
        block = digest.get(source)
        if not block:
            continue
        has_data = any(v for k, v in block.items() if k != "source" and v not in (None, {}, [], "", 0))
        if has_data:
            active += 1
    source_completeness = active / len(DIGEST_SOURCES) * 100

    edgar = digest.get("edgar_capex") or {}
    capex_ok = 100.0 if edgar.get("aggregate_capex_yoy_pct") is not None else 40.0

    jobs = digest.get("jobs_hn") or {}
    total_comments = jobs.get("total_comments", 0) or 0
    comment_volume_score = max(0.0, min(100.0, total_comments / 500 * 100))

    return round((source_completeness + capex_ok + comment_volume_score) / 3, 1)


def compute_risks(
    top_pick: dict[str, Any] | None,
    option: dict[str, Any] | None,
    emergence_result: dict[str, Any] | None,
    overheated_threshold: float = 80.0,
    data_quality_score: float | None = None,
) -> list[str]:
    """Compute automatic risk flags for the report (German user-facing text)."""
    risks: list[str] = []
    if top_pick is None:
        return risks

    scores = top_pick.get("scores", {}) or {}
    divergence = scores.get("divergence")
    if isinstance(divergence, (int, float)) and divergence < 40:
        risks.append("Divergenz niedrig — der Titel ist bereits stark gelaufen, ein Großteil der Bewegung könnte eingepreist sein.")

    source_count = top_pick.get("source_count", 0)
    if isinstance(source_count, (int, float)) and source_count <= 1:
        risks.append("Nur eine unabhängige Quelle bestätigt diesen Kandidaten — erhöhtes Einzelquellen-Risiko.")

    if option is None:
        risks.append("Keine passende Option gefunden — Signal beruht nur auf der Aktie, keine gehebelte Positionierung möglich.")
    else:
        oi = option.get("open_interest")
        spread_pct = option.get("spread_pct")
        if (isinstance(oi, (int, float)) and oi < 100) or (isinstance(spread_pct, (int, float)) and spread_pct > 0.10):
            risks.append("Options-Liquidität schlecht (niedriges Open Interest oder breiter Spread).")

    discovery = top_pick.get("discovery") or {}
    theme_id = discovery.get("theme_id")
    if theme_id and emergence_result:
        theme_score = (emergence_result.get("all_theme_scores") or {}).get(theme_id)
        if isinstance(theme_score, (int, float)) and theme_score >= overheated_threshold:
            risks.append(f"Thema '{theme_id}' ist bereits überhitzt (Emergence Score {theme_score} ≥ {overheated_threshold}).")

        emergent_theme_info = next(
            (t for t in (emergence_result.get("emergent_themes") or []) if t.get("theme_id") == theme_id), None
        )
        if emergent_theme_info and emergent_theme_info.get("novelty", 0) >= 0.8:
            risks.append("Hohe Novelty — das Thema ist noch unbewährt und hat kaum historische Vergleichswerte.")

    if isinstance(data_quality_score, (int, float)) and data_quality_score < 50:
        risks.append(f"Schwache Datenqualität in diesem Lauf (Score {data_quality_score}/100).")

    return risks


def build_result(
    settings: Settings,
    digest: dict[str, Any],
    llm_result: dict[str, Any],
    top_pick: dict[str, Any] | None,
    top5: list[dict[str, Any]],
    track_record: dict[str, Any],
    emergent_themes: list[dict[str, Any]] | None = None,
    reward_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stages_config": settings.stages,
        "current_stage": llm_result.get("current_stage"),
        "next_stage": llm_result.get("next_stage"),
        "stage_reasoning": llm_result.get("reasoning", ""),
        "top_pick": top_pick,
        "top5": top5,
        "track_record": track_record,
        "emergent_themes": emergent_themes or [],
        "reward": reward_status,
    }


def hit_rate_by_horizon(signals: list[dict[str, Any]], horizons: list[int]) -> dict[str, float | None]:
    """Compute hit rate (%) per horizon across all signals with a filled horizon_eval."""
    result: dict[str, float | None] = {}
    for horizon in horizons:
        key = str(horizon)
        evals = [
            (s.get("horizon_evals") or {}).get(key)
            for s in signals
            if (s.get("horizon_evals") or {}).get(key)
        ]
        if not evals:
            result[key] = None
            continue
        hits = sum(1 for e in evals if e.get("hit"))
        result[key] = round(hits / len(evals) * 100, 1)
    return result


def run(dry_run: bool = False) -> int:
    settings = load_settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== NXT LVL run started (dry_run=%s, offline=%s) ===", dry_run, settings.offline)

    # 1. Collect (includes edgar_fts, appended fault-tolerantly like the rest)
    digest = run_collectors(settings)

    # 2. Emergence detection: baseline load -> detect -> baseline save.
    emergence_cfg = settings.emergence_config
    baseline_obj = baseline_mod.load(BASELINE_PATH)
    try:
        emergence_result = emergence_detector.detect(
            digest,
            settings.themes,
            emergence_cfg,
            baseline_obj,
            entity_aliases=settings.entity_aliases,
            megacap_exclude=settings.megacap_exclude,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.detector.detect failed, continuing without emergence data: %s", exc)
        emergence_result = {"emergent_themes": [], "emergent_candidates": [], "all_theme_scores": {}}
    baseline_mod.save(baseline_obj, BASELINE_PATH)

    watchlist_tickers = settings.watchlist_tickers()
    for cand in emergence_result.get("emergent_candidates", []):
        cand["in_watchlist"] = cand.get("ticker") in watchlist_tickers

    # Compact summary handed to the LLM prompt (via the digest, see module
    # docstring note below) so it can also propose brand-new tickers for the
    # top emergent themes — a second path to candidates outside the watchlist.
    digest["emergence_summary"] = [
        {
            "theme_id": t["theme_id"],
            "name": t["name"],
            "emergence_score": t["emergence_score"],
            "drivers": t["drivers"],
        }
        for t in emergence_result.get("emergent_themes", [])[:3]
    ]

    write_json(LAST_DIGEST_PATH, digest)
    logger.info("Digest written to %s", LAST_DIGEST_PATH)

    # 3. Tradier client (used for tracking evaluation, divergence, and option selection)
    tradier_client: TradierClient | None = None
    if not dry_run and settings.tradier_api_key:
        tradier_client = TradierClient(settings.tradier_api_key, settings.tradier_env)

    # 4. Evaluate open signals (legacy per-signal quote checkpoints; skip in dry-run)
    if not dry_run and tradier_client is not None:
        tracking_cfg = settings.tracking_config
        try:
            tracking.evaluate_open_signals(
                tradier_client,
                close_after_trading_days=tracking_cfg.get("close_after_trading_days", 60),
                close_at_dte=tracking_cfg.get("close_at_dte", 40),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("evaluate_open_signals failed, continuing: %s", exc)
    else:
        logger.info("Skipping evaluate_open_signals (dry-run or no Tradier key)")

    # 5. Reward Engine: load weights, retroactively evaluate signals + update
    # weights in real (non-dry-run) runs; dry-run just uses whatever weights
    # are currently on disk (or config defaults) without mutating them via Tradier data.
    reward_cfg = settings.reward_config
    weights_obj = weights_mod.load(
        WEIGHTS_PATH,
        defaults_feature=settings.scoring_weights,
        defaults_reliability={src: 1.0 for src in scoring.ALL_SOURCES},
    )

    if not dry_run and tradier_client is not None:
        try:
            signals_for_eval = tracking.load_signals()
            reward_evaluator.evaluate_signals(
                signals_for_eval,
                tradier_client,
                reward_cfg,
                current_emergence_scores=emergence_result.get("all_theme_scores"),
            )
            primary_horizon = int(reward_cfg.get("primary_horizon", 90))
            overheated_threshold = float(reward_cfg.get("overheated_score_threshold", 80))
            newly = reward_engine.accumulate_ledger(
                weights_obj, signals_for_eval, primary_horizon, overheated_threshold
            )
            logger.info("Reward engine: %d newly-matured signal evaluation(s) consumed into the ledger", newly)
            # recompute_weights is deterministic/convergent from the cumulative
            # ledger, so it is safe (and idempotent once converged) to call
            # every run, even when `newly == 0`.
            weights_obj = reward_engine.recompute_weights(
                weights_obj,
                reward_cfg,
                base_feature_weights=settings.scoring_weights,
                base_reliability={src: 1.0 for src in scoring.ALL_SOURCES},
            )
            weights_mod.save(weights_obj, WEIGHTS_PATH)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reward engine evaluation/update failed, continuing: %s", exc)
    else:
        logger.info("Skipping reward evaluation/update (dry-run or no Tradier key)")

    effective_weights = weights_mod.get_effective_weights(settings.scoring_weights, WEIGHTS_PATH)
    effective_reliability = weights_mod.current_reliability(weights_obj)
    all_theme_scores = emergence_result.get("all_theme_scores", {})

    # 6. LLM analysis (digest now includes emergence_summary, see step 2)
    if dry_run:
        logger.info("Dry-run: using LLM stub instead of a real Claude (Haiku) call")
        llm_result = llm.dry_run_stub()
    else:
        llm_result = llm.analyze(digest, settings.anthropic_api_key)

    llm_candidates = llm_result.get("candidates", [])
    next_stage = llm_result.get("next_stage")

    # Normalize LLM-proposed tickers to uppercase so downstream lookups
    # (watchlist tagging, theme mapping, perf_lookup, option selection) match
    # the uppercase convention used by the watchlist and theme ticker lists.
    for cand in llm_candidates:
        if cand.get("ticker"):
            cand["ticker"] = str(cand["ticker"]).strip().upper()

    # Tag LLM candidates with discovery metadata (watchlist vs. llm-proposed).
    for cand in llm_candidates:
        cand.setdefault(
            "discovery",
            {"via": "watchlist" if cand.get("ticker") in watchlist_tickers else "llm"},
        )

    # 7. Merge LLM candidates with emergent candidates (dedupe per ticker).
    merged_by_ticker: dict[str, dict[str, Any]] = {}
    for cand in llm_candidates:
        ticker = cand.get("ticker")
        if ticker:
            merged_by_ticker[ticker] = cand

    for emergent in emergence_result.get("emergent_candidates", []):
        ticker = emergent.get("ticker")
        if not ticker:
            continue
        theme_id = emergent.get("theme_id")
        theme_name = next(
            (t.get("name") for t in settings.themes if t.get("id") == theme_id), theme_id
        )
        discovery = {
            "via": "emergence",
            "theme_id": theme_id,
            "theme": theme_name,
            "drivers": emergent.get("drivers", {}),
            "confirming_sources": emergent.get("sources", []),
        }
        if ticker in merged_by_ticker:
            # Already proposed by the LLM/watchlist path — enrich with discovery info.
            merged_by_ticker[ticker].setdefault("discovery", discovery)
            merged_by_ticker[ticker]["discovery"].setdefault("theme_id", theme_id)
        else:
            merged_by_ticker[ticker] = {
                "ticker": ticker,
                "stage_id": None,
                "thesis": f"Emergentes Thema '{theme_name}' beschleunigt (Score {emergent.get('emergence_score')}).",
                "source_evidence": emergent.get("sources", []),
                "conviction": 0.6,
                "theme_id": theme_id,
                "discovery": discovery,
            }

    candidates = list(merged_by_ticker.values())

    # 7b. Map candidates with no theme_id to the highest-scoring emergent
    # theme whose ticker list contains them, so score_emergence can pick up
    # a real (non-neutral) theme score during the pre-gate ranking pass
    # instead of only after a top_pick has already been chosen. Done for all
    # runs (dry-run too) — harmless, and keeps offline behavior consistent;
    # it only actually changes anything once all_theme_scores is populated.
    for cand in candidates:
        has_theme_id = cand.get("theme_id") is not None or (
            isinstance(cand.get("discovery"), dict) and cand["discovery"].get("theme_id") is not None
        )
        if has_theme_id:
            continue
        ticker = cand.get("ticker")
        if not ticker:
            continue
        matching_theme_ids = [
            theme.get("id")
            for theme in settings.themes
            if ticker in (theme.get("tickers") or [])
        ]
        if not matching_theme_ids:
            continue
        best_theme_id = max(
            matching_theme_ids,
            key=lambda tid: all_theme_scores.get(tid, 0.0),
        )
        cand["theme_id"] = best_theme_id

    # 7c. Build perf_lookup (real 3-month performance) BEFORE the ranking
    # pass so divergence is real during the signal gate, not just after a
    # top_pick has already been chosen. Fault-tolerant: any failed lookup
    # stores None, which score_divergence treats as neutral (50). Capped at
    # 15 tickers to bound API calls. Skipped in dry-run / without Tradier.
    perf_lookup: dict[str, float | None] = {}
    if not dry_run and tradier_client is not None:
        tickers_to_fetch = [c.get("ticker") for c in candidates if c.get("ticker")][:15]
        for ticker in tickers_to_fetch:
            try:
                perf_lookup[ticker] = tradier_client.get_three_month_performance_pct(ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning("3-month performance pre-fetch failed for %s: %s", ticker, exc)
                perf_lookup[ticker] = None
        logger.info(
            "Pre-gate perf_lookup: fetched 3-month performance for %d/%d candidates",
            sum(1 for v in perf_lookup.values() if v is not None),
            len(tickers_to_fetch),
        )

    # 8. Scoring (pre-option-selection pass, option_quality neutral at 50;
    # divergence real when perf_lookup was populated above)
    scored = scoring.score_candidates(
        candidates,
        digest,
        next_stage,
        weights=effective_weights,
        perf_lookup=perf_lookup,
        all_theme_scores=all_theme_scores,
        reliability=effective_reliability,
    )

    # 9. Select top candidate (score >= threshold, min_sources satisfied, not in cooldown)
    top_pick: dict[str, Any] | None = None
    signals_list = tracking.load_signals()

    for candidate in scored:
        ticker = candidate.get("ticker")
        if not ticker:
            continue
        if candidate.get("total_score", 0) < settings.signal_threshold:
            break  # sorted descending — nothing further will qualify
        if candidate.get("source_count", 0) < settings.min_sources:
            continue
        if tracking.in_cooldown(ticker, settings.cooldown_days, signals=signals_list):
            logger.info("Candidate %s is in cooldown, skipping", ticker)
            continue
        top_pick = candidate
        break

    # 10. Option selection + divergence rescoring for the chosen top pick
    option = None
    three_month_perf = None
    if top_pick is not None:
        ticker = top_pick["ticker"]
        if dry_run or tradier_client is None:
            logger.info("Skipping Tradier option selection (dry-run or no Tradier key)")
        else:
            opts_cfg = settings.options_config
            try:
                option = tradier_client.select_option(
                    ticker,
                    dte_min=opts_cfg.get("dte_min", 90),
                    dte_max=opts_cfg.get("dte_max", 180),
                    delta_min=opts_cfg.get("delta_min", 0.60),
                    delta_max=opts_cfg.get("delta_max", 0.70),
                    min_open_interest=opts_cfg.get("min_open_interest", 100),
                    max_spread_pct=opts_cfg.get("max_spread_pct", 0.10),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Option selection failed for %s: %s", ticker, exc)
                option = None

            if ticker in perf_lookup:
                # Already fetched during the pre-gate enrichment pass — avoid
                # a redundant Tradier call.
                three_month_perf = perf_lookup[ticker]
            else:
                try:
                    three_month_perf = tradier_client.get_three_month_performance_pct(ticker)
                    perf_lookup[ticker] = three_month_perf
                except Exception as exc:  # noqa: BLE001
                    logger.warning("3-month performance lookup failed for %s: %s", ticker, exc)

            # Re-score just the top pick with real divergence + option quality
            rescored = scoring.score_candidate(
                {k: v for k, v in top_pick.items() if k not in ("scores", "total_score", "source_count")},
                digest,
                next_stage,
                weights=effective_weights,
                three_month_perf_pct=three_month_perf,
                option=option,
                all_theme_scores=all_theme_scores,
                reliability=effective_reliability,
            )
            top_pick.update(rescored)
            # Keep list consistent: replace the entry in `scored` too
            for idx, c in enumerate(scored):
                if c.get("ticker") == ticker:
                    scored[idx] = top_pick
                    break
            scored.sort(key=lambda c: c["total_score"], reverse=True)

        top_pick["option"] = option

        # 11. Risk flags (rendered by the mailer, computed here).
        data_quality_score = compute_data_quality(digest)
        top_pick["risks"] = compute_risks(
            top_pick,
            option,
            emergence_result,
            overheated_threshold=float(reward_cfg.get("overheated_score_threshold", 80)),
            data_quality_score=data_quality_score,
        )

        # Persist the new signal (only for real runs where we'd actually track it)
        if not dry_run:
            entry_mid = option.get("mid") if option else None
            entry_underlying = option.get("underlying_price") if option else None

            benchmark_symbol = reward_cfg.get("benchmark_symbol", "SPY")
            benchmark_at_signal = None
            if tradier_client is not None:
                try:
                    bench_quote = tradier_client.get_quote(benchmark_symbol)
                    if bench_quote:
                        benchmark_at_signal = bench_quote.get("last") or bench_quote.get("close")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Benchmark quote lookup failed for %s: %s", benchmark_symbol, exc)

            discovery = top_pick.get("discovery") or {}
            source_attribution = list(
                set(top_pick.get("source_evidence", [])) & set(scoring.ALL_SOURCES)
            )
            candidate_scores = top_pick.get("scores", {})
            feature_attribution = {
                feature: round(candidate_scores.get(feature, 0) * effective_weights.get(feature, 0), 2)
                for feature in ("breadth", "momentum", "stage_fit", "divergence", "option_quality", "emergence")
                if feature in candidate_scores
            }
            recommended_horizon = option.get("dte") if option else int(reward_cfg.get("primary_horizon", 90))
            emergence_at_signal = all_theme_scores.get(discovery.get("theme_id")) if discovery.get("theme_id") else None

            tracking.add_signal(
                ticker=ticker,
                score=top_pick.get("total_score", 0),
                thesis=top_pick.get("thesis", ""),
                occ_symbol=option.get("occ_symbol") if option else None,
                strike=option.get("strike") if option else None,
                expiration=option.get("expiration") if option else None,
                entry_option_mid=entry_mid,
                entry_underlying=entry_underlying,
                data_sources=source_attribution,
                reasoning=top_pick.get("thesis", ""),
                recommended_horizon_days=recommended_horizon,
                price_at_signal=entry_underlying,
                benchmark_symbol=benchmark_symbol,
                benchmark_at_signal=benchmark_at_signal,
                option_idea=option,
                data_quality_score=data_quality_score,
                source_attribution=source_attribution,
                feature_attribution=feature_attribution,
                discovery=discovery,
                emergence_at_signal=emergence_at_signal,
            )

    # 12. Track record stats
    track_record = tracking.stats()
    reward_status = {
        "feature_weights": weights_mod.current_feature_weights(weights_obj),
        "source_reliability": weights_mod.current_reliability(weights_obj),
        "history": weights_obj.get("history", []),
        "hit_rate_by_horizon": hit_rate_by_horizon(
            tracking.load_signals(), reward_cfg.get("horizons", [30, 60, 90, 180])
        ),
    }

    # 12b. Archive this run's scores to data/digest_history.jsonl — the
    # substrate for a real forward IC calibration once it accumulates (see
    # src/backtest/calibrate.py). Runs in all modes, including dry-run.
    append_digest_history(
        DIGEST_HISTORY_PATH,
        current_stage=llm_result.get("current_stage"),
        next_stage=next_stage,
        top_pick=top_pick,
        scored_candidates=scored,
        all_theme_scores=all_theme_scores,
        emergent_themes=emergence_result.get("emergent_themes", []),
    )

    # 13. Build + send/write email
    result = build_result(
        settings,
        digest,
        llm_result,
        top_pick,
        scored[:5],
        track_record,
        emergent_themes=emergence_result.get("emergent_themes", []),
        reward_status=reward_status,
    )
    subject, html = build_email(result)

    if dry_run:
        LAST_EMAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LAST_EMAIL_PATH, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("Dry-run: email written to %s (subject: %s)", LAST_EMAIL_PATH, subject)
        # Dry-run also persists the emergence/reward artifacts so they can be inspected.
        weights_mod.save(weights_obj, WEIGHTS_PATH)
    else:
        if settings.gmail_app_password and settings.mail_from and settings.mail_to:
            try:
                send(subject, html, settings.mail_from, settings.mail_to, settings.gmail_app_password)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to send email: %s", exc)
        else:
            logger.warning("Mail secrets not fully configured, skipping send and writing to disk instead")
            LAST_EMAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_EMAIL_PATH, "w", encoding="utf-8") as fh:
                fh.write(html)

    logger.info("=== NXT LVL run finished: %s ===", subject)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error in pipeline run: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
