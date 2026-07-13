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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src import tracking
from src.analysis import calibration, earnings
from src.analysis import insider as insider_mod
from src.analysis import iv_rank
from src.analysis import llm, phases, scoring, structures, trend
from src.collectors import arxiv_trends, edgar_capex, edgar_fts, edgar_language, github_trends, hn_buzz, jobs_hn
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
IV_HISTORY_PATH = DATA_DIR / "iv_history.jsonl"

# Deceleration filter, free parts (#10): a top_pick with at least this many
# Form 4 filings in the lookback window is flagged as "heavy insider
# activity" even when the best-effort net-sell direction couldn't be
# determined (recent_form4 count alone is a weaker but still useful proxy).
INSIDER_FORM4_HIGH_THRESHOLD = 5


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

    logger.info("Running collector: edgar_language")
    try:
        lang_cfg = settings.edgar_language_config
        if lang_cfg.get("enabled", True):
            lang_companies = lang_cfg.get("companies") or settings.capex_companies
            lang_phrases = lang_cfg.get("phrases") or edgar_language.DEFAULT_PHRASES
            digest["edgar_language"] = edgar_language.collect(lang_companies, lang_phrases)
        else:
            logger.info("edgar_language collector disabled via config")
            digest["edgar_language"] = {"source": "edgar_language", "companies": {}, "aggregate": {}}
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar_language collector crashed unexpectedly: %s", exc)
        digest["edgar_language"] = {"source": "edgar_language", "companies": {}, "aggregate": {}}

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


def _ticker_tradeable(tradier: Any, ticker: str) -> bool:
    """True if `ticker` resolves to a real Tradier quote with a positive price.

    Filters out LLM-hallucinated company names / invalid symbols (e.g.
    "MOBILEYE" instead of "MBLY") before they become a useless, non-tradeable
    stock-only signal that would also pollute the track record.
    """
    try:
        quote = tradier.get_quote(ticker)
    except Exception:  # noqa: BLE001
        return False
    if not quote:
        return False
    price = quote.get("last") or quote.get("close")
    try:
        return price is not None and float(price) > 0
    except (TypeError, ValueError):
        return False


def _drop_megacaps(candidates: list[dict[str, Any]], exclude: list[str]) -> list[dict[str, Any]]:
    """Filter out any candidate whose ticker is on the mega-cap exclusion list.

    Pure helper backing the mega-cap exclusion gate (#1): the emergence path
    already filters mega-caps via entities.detect_entities(megacap_exclude=...),
    but the LLM proposes candidates independently and, despite the system
    prompt's explicit instruction to avoid current mega-cap winners, has been
    observed to still pick one (e.g. TSLA, which is on this list) — a direct
    contradiction of the "next-stage beneficiaries, NOT current winners"
    mission. This makes the exclusion unconditional in code, regardless of
    what the LLM does. Comparison is case-insensitive; candidates without a
    ticker are kept as-is (nothing to compare).
    """
    exclude_upper = {str(t).upper() for t in (exclude or [])}
    kept: list[dict[str, Any]] = []
    for cand in candidates:
        ticker = cand.get("ticker")
        if ticker and str(ticker).upper() in exclude_upper:
            logger.info("Dropping mega-cap LLM candidate %s (on megacap_exclude list)", ticker)
            continue
        kept.append(cand)
    return kept


def _regime_blocks(risk_on: bool | None) -> bool:
    """Pure predicate for the regime gate (CONCEPT_PROFIT.md Phase D).

    True only when `trend.regime_risk_on(...)` returned a CONFIRMED False
    (risk-off: benchmark below its 200-day SMA). `None` (insufficient or
    unavailable history) fails OPEN — a data gap must never suppress a
    signal on its own, only a measured risk-off regime does.
    """
    return risk_on is False


def _calibration_status(calib: dict[str, Any] | None) -> str:
    """German status word for the no-signal reason / email footer.

    "fehlt" when no calibration file exists at all, "passed=false" when one
    exists but did not show out-of-sample edge, "passed" when it did.
    """
    if calib is None:
        return "fehlt"
    return "passed" if calibration.is_validated(calib) else "passed=false"


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


def _cluster_member_count(
    scored: list[dict[str, Any]],
    stage_id: Any,
    theme_id: str | None,
    bar: float,
) -> int:
    """Count scored candidates that share `stage_id` or `theme_id` and clear `bar`.

    Pure helper backing the signal-clustering gate (#20): a genuinely
    emergent theme/stage should surface more than one credible name (total
    score >= `bar`), not just a single possibly-noisy candidate. Includes the
    candidate itself when it is present in `scored` (it typically is, since
    it was picked FROM `scored`).
    """
    count = 0
    for c in scored:
        if c.get("total_score", 0) < bar:
            continue
        same_stage = stage_id is not None and c.get("stage_id") == stage_id
        c_theme_id = (c.get("discovery") or {}).get("theme_id") or c.get("theme_id")
        same_theme = theme_id is not None and c_theme_id == theme_id
        if same_stage or same_theme:
            count += 1
    return count


def compute_invalidation(
    tradier: Any,
    ticker: str,
    closes: list[float] | None = None,
) -> dict[str, Any] | None:
    """Compute invalidation levels (#14) for a persisted signal.

    Primary trigger is a 50-day simple moving average of the underlying:
    reuses an already-fetched `closes` list (oldest->newest) when available
    and non-empty, otherwise fetches ~70 daily closes via Tradier. Also
    carries qualitative invalidation notes (theme-score drop, earnings
    confirmation). Fault-tolerant: any failure returns None so it never
    breaks the pipeline.
    """
    try:
        usable = list(closes) if closes else []
        if not usable:
            hist_start = (date.today() - timedelta(days=100)).strftime("%Y-%m-%d")
            history = tradier.get_history(ticker, start=hist_start)
            usable = [
                bar.get("close")
                for bar in history
                if isinstance(bar, dict) and bar.get("close") is not None
            ]
        window = usable[-50:] if usable else []
        sma50 = sum(window) / len(window) if window else None
        note = (
            "These ungültig, wenn Underlying nachhaltig unter 50-Tage-Linie schließt, "
            "der Theme-Score abfällt, oder der nächste Earnings-Call den AI-Backlog nicht bestätigt."
        )
        if window and len(window) < 50:
            note += f" (Hinweis: nur {len(window)} Handelstage verfügbar für den Durchschnitt.)"
        return {
            "below_50dma": round(sma50, 2) if sma50 else None,
            "theme_score_drop": "Emergence-Score des Themas fällt in 2 Folgeläufen",
            "note": note,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_invalidation failed for %s: %s", ticker, exc)
        return None


def apply_claims_adjustment(
    scored: list[dict[str, Any]],
    digest: dict[str, Any],
    stage_distribution: dict[str, Any],
    claims_conviction_floor: float = 0.5,
) -> list[dict[str, Any]]:
    """Down-weight each candidate's total_score by how well its
    machine-checkable `claims` (#18) verify against the digest.

    For every candidate: cv = phases.verify_claims(...) is stored as
    candidate["claims_verified"], and total_score is scaled by
    `factor = claims_conviction_floor + (1 - claims_conviction_floor) *
    cv["fraction"]` (stored as candidate["claims_factor"]). A candidate with
    no claims at all is neutral (fraction 1.0 -> factor 1.0, unchanged); one
    whose claims don't hold up against the digest gets scaled down towards
    `claims_conviction_floor`, never zeroed out. Re-sorts `scored` by the
    adjusted total_score, descending. Mutates and returns `scored`.
    """
    for candidate in scored:
        cv = phases.verify_claims(candidate, digest, stage_distribution)
        candidate["claims_verified"] = cv
        factor = claims_conviction_floor + (1 - claims_conviction_floor) * cv["fraction"]
        candidate["claims_factor"] = round(factor, 3)
        candidate["total_score"] = round(candidate.get("total_score", 0.0) * factor, 1)
    scored.sort(key=lambda c: c["total_score"], reverse=True)
    return scored


def build_result(
    settings: Settings,
    digest: dict[str, Any],
    llm_result: dict[str, Any],
    stage_distribution: dict[str, Any],
    next_stage: int | None,
    top_pick: dict[str, Any] | None,
    top5: list[dict[str, Any]],
    track_record: dict[str, Any],
    emergent_themes: list[dict[str, Any]] | None = None,
    reward_status: dict[str, Any] | None = None,
    no_signal_reason: str | None = None,
    edgar_language: dict[str, Any] | None = None,
    observation_mode: bool = False,
    calibration_status: str | None = None,
) -> dict[str, Any]:
    return {
        "stages_config": settings.stages,
        # Code wins (#5): current_stage/next_stage are the deterministic
        # stage-model's picks, not the LLM's. The LLM's own picks + reasoning
        # are carried separately below as a human-readable cross-check.
        "current_stage": stage_distribution.get("current_stage"),
        "next_stage": next_stage,
        "stage_reasoning": llm_result.get("reasoning", ""),
        "stage_distribution": stage_distribution,
        "llm_current_stage": llm_result.get("current_stage"),
        "llm_next_stage": llm_result.get("next_stage"),
        "top_pick": top_pick,
        "top5": top5,
        "track_record": track_record,
        "emergent_themes": emergent_themes or [],
        "reward": reward_status,
        "no_signal_reason": no_signal_reason,
        "edgar_language": edgar_language,
        "observation_mode": observation_mode,
        "calibration_status": calibration_status,
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

    # 2b. Probabilistic phase model (#5): compute the code's own stage
    # distribution BEFORE the LLM call, so the LLM can anchor its candidates
    # on it, and so scoring below uses the deterministic next_stage instead
    # of blindly trusting the LLM's guess. Only a compact subset is injected
    # into the digest (the full dict, including activity/momentum, is kept
    # in the local `stage_distribution` var for scoring/claims/email use).
    stage_distribution = phases.compute_stage_distribution(digest, settings.stages)
    digest["stage_distribution"] = {
        "probabilities": stage_distribution["probabilities"],
        "current_stage": stage_distribution["current_stage"],
        "next_stage": stage_distribution["next_stage"],
        "confidence": stage_distribution["confidence"],
    }

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

    # Phase D validation-gate weights (CONCEPT_PROFIT.md): read once here and
    # reused by the validation gate further below (avoids reading the file
    # twice per run). When src/backtest/optimize.py produced a VALIDATED
    # out-of-sample calibration, its weights_final become the effective
    # scoring weights for this run — merged over the config/reward-engine
    # weights, so any weighted component the calibration doesn't cover keeps
    # its existing config/reward value rather than being dropped.
    calib = calibration.load_calibration()
    calibrated_weights = calibration.validated_weights(calib)
    if calibrated_weights:
        effective_weights = {**effective_weights, **calibrated_weights}
        logger.info(
            "Validation gate: validated calibration weights in effect (from %s): %s",
            calibration.CALIBRATION_PATH,
            calibrated_weights,
        )
    else:
        logger.info(
            "Validation gate: no validated calibration weights — using config/reward-engine weights"
        )

    # 6. LLM analysis (digest now includes emergence_summary, see step 2)
    if dry_run:
        logger.info("Dry-run: using LLM stub instead of a real Claude (Haiku) call")
        llm_result = llm.dry_run_stub()
    else:
        llm_result = llm.analyze(digest, settings.anthropic_api_key)

    llm_candidates = llm_result.get("candidates", [])
    # Code wins for scoring (#5/#18): use the deterministic stage-model's
    # next_stage, not the LLM's own guess. The LLM's current_stage/next_stage
    # /reasoning are still carried through build_result as a cross-check for
    # the email ("LLM-Einschätzung").
    next_stage = stage_distribution["next_stage"]

    # Normalize LLM-proposed tickers to uppercase so downstream lookups
    # (watchlist tagging, theme mapping, perf_lookup, option selection) match
    # the uppercase convention used by the watchlist and theme ticker lists.
    for cand in llm_candidates:
        if cand.get("ticker"):
            cand["ticker"] = str(cand["ticker"]).strip().upper()

    # Mega-cap exclusion gate (#1): enforce settings.megacap_exclude on the
    # LLM's own candidates in code, not just via the system prompt — see
    # _drop_megacaps() docstring for why this is necessary.
    llm_candidates = _drop_megacaps(llm_candidates, settings.megacap_exclude)

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

    # 8b. Machine-checkable-claims verification (#18): dampen each
    # candidate's total_score by how well its claims verify against the
    # digest, BEFORE the top-pick gate below, so an LLM thesis whose claims
    # don't hold up loses influence on which candidate actually gets picked.
    scored = apply_claims_adjustment(
        scored, digest, stage_distribution, settings.claims_conviction_floor
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
        if not dry_run and tradier_client is not None and not _ticker_tradeable(tradier_client, ticker):
            # The LLM occasionally emits a company name or wrong symbol (e.g.
            # "MOBILEYE" instead of "MBLY"); such a ticker resolves to no
            # Tradier quote. Rather than discard a potentially good idea
            # outright (#3), attempt to resolve it to a real tradeable ticker
            # via Tradier's symbol search — bounded to one search per
            # considered candidate.
            resolved: str | None = None
            try:
                resolved = tradier_client.search_symbol(ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning("search_symbol failed for %s: %s", ticker, exc)
                resolved = None
            resolved_ok = (
                bool(resolved)
                and resolved.upper() not in {str(t).upper() for t in settings.megacap_exclude}
                and _ticker_tradeable(tradier_client, resolved)
            )
            if resolved_ok:
                logger.info("Resolved candidate ticker %s -> %s via Tradier symbol search", ticker, resolved)
                candidate["ticker"] = resolved
                ticker = resolved
            else:
                logger.info("Candidate %s is not a resolvable/tradeable symbol, skipping", ticker)
                continue
        top_pick = candidate
        break

    # 9b. Signal-quality gates (#17 data-quality, #20 clustering) — deliberately
    # BEFORE any Tradier calls so a low-confidence/single-outlier pick never
    # wastes an option-chain lookup or gets emitted as a signal.
    no_signal_reason: str | None = None
    if top_pick is not None:
        dq = compute_data_quality(digest)
        if dq < settings.min_data_quality:
            no_signal_reason = (
                f"Datenqualität zu niedrig ({dq}/100, Minimum {settings.min_data_quality}) "
                "— Signal unterdrückt (mögliche Quellen fehlen/rate-limited)."
            )
            logger.info("Data-quality gate fired for %s: %s", top_pick.get("ticker"), no_signal_reason)
            top_pick = None

    if top_pick is not None:
        discovery = top_pick.get("discovery") or {}
        theme_id = discovery.get("theme_id") or top_pick.get("theme_id")
        cluster_count = _cluster_member_count(
            scored, top_pick.get("stage_id"), theme_id, settings.cluster_score_bar
        )
        if cluster_count < settings.cluster_min_members:
            no_signal_reason = (
                "Kein Cluster: nur ein glaubwürdiger Kandidat im Thema/der Stufe — "
                "mögliches Einzelrauschen statt echtem emergentem Cluster."
            )
            logger.info("Cluster gate fired for %s: %s", top_pick.get("ticker"), no_signal_reason)
            top_pick = None

    # 9c. Regime gate (CONCEPT_PROFIT.md Phase D): a genuine risk-off regime
    # (settings.regime_benchmark, e.g. SPY, below its 200-day SMA) hard-blocks
    # Long-Call signals — trend-following calls get run over in a broad
    # risk-off market, and not-trading is itself a valid, often profitable
    # outcome (see CONCEPT_PROFIT.md). Skipped in dry-run / without Tradier
    # (no history to fetch); insufficient/unavailable history fails OPEN
    # (never blocks on a data gap — see _regime_blocks).
    if top_pick is not None and settings.regime_gate and not dry_run and tradier_client is not None:
        risk_on: bool | None = None
        try:
            regime_hist_start = (date.today() - timedelta(days=400)).strftime("%Y-%m-%d")
            regime_history = tradier_client.get_history(settings.regime_benchmark, start=regime_hist_start)
            regime_closes = [
                bar.get("close")
                for bar in regime_history
                if isinstance(bar, dict) and bar.get("close") is not None
            ]
            risk_on = trend.regime_risk_on(regime_closes)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Regime gate history fetch failed for %s, not blocking: %s", settings.regime_benchmark, exc
            )
            risk_on = None

        if risk_on is None:
            logger.info(
                "Regime gate: insufficient/unavailable history for %s, not blocking", settings.regime_benchmark
            )
        elif _regime_blocks(risk_on):
            no_signal_reason = (
                f"Risk-off-Regime: {settings.regime_benchmark} unter 200-Tage-Linie — "
                "keine Long-Call-Signale in diesem Regime."
            )
            logger.info("Regime gate fired for %s: %s", top_pick.get("ticker"), no_signal_reason)
            top_pick = None

    # 9d. Validation gate (CONCEPT_PROFIT.md Phase D): the production system
    # only emits a tradeable Option signal when src/backtest/optimize.py
    # produced a calibration with a VALIDATED (out-of-sample, `calib` loaded
    # once above alongside effective_weights) edge. Otherwise the honest
    # answer is "no validated edge" — the run stays in OBSERVATION MODE:
    # candidates/scores/stage-distribution/emergence are still computed,
    # archived (digest_history, step 12b below), and reported in the email so
    # the forward record keeps growing; only the tradeable Option signal is
    # withheld.
    observation_mode = False
    calibration_status = _calibration_status(calib)
    if settings.validation_gate and not calibration.is_validated(calib):
        observation_mode = True
        no_signal_reason = (
            "Keine validierte Kante: Das Scoring hat im Out-of-Sample-Backtest (noch) keine "
            "positive Kante gezeigt — Beobachtungsmodus, kein Trade-Signal. "
            f"(Kalibrierung: {calibration_status})"
        )
        logger.info("Validation gate fired: %s", no_signal_reason)
        top_pick = None

    # 10. Option selection + divergence rescoring for the chosen top pick
    option = None
    three_month_perf = None
    structure = None
    rvol = None
    edate = None
    earnings_trap = False
    earnings_risk_msg = None
    trend_ok_val: bool | None = None
    closes: list[float] = []
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

            # --- Options structure-selection layer (deterministic EV check) ---
            # Only meaningful when a long call was actually selected. Every step
            # degrades to None/skip on any failure, so the plain long-call path
            # keeps working exactly as before when nothing special applies.
            # (`closes` was already initialized to [] above the `if top_pick is
            # not None:` branch so it's always defined even when this whole
            # `else:` — dry-run / no Tradier key — is skipped.)
            # History window is 400 calendar days (not just the ~130 rvol
            # needs) so the SAME fetch also covers the Phase D trend filter's
            # >=200-cleaned-close requirement (trend.trend_ok) — one Tradier
            # call serves both, no redundant fetch.
            if option is not None:
                try:
                    hist_start = (date.today() - timedelta(days=400)).strftime("%Y-%m-%d")
                    history = tradier_client.get_history(ticker, start=hist_start)
                    closes = [
                        bar.get("close")
                        for bar in history
                        if isinstance(bar, dict) and bar.get("close") is not None
                    ]
                    rvol = structures.realized_vol(closes)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("realized_vol computation failed for %s: %s", ticker, exc)
                    rvol = None

                # Trend filter (#2 concept, CONCEPT_PROFIT.md Phase D): a RISK
                # FLAG only, never a hard block (the regime gate above is the
                # hard one) — surfaced as a risk string + persisted on the
                # signal for forward measurement.
                try:
                    trend_ok_val = trend.trend_ok(closes)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("trend_ok computation failed for %s: %s", ticker, exc)
                    trend_ok_val = None

                short_leg = None
                try:
                    short_leg = tradier_client.select_short_leg(
                        ticker,
                        option["expiration"],
                        target_delta=opts_cfg.get("spread_short_delta", 0.32),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("select_short_leg failed for %s: %s", ticker, exc)
                    short_leg = None

                try:
                    underlying_ref = option.get("underlying_price")
                    if not underlying_ref and closes:
                        underlying_ref = closes[-1]
                    if underlying_ref:
                        structure = structures.choose_structure(
                            underlying_ref, option, short_leg, rvol, opts_cfg
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("choose_structure failed for %s: %s", ticker, exc)
                    structure = None

                # Earnings-trap gate (#16): opt-in, inactive when the earnings
                # endpoint is unavailable. Never hard-blocks; downgrades the
                # note and flags a German risk string when an earnings event
                # falls inside the option's early life (theta/IV-crush trap).
                try:
                    # Tradier's fundamentals calendar is unavailable on many
                    # accounts (returns None); fall back to the free, keyless
                    # Yahoo Finance quoteSummary endpoint (#4) so this gate
                    # actually has a chance to fire instead of staying
                    # permanently dormant.
                    edate = tradier_client.get_next_earnings_date(ticker) or earnings.next_earnings_date(ticker)
                    if edate and option.get("dte"):
                        earnings_dt = datetime.strptime(edate, "%Y-%m-%d").date()
                        days_to_earnings = (earnings_dt - date.today()).days
                        trap_window = opts_cfg.get("earnings_trap_dte_fraction", 0.34) * option["dte"]
                        if 0 <= days_to_earnings <= trap_window:
                            earnings_trap = True
                            earnings_risk_msg = (
                                f"Earnings am {edate} liegen innerhalb der frühen Laufzeit der Option "
                                f"(~{int(trap_window)} Tage) — erhöhtes Theta-/IV-Crush-Risiko rund um den Termin."
                            )
                            if structure and structure.get("structure") == "long_call":
                                structure["earnings_downgrade"] = True
                                structure["reason"] = (
                                    structure.get("reason", "")
                                    + " | Achtung: Earnings-Termin in der frühen Laufzeit — Long Call besonders anfällig."
                                )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("earnings-trap gate failed for %s: %s", ticker, exc)
                    edate = None
                    earnings_trap = False

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
                {
                    k: v
                    for k, v in top_pick.items()
                    if k not in ("scores", "total_score", "source_count", "claims_verified", "claims_factor")
                },
                digest,
                next_stage,
                weights=effective_weights,
                three_month_perf_pct=three_month_perf,
                option=option,
                all_theme_scores=all_theme_scores,
                reliability=effective_reliability,
            )
            # Re-apply the claims dampening (#18) — score_candidate() above
            # recomputed a fresh total_score that doesn't carry the earlier
            # claims_factor, so redo it here on the single rescored candidate.
            apply_claims_adjustment([rescored], digest, stage_distribution, settings.claims_conviction_floor)
            top_pick.update(rescored)
            # Keep list consistent: replace the entry in `scored` too
            for idx, c in enumerate(scored):
                if c.get("ticker") == ticker:
                    scored[idx] = top_pick
                    break
            scored.sort(key=lambda c: c["total_score"], reverse=True)

        top_pick["option"] = option
        # Expose the structure recommendation + earnings-trap flag so the
        # mailer can render them. Absent (None/False) keeps the old rendering.
        top_pick["structure"] = structure
        top_pick["earnings_trap"] = bool(earnings_trap)

        # 11. Risk flags (rendered by the mailer, computed here).
        data_quality_score = compute_data_quality(digest)
        risks = compute_risks(
            top_pick,
            option,
            emergence_result,
            overheated_threshold=float(reward_cfg.get("overheated_score_threshold", 80)),
            data_quality_score=data_quality_score,
        )
        if earnings_risk_msg:
            risks.append(earnings_risk_msg)
        if trend_ok_val is False:
            risks.append(
                "Kurs unter dem Aufwärtstrend (nicht Kurs>50-Tage>200-Tage-Linie) — "
                "Gegentrend-Signal, erhöhtes Risiko."
            )
        top_pick["risks"] = risks

        # Persist the new signal (only for real runs where we'd actually track it)
        if not dry_run:
            entry_mid = option.get("mid") if option else None
            entry_underlying = option.get("underlying_price") if option else None

            # Sector benchmarks (#11): measure this signal against the ETF
            # mapped to its stage (e.g. SOXX for compute/semis) instead of
            # always SPY.
            benchmark_symbol = settings.benchmark_for(top_pick.get("stage_id"))
            benchmark_at_signal = None
            if tradier_client is not None:
                try:
                    bench_quote = tradier_client.get_quote(benchmark_symbol)
                    if bench_quote:
                        benchmark_at_signal = bench_quote.get("last") or bench_quote.get("close")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Benchmark quote lookup failed for %s: %s", benchmark_symbol, exc)
            top_pick["benchmark_symbol"] = benchmark_symbol

            # Deceleration filter, free parts (#10): insider selling (Form 4)
            # and relative-strength roll-over — computed only for this single
            # top_pick, bounded to a handful of extra requests. Everything is
            # wrapped so any failure just skips the flags/risk strings.
            insider_signal = None
            try:
                if tradier_client is not None:
                    ticker_to_cik = edgar_capex.build_ticker_to_cik_map()
                    insider_signal = insider_mod.insider_selling_signal(
                        ticker, lambda t, _map=ticker_to_cik: _map.get(t)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("insider_selling_signal failed for %s: %s", ticker, exc)
                insider_signal = None

            rs_signal = None
            try:
                if tradier_client is not None and closes:
                    bench_hist_start = (date.today() - timedelta(days=130)).strftime("%Y-%m-%d")
                    bench_history = tradier_client.get_history(benchmark_symbol, start=bench_hist_start)
                    benchmark_closes = [
                        bar.get("close")
                        for bar in bench_history
                        if isinstance(bar, dict) and bar.get("close") is not None
                    ]
                    rs_signal = insider_mod.relative_strength_decel(closes, benchmark_closes)
            except Exception as exc:  # noqa: BLE001
                logger.warning("relative_strength_decel failed for %s: %s", ticker, exc)
                rs_signal = None

            if insider_signal and (
                insider_signal.get("net_sell_hint") == "sell"
                or (insider_signal.get("recent_form4") or 0) >= INSIDER_FORM4_HIGH_THRESHOLD
            ):
                risks.append("Auffällige Insider-Verkäufe (Form 4) in den letzten ~90 Tagen.")
            if rs_signal and rs_signal.get("decelerating"):
                risks.append(
                    "Relative Stärke dreht nach unten (kurzfristige RS < mittelfristige) — "
                    "mögliche Erschöpfung trotz guter Nachrichten."
                )
            top_pick["risks"] = risks

            # Invalidation levels (#14): 50-day moving average + qualitative notes.
            invalidation = None
            if tradier_client is not None:
                invalidation = compute_invalidation(tradier_client, ticker, closes)
            top_pick["invalidation"] = invalidation

            # IV-rank forward (#6): rank this option's IV against this ticker's
            # OWN accumulated history (Tradier exposes no IV history, so we grow
            # it ourselves). Rank against PRIOR observations, then record the
            # current one so it counts next time.
            iv_pct = None
            option_iv = option.get("iv") if option else None
            if option_iv:
                iv_pct = iv_rank.iv_percentile(
                    ticker,
                    option_iv,
                    iv_rank.load_iv_history(IV_HISTORY_PATH),
                    settings.iv_history_min_samples,
                )
                iv_rank.append_iv(IV_HISTORY_PATH, ticker, option_iv)
                if iv_pct is not None and iv_pct >= 80:
                    risks.append(
                        f"Options-IV im oberen Bereich der eigenen Historie (Perzentil {iv_pct:.0f}) "
                        "— Option historisch teuer, Zeitwert-/Crush-Risiko erhöht."
                    )
                    top_pick["risks"] = risks
            top_pick["iv_percentile"] = iv_pct

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
                structure=structure,
                invalidation=invalidation,
                realized_vol=rvol,
                earnings_date=edate,
                insider=insider_signal,
                rs=rs_signal,
                iv_percentile=iv_pct,
                trend_ok=trend_ok_val,
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
        current_stage=stage_distribution.get("current_stage"),
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
        stage_distribution,
        next_stage,
        top_pick,
        scored[:5],
        track_record,
        emergent_themes=emergence_result.get("emergent_themes", []),
        reward_status=reward_status,
        no_signal_reason=no_signal_reason,
        edgar_language=digest.get("edgar_language"),
        observation_mode=observation_mode,
        calibration_status=calibration_status,
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
