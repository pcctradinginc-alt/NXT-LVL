"""Deterministic scoring of LLM candidates against collector data.

The LLM proposes theses; this module turns them into a reproducible 0-100
score per candidate so the same digest always produces the same ranking.

Components (0-100 each), combined via config-driven weights:
  breadth        - how many independent sources support the candidate
  momentum       - normalized growth signal strength for the candidate's stage
  stage_fit      - does the candidate belong to the identified NEXT stage?
  divergence     - is the stock's recent price move still small (not priced in)?
  option_quality - liquidity of the selected option contract (OI, spread)

conviction (LLM, 0-1) is applied as a final multiplier scaled to 0.8-1.0 so
low-conviction ideas are dampened but never zeroed out.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ALL_SOURCES = ["edgar_capex", "github_trends", "jobs_hn", "arxiv_trends", "hn_buzz"]

# Neutral emergence score used whenever a candidate has no associated
# emergent theme (e.g. plain watchlist candidates) or no all_theme_scores
# lookup was supplied at all. Keeps score_candidate/score_candidates
# backward compatible: existing callers (and tests) that never pass
# `all_theme_scores` get the same relative ordering as before the emergence
# feature was introduced.
NEUTRAL_EMERGENCE_SCORE = 50.0

DEFAULT_WEIGHTS = {
    "breadth": 0.20,
    "momentum": 0.20,
    "stage_fit": 0.15,
    "divergence": 0.15,
    "option_quality": 0.10,
    "emergence": 0.20,
}


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _active_sources(digest: dict[str, Any]) -> set[str]:
    """Sources that actually delivered data in this run's digest."""
    active = set()
    for source in ALL_SOURCES:
        block = digest.get(source)
        if not block:
            continue
        # Heuristic: consider a source "active" if it has at least one
        # non-empty collection beyond the "source" name field.
        has_data = any(
            v for k, v in block.items() if k != "source" and v not in (None, {}, [], "")
        )
        if has_data:
            active.add(source)
    return active


def _stage_keyed_value(block: dict[str, Any], key: str, stage_id: Any) -> Any:
    """Look up `block[key][stage_id]`, tolerating int/str stage-key mismatches.

    Digests loaded fresh from collectors key stages by int; digests that have
    been through a JSON round-trip (e.g. reloaded from data/last_digest.json)
    key them by str. Mirrors the pattern already used in score_momentum.
    """
    sub = block.get(key) or {}
    if stage_id in sub:
        return sub[stage_id]
    return sub.get(str(stage_id))


def _stage_active_sources(candidate: dict[str, Any], digest: dict[str, Any]) -> set[str]:
    """Sources showing non-zero signal for the candidate's stage in the digest.

    A source "confirms" a candidate either via explicit LLM source_evidence
    (handled separately by the caller) or implicitly, when the collector's
    own per-stage aggregate shows real activity for the candidate's stage.
    Candidates without a stage_id (rare emergent-only candidates) get no
    stage-active credit here — they fall back to source_evidence only.
    """
    stage_id = candidate.get("stage_id")
    if stage_id is None:
        return set()

    active: set[str] = set()

    github = digest.get("github_trends") or {}
    heat = _stage_keyed_value(github, "stage_heat", stage_id)
    if isinstance(heat, (int, float)) and heat > 0:
        active.add("github_trends")

    jobs = digest.get("jobs_hn") or {}
    job_count = _stage_keyed_value(jobs, "stage_job_counts", stage_id)
    if isinstance(job_count, (int, float)) and job_count > 0:
        active.add("jobs_hn")

    arxiv = digest.get("arxiv_trends") or {}
    paper_count = _stage_keyed_value(arxiv, "stage_paper_counts", stage_id)
    if isinstance(paper_count, (int, float)) and paper_count > 0:
        active.add("arxiv_trends")

    hn = digest.get("hn_buzz") or {}
    buzz = _stage_keyed_value(hn, "stage_buzz", stage_id)
    if isinstance(buzz, dict) and (
        (buzz.get("stories") or 0) > 0 or (buzz.get("points") or 0) > 0
    ):
        active.add("hn_buzz")

    # edgar_capex has no per-stage breakdown; count it whenever the
    # aggregate capex figure is present at all (it's a stage-agnostic
    # macro signal for the whole AI infra buildout).
    edgar = digest.get("edgar_capex") or {}
    if edgar.get("aggregate_capex_yoy_pct") is not None:
        active.add("edgar_capex")

    return active


def _credited_sources(
    candidate: dict[str, Any],
    digest: dict[str, Any],
) -> set[str]:
    """Union of explicit source_evidence and stage-active sources, restricted
    to sources that are both in ALL_SOURCES and actually active in this run's
    digest. Shared by score_breadth and score_candidate's source_count so the
    two never diverge.
    """
    active = _active_sources(digest)
    evidence = set(candidate.get("source_evidence", [])) & active & set(ALL_SOURCES)
    stage_active = _stage_active_sources(candidate, digest) & active & set(ALL_SOURCES)
    return evidence | stage_active


def score_breadth(
    candidate: dict[str, Any],
    digest: dict[str, Any],
    reliability: dict[str, float] | None = None,
) -> float:
    """Breadth score from real multi-source confirmation.

    A source counts as supporting the candidate if EITHER it is in the
    candidate's `source_evidence`, OR that source shows non-zero signal for
    the candidate's stage in the digest (see `_stage_active_sources`).
    Candidates with no `stage_id` fall back to source_evidence only.

    `reliability` maps source name -> multiplier (default 1.0 for every
    source when omitted, preserving the original unweighted behavior).
    """
    counted = _credited_sources(candidate, digest)

    if not reliability:
        return _clip(len(counted) / len(ALL_SOURCES) * 100)

    weighted = sum(reliability.get(src, 1.0) for src in counted)
    max_possible = len(ALL_SOURCES)  # reliability=1.0 ceiling per source
    return _clip(weighted / max_possible * 100)


def score_emergence(
    candidate: dict[str, Any], all_theme_scores: dict[str, float] | None = None
) -> float:
    """Emergence score of the candidate's associated theme (0-100), as a
    non-penalizing bonus.

    Looks up candidate["theme_id"] first, then candidate["discovery"]["theme_id"].
    When that theme id has a score in `all_theme_scores`, returns
    max(NEUTRAL_EMERGENCE_SCORE, theme_score) — being in an accelerating theme
    boosts the candidate, but not being mapped (or the theme scoring low)
    never drags it below neutral. Returns NEUTRAL_EMERGENCE_SCORE when no
    theme association or no all_theme_scores lookup is available at all.
    """
    if not all_theme_scores:
        return NEUTRAL_EMERGENCE_SCORE

    theme_id = candidate.get("theme_id")
    if theme_id is None:
        discovery = candidate.get("discovery")
        if isinstance(discovery, dict):
            theme_id = discovery.get("theme_id")

    if theme_id is None:
        return NEUTRAL_EMERGENCE_SCORE

    score = all_theme_scores.get(theme_id)
    if score is None:
        return NEUTRAL_EMERGENCE_SCORE
    return _clip(max(NEUTRAL_EMERGENCE_SCORE, float(score)))


def score_momentum(candidate: dict[str, Any], digest: dict[str, Any]) -> float:
    """Combine stage-relevant growth metrics into a single 0-100 score.

    Documented normalization:
    - capex YoY growth: clipped to [0, 50]% mapped -> [0, 100]
    - job MoM keyword change: clipped to [-10, +20] mapped -> [0, 100]
    - github stage_heat: relative to the max heat across all stages -> [0, 100]
    - arxiv paper count + hn_buzz points: relative to max across stages,
      averaged together -> [0, 100]
    The final momentum score is the mean of whichever sub-scores have data.
    """
    stage_id = candidate.get("stage_id")
    sub_scores: list[float] = []

    # Capex YoY (most relevant for stages 2-3, but included whenever present)
    edgar = digest.get("edgar_capex") or {}
    capex_yoy = edgar.get("aggregate_capex_yoy_pct")
    if capex_yoy is not None:
        sub_scores.append(_clip((capex_yoy - 0) / 50 * 100))

    # Job postings MoM change for this stage
    jobs = digest.get("jobs_hn") or {}
    mom_change = (jobs.get("stage_job_mom_change") or {}).get(stage_id)
    if mom_change is None:
        mom_change = (jobs.get("stage_job_mom_change") or {}).get(str(stage_id))
    if mom_change is not None:
        sub_scores.append(_clip((mom_change + 10) / 30 * 100))

    # GitHub developer heat, relative to max across stages
    github = digest.get("github_trends") or {}
    stage_heat = github.get("stage_heat") or {}
    if stage_heat:
        max_heat = max(stage_heat.values()) or 1
        heat = stage_heat.get(stage_id, stage_heat.get(str(stage_id), 0))
        sub_scores.append(_clip(heat / max_heat * 100))

    # arXiv paper counts, relative to max across stages
    arxiv = digest.get("arxiv_trends") or {}
    paper_counts = arxiv.get("stage_paper_counts") or {}
    if paper_counts:
        max_count = max(paper_counts.values()) or 1
        count = paper_counts.get(stage_id, paper_counts.get(str(stage_id), 0))
        sub_scores.append(_clip(count / max_count * 100))

    # HN buzz points, relative to max across stages
    hn = digest.get("hn_buzz") or {}
    stage_buzz = hn.get("stage_buzz") or {}
    if stage_buzz:
        points_by_stage = {k: v.get("points", 0) for k, v in stage_buzz.items()}
        max_points = max(points_by_stage.values()) or 1
        points = points_by_stage.get(stage_id, points_by_stage.get(str(stage_id), 0))
        sub_scores.append(_clip(points / max_points * 100))

    if not sub_scores:
        return 50.0  # neutral when no momentum data is available at all
    return _clip(sum(sub_scores) / len(sub_scores))


def score_stage_fit(candidate: dict[str, Any], next_stage: int | None) -> float:
    if next_stage is None:
        return 0.0
    stage_id = candidate.get("stage_id")
    if stage_id == next_stage:
        return 100.0
    try:
        if abs(int(stage_id) - int(next_stage)) == 1:
            return 50.0
    except (TypeError, ValueError):
        pass
    return 0.0


def score_divergence(three_month_perf_pct: float | None) -> float:
    """Score based on how much of the move has already happened.

    None (no Tradier data, e.g. dry-run) -> neutral 50.

    SIGNED mapping (fix 5): an already-negative 3-month move must NOT score
    as favorably as a flat/small-up move just because abs() makes them look
    similar. A stock down 20% is a falling knife, not a coiled spring — it
    gets a LOW divergence score, not a high one. Mild pullbacks (small
    negative moves) are still treated as fine, since a modest dip can be
    healthy digestion rather than a trend reversal.
    """
    if three_month_perf_pct is None:
        return 50.0
    p = three_month_perf_pct
    if p >= 30:
        return 10.0
    if p >= 15:
        return 40.0
    if p >= 5:
        return 70.0
    if p >= 0:
        return 100.0
    if p >= -5:
        return 70.0
    if p >= -15:
        return 40.0
    return 10.0


def score_option_quality(option: dict[str, Any] | None) -> float:
    """Score based on open interest and spread of the selected option.

    None (no option selected yet, or Tradier unavailable) -> neutral 50.
    """
    if not option:
        return 50.0

    open_interest = option.get("open_interest")
    bid = option.get("bid")
    ask = option.get("ask")
    mid = option.get("mid")

    oi_score = 50.0
    if open_interest is not None:
        oi_score = _clip(open_interest / 500 * 100)  # 500+ OI -> full score

    spread_score = 50.0
    if bid is not None and ask is not None and mid:
        spread_pct = (ask - bid) / mid if mid else 1.0
        spread_score = _clip((0.10 - spread_pct) / 0.10 * 100)

    return _clip((oi_score + spread_score) / 2)


def conviction_multiplier(conviction: float | None) -> float:
    """Map LLM conviction (0-1) to a dampening multiplier (0.8-1.0)."""
    if conviction is None:
        conviction = 0.5
    conviction = max(0.0, min(1.0, float(conviction)))
    return 0.8 + conviction * 0.2


def score_candidate(
    candidate: dict[str, Any],
    digest: dict[str, Any],
    next_stage: int | None,
    weights: dict[str, float] | None = None,
    three_month_perf_pct: float | None = None,
    option: dict[str, Any] | None = None,
    all_theme_scores: dict[str, float] | None = None,
    reliability: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a single candidate. Returns candidate dict enriched with scores.

    `all_theme_scores` and `reliability` are optional additions for the
    Emergence & Reward Engine; omitting them reproduces the original
    (pre-emergence) scoring behavior exactly aside from the rebalanced
    DEFAULT_WEIGHTS, since score_emergence() is neutral (50) without theme data.
    """
    weights = weights or DEFAULT_WEIGHTS

    breadth = score_breadth(candidate, digest, reliability=reliability)
    momentum = score_momentum(candidate, digest)
    stage_fit = score_stage_fit(candidate, next_stage)
    divergence = score_divergence(three_month_perf_pct)
    option_quality = score_option_quality(option)
    emergence = score_emergence(candidate, all_theme_scores)

    weighted_sum = (
        breadth * weights.get("breadth", DEFAULT_WEIGHTS["breadth"])
        + momentum * weights.get("momentum", DEFAULT_WEIGHTS["momentum"])
        + stage_fit * weights.get("stage_fit", DEFAULT_WEIGHTS["stage_fit"])
        + divergence * weights.get("divergence", DEFAULT_WEIGHTS["divergence"])
        + option_quality * weights.get("option_quality", DEFAULT_WEIGHTS["option_quality"])
        + emergence * weights.get("emergence", DEFAULT_WEIGHTS["emergence"])
    )

    multiplier = conviction_multiplier(candidate.get("conviction"))
    total_score = _clip(weighted_sum * multiplier)

    enriched = dict(candidate)
    enriched["scores"] = {
        "breadth": round(breadth, 1),
        "momentum": round(momentum, 1),
        "stage_fit": round(stage_fit, 1),
        "divergence": round(divergence, 1),
        "option_quality": round(option_quality, 1),
        "emergence": round(emergence, 1),
        "conviction_multiplier": round(multiplier, 3),
    }
    enriched["total_score"] = round(total_score, 1)
    enriched["source_count"] = len(_credited_sources(candidate, digest))
    return enriched


def score_candidates(
    candidates: list[dict[str, Any]],
    digest: dict[str, Any],
    next_stage: int | None,
    weights: dict[str, float] | None = None,
    perf_lookup: dict[str, float | None] | None = None,
    option_lookup: dict[str, dict[str, Any] | None] | None = None,
    all_theme_scores: dict[str, float] | None = None,
    reliability: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Score all candidates and return them sorted by total_score descending."""
    perf_lookup = perf_lookup or {}
    option_lookup = option_lookup or {}

    scored = [
        score_candidate(
            candidate,
            digest,
            next_stage,
            weights=weights,
            three_month_perf_pct=perf_lookup.get(candidate.get("ticker")),
            option=option_lookup.get(candidate.get("ticker")),
            all_theme_scores=all_theme_scores,
            reliability=reliability,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda c: c["total_score"], reverse=True)
    return scored
