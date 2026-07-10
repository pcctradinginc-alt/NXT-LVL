"""Deterministic probabilistic phase model (#5) + machine-checkable-claims
verification (#18).

The LLM is a hypothesis engine, not the source of truth for "which stage is
the AI buildout in": this module turns the SAME stage-level digest metrics
`scoring.score_momentum` already reuses (github stage_heat, jobs
stage_job_counts/stage_job_mom_change, arxiv stage_paper_counts, hn_buzz
stage_buzz points, edgar_capex aggregate YoY) into a reproducible probability
distribution over the 7-stage value chain, via a numerically-stable softmax
over a per-stage "activity" score. `next_stage` picked by this module (not
the LLM's own guess) is what drives `scoring.score_stage_fit` downstream.

`verify_claims` closes the loop the other way: the LLM proposes a `claims`
list per candidate (machine-checkable assertions like "github_trends is
high for this stage"), and this module checks each claim against the same
digest data, deterministically, so the LLM's conviction can be dampened when
its stated reasons don't actually hold up.

Both functions are fully fault-tolerant: partial/empty digests never raise,
they just degrade to neutral/uniform output.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# The 5 free-data sources this module draws stage signal from. Restated
# locally (rather than importing scoring.ALL_SOURCES) to keep phases.py
# self-contained; the set is identical by design.
PHASE_SOURCES = ["github_trends", "jobs_hn", "arxiv_trends", "hn_buzz", "edgar_capex"]

# Aggregate capex YoY growth has no per-stage breakdown, but Datacenter
# Infrastructure (2) and Energy/Cooling/Grid (3) are the stages that most
# directly consume that capex build-out — mirrors the note in
# scoring.score_momentum ("most relevant for stages 2-3").
CAPEX_RELEVANT_STAGES = (2, 3)

DEFAULT_TEMPERATURE = 1.0

VALID_CLAIM_SOURCES = {"edgar_capex", "github_trends", "jobs_hn", "arxiv_trends", "hn_buzz"}
VALID_CLAIM_DIRECTIONS = {"up", "high"}


def _stage_keyed_value(block: dict[str, Any], key: str, stage_id: Any) -> Any:
    """Look up `block[key][stage_id]`, tolerating int/str stage-key mismatches.

    Mirrors scoring._stage_keyed_value: digests loaded fresh from collectors
    key stages by int; digests round-tripped through JSON key them by str.
    """
    sub = block.get(key) or {}
    if stage_id in sub:
        return sub[stage_id]
    return sub.get(str(stage_id))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _median(values: Any) -> float | None:
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0


def _softmax(values: dict[Any, float], temperature: float = DEFAULT_TEMPERATURE) -> dict[Any, float]:
    """Numerically-stable softmax over a {key: activity} mapping (subtract max before exp)."""
    if not values:
        return {}
    temperature = temperature if temperature and temperature > 0 else DEFAULT_TEMPERATURE
    scaled = {k: v / temperature for k, v in values.items()}
    max_v = max(scaled.values())
    exps = {k: math.exp(v - max_v) for k, v in scaled.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


def compute_stage_distribution(digest: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic probability distribution over the 7-stage AI value chain.

    Returns a dict with `probabilities` (sums to ~1), `activity`, `momentum`
    (all keyed by stage id), plus `current_stage` (argmax probability),
    `next_stage` (the accelerating stage at/after current_stage), and
    `confidence` (0-1: concentration of the distribution scaled by how much
    of the digest actually had data).

    Fault-tolerant: an empty/partial digest degrades to a near-uniform
    distribution with low confidence rather than raising. An empty `stages`
    list returns an all-empty/None result.
    """
    stage_ids = [s.get("id") for s in (stages or []) if s.get("id") is not None]

    if not stage_ids:
        return {
            "probabilities": {},
            "activity": {},
            "momentum": {},
            "current_stage": None,
            "next_stage": None,
            "confidence": 0.0,
        }

    digest = digest or {}
    github = digest.get("github_trends") or {}
    jobs = digest.get("jobs_hn") or {}
    arxiv = digest.get("arxiv_trends") or {}
    hn = digest.get("hn_buzz") or {}
    edgar = digest.get("edgar_capex") or {}

    stage_heat = github.get("stage_heat") or {}
    job_counts = jobs.get("stage_job_counts") or {}
    paper_counts = arxiv.get("stage_paper_counts") or {}
    stage_buzz = hn.get("stage_buzz") or {}
    capex_yoy = edgar.get("aggregate_capex_yoy_pct")

    max_heat = max(stage_heat.values()) if stage_heat else 0
    max_jobs = max(job_counts.values()) if job_counts else 0
    max_papers = max(paper_counts.values()) if paper_counts else 0
    buzz_points = {
        k: (v.get("points", 0) if isinstance(v, dict) else 0) for k, v in stage_buzz.items()
    }
    max_buzz = max(buzz_points.values()) if buzz_points else 0

    # Data-completeness factor for `confidence`: how many of the 5 sources
    # actually returned any usable data this run (same non-empty heuristic
    # scoring._active_sources uses).
    active_sources = 0
    for block in (github, jobs, arxiv, hn):
        if block and any(v for k, v in block.items() if k != "source" and v not in (None, {}, [], "", 0)):
            active_sources += 1
    if capex_yoy is not None:
        active_sources += 1
    completeness = active_sources / len(PHASE_SOURCES)

    activity: dict[Any, float] = {}
    momentum: dict[Any, float] = {}

    for stage_id in stage_ids:
        sub_scores: list[float] = []

        if stage_heat:
            heat = _stage_keyed_value(github, "stage_heat", stage_id) or 0
            sub_scores.append(_clip01(heat / max_heat) if max_heat else 0.0)

        if job_counts:
            count = _stage_keyed_value(jobs, "stage_job_counts", stage_id) or 0
            sub_scores.append(_clip01(count / max_jobs) if max_jobs else 0.0)

        if paper_counts:
            count = _stage_keyed_value(arxiv, "stage_paper_counts", stage_id) or 0
            sub_scores.append(_clip01(count / max_papers) if max_papers else 0.0)

        if buzz_points:
            points = buzz_points.get(stage_id, buzz_points.get(str(stage_id), 0))
            sub_scores.append(_clip01(points / max_buzz) if max_buzz else 0.0)

        try:
            stage_id_int = int(stage_id)
        except (TypeError, ValueError):
            stage_id_int = None
        if capex_yoy is not None and stage_id_int in CAPEX_RELEVANT_STAGES:
            sub_scores.append(_clip01(capex_yoy / 50.0))

        activity[stage_id] = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0

        mom_change = _stage_keyed_value(jobs, "stage_job_mom_change", stage_id)
        momentum[stage_id] = float(mom_change) if isinstance(mom_change, (int, float)) else 0.0

    probabilities = _softmax(activity, temperature=DEFAULT_TEMPERATURE)

    current_stage = max(stage_ids, key=lambda sid: probabilities.get(sid, 0.0))

    def _at_or_after_current(sid: Any) -> bool:
        try:
            return int(sid) >= int(current_stage)
        except (TypeError, ValueError):
            return False

    forward_candidates = [sid for sid in stage_ids if _at_or_after_current(sid)]
    best_stage = None
    best_score = None
    for sid in forward_candidates:
        mom = momentum.get(sid, 0.0)
        if mom <= 0:
            continue
        score = probabilities.get(sid, 0.0) * mom
        if best_score is None or score > best_score:
            best_score = score
            best_stage = sid

    if best_stage is not None:
        next_stage = best_stage
    else:
        try:
            fallback = int(current_stage) + 1
        except (TypeError, ValueError):
            fallback = None
        next_stage = fallback if fallback in stage_ids else current_stage

    n = len(stage_ids)
    if n > 1:
        entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
        max_entropy = math.log(n)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0
    confidence = _clip01((1 - normalized_entropy) * completeness)

    return {
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "activity": {k: round(v, 4) for k, v in activity.items()},
        "momentum": {k: round(v, 4) for k, v in momentum.items()},
        "current_stage": current_stage,
        "next_stage": next_stage,
        "confidence": round(confidence, 3),
    }


def verify_claims(
    candidate: dict[str, Any],
    digest: dict[str, Any],
    stage_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a candidate's machine-checkable `claims` against the digest.

    `stage_distribution` is accepted for interface symmetry / future use
    (e.g. claim rules that reference the code's own current/next stage read)
    but the rules below are all self-contained digest lookups, so it is not
    currently consulted.

    Returns `{"verified": int, "total": int, "fraction": float}`. `fraction`
    is 1.0 when `total == 0` — a candidate with no claims (or an LLM/older
    caller that omitted the field) is neutral, never punished. Never raises:
    any per-claim lookup error is logged and simply not counted as verified.
    """
    claims = candidate.get("claims")
    if not isinstance(claims, list) or not claims:
        return {"verified": 0, "total": 0, "fraction": 1.0}

    stage_id = candidate.get("stage_id")
    digest = digest or {}

    github = digest.get("github_trends") or {}
    jobs = digest.get("jobs_hn") or {}
    arxiv = digest.get("arxiv_trends") or {}
    hn = digest.get("hn_buzz") or {}
    edgar = digest.get("edgar_capex") or {}

    heat_median = _median((github.get("stage_heat") or {}).values())
    jobs_median = _median((jobs.get("stage_job_counts") or {}).values())
    papers_median = _median((arxiv.get("stage_paper_counts") or {}).values())

    verified = 0
    total = 0
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        total += 1
        source = claim.get("source")
        direction = claim.get("direction")
        try:
            if source == "github_trends" and direction == "high":
                heat = _stage_keyed_value(github, "stage_heat", stage_id) or 0
                if heat > 0 and heat_median is not None and heat >= heat_median:
                    verified += 1

            elif source == "jobs_hn" and direction == "up":
                mom = _stage_keyed_value(jobs, "stage_job_mom_change", stage_id)
                if isinstance(mom, (int, float)) and mom > 0:
                    verified += 1

            elif source == "jobs_hn" and direction == "high":
                count = _stage_keyed_value(jobs, "stage_job_counts", stage_id) or 0
                if jobs_median is not None and count >= jobs_median:
                    verified += 1

            elif source == "arxiv_trends" and direction == "high":
                count = _stage_keyed_value(arxiv, "stage_paper_counts", stage_id) or 0
                if papers_median is not None and count >= papers_median:
                    verified += 1

            elif source == "hn_buzz" and direction == "high":
                buzz = _stage_keyed_value(hn, "stage_buzz", stage_id) or {}
                points = buzz.get("points", 0) if isinstance(buzz, dict) else 0
                if isinstance(points, (int, float)) and points > 0:
                    verified += 1

            elif source == "edgar_capex" and direction in ("up", "high"):
                capex_yoy = edgar.get("aggregate_capex_yoy_pct")
                if capex_yoy is not None and capex_yoy > 0:
                    verified += 1

            # Any other source/direction combo (including unknown sources,
            # or combos with no defined rule, e.g. github_trends/up) counts
            # toward `total` but is never verified — lenient by design.
        except Exception as exc:  # noqa: BLE001
            logger.warning("verify_claims: error checking claim %s: %s", claim, exc)

    fraction = verified / total if total else 1.0
    return {"verified": verified, "total": total, "fraction": round(fraction, 3)}
