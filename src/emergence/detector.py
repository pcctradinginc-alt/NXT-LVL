"""Theme emergence detection: frequency, acceleration, diversity, novelty -> score.

For each configured theme, this module:
  1. Aggregates a per-source count for the current run from whatever the
     digest actually offers: edgar_fts.theme_counts[theme_id] (real filing
     hits) plus a lightweight digest-based keyword count over the free-text
     fields the existing collectors expose (github top_new_repos, arxiv
     sample_hot_titles). Where a source has no usable text, it simply
     contributes 0 — this keeps the detector fully decoupled from the
     stage-oriented collectors without requiring any collector changes.
  2. Computes acceleration/novelty against the theme's baseline history
     (BEFORE the current observation is appended).
  3. Combines frequency/acceleration/diversity/novelty into a 0-100
     Emergence Score per documented, clipped normalization (CONCEPT A.4).
  4. Flags a theme as "emergent" when score >= theme_threshold AND
     source_diversity >= min_sources.
  5. Appends the new observation to the baseline (growing it for next run).
  6. Cross-references entities.detect_entities() to produce
     emergent_candidates: companies tied to an emergent theme, seen in
     enough sources, and not on the megacap exclusion list.

Output shape (CONCEPT A.6):
  {"emergent_themes": [...], "emergent_candidates": [...],
   "all_theme_scores": {theme_id: score}}
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from src.emergence import baseline as baseline_mod
from src.emergence import entities as entities_mod

logger = logging.getLogger(__name__)

# Sources considered for a theme's source_diversity count. edgar_fts is a
# real, independent source; the remaining three are digest-text-derived
# proxies for github/arxiv/hn activity around the theme's keywords.
THEME_SOURCES = ["edgar_fts", "github_trends", "arxiv_trends", "hn_buzz"]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _theme_text_corpus(digest: dict[str, Any]) -> dict[str, list[str]]:
    """Collect {source: [text, ...]} from the digest's available free-text fields."""
    corpus: dict[str, list[str]] = {"github_trends": [], "arxiv_trends": [], "hn_buzz": []}

    try:
        github = digest.get("github_trends") or {}
        for repo in github.get("top_new_repos", []) or []:
            parts = [
                str(repo.get("name") or ""),
                str(repo.get("description") or ""),
                " ".join(repo.get("topics") or []),
            ]
            corpus["github_trends"].append(" ".join(parts))
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.detector: failed to read github corpus: %s", exc)

    try:
        arxiv = digest.get("arxiv_trends") or {}
        corpus["arxiv_trends"].extend(str(t) for t in (arxiv.get("sample_hot_titles") or []) if t)
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.detector: failed to read arxiv corpus: %s", exc)

    try:
        hn = digest.get("hn_buzz") or {}
        # hn_buzz currently only exposes aggregated stage_buzz counts, no raw
        # text. If a future version adds sample titles, pick them up here.
        sample_titles = hn.get("sample_titles") or []
        corpus["hn_buzz"].extend(str(t) for t in sample_titles if t)
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.detector: failed to read hn_buzz corpus: %s", exc)

    return corpus


def _count_keyword_hits(texts: list[str], keywords: list[str]) -> int:
    if not texts or not keywords:
        return 0
    lowered = [t.lower() for t in texts]
    total = 0
    for keyword in keywords:
        kw = str(keyword).lower()
        total += sum(1 for text in lowered if kw in text)
    return total


def _per_source_counts_for_theme(
    theme: dict[str, Any], digest: dict[str, Any], text_corpus: dict[str, list[str]]
) -> dict[str, int]:
    theme_id = theme.get("id")
    keywords = theme.get("keywords") or []

    edgar_fts = digest.get("edgar_fts") or {}
    edgar_count = (edgar_fts.get("theme_counts") or {}).get(theme_id, 0) or 0

    counts = {
        "edgar_fts": int(edgar_count),
        "github_trends": _count_keyword_hits(text_corpus.get("github_trends", []), keywords),
        "arxiv_trends": _count_keyword_hits(text_corpus.get("arxiv_trends", []), keywords),
        "hn_buzz": _count_keyword_hits(text_corpus.get("hn_buzz", []), keywords),
    }
    return counts


def _novelty(baseline_stats_obj: dict[str, Any], min_history_for_baseline: int, novelty_window_days: int) -> float:
    """High novelty when history is thin or the theme just recently went nonzero."""
    n = baseline_stats_obj.get("n", 0)
    if n < min_history_for_baseline:
        return 1.0

    last_nonzero_date = baseline_stats_obj.get("last_nonzero_date")
    if not last_nonzero_date:
        # Never observed nonzero before -> maximally novel.
        return 1.0

    try:
        last_date = datetime.strptime(last_nonzero_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 1.0

    days_since = (date.today() - last_date).days
    if days_since <= 0:
        # First time going nonzero right now (baseline had zero-only history).
        return 1.0
    if days_since >= novelty_window_days:
        return 0.0
    return _clip01(1.0 - (days_since / novelty_window_days))


def detect(
    digest: dict[str, Any],
    themes: list[dict[str, Any]],
    emergence_cfg: dict[str, Any],
    baseline_obj: dict[str, Any],
    entity_aliases: dict[str, list[str]] | None = None,
    megacap_exclude: list[str] | None = None,
) -> dict[str, Any]:
    """Detect emergent themes and candidates for the current run.

    Mutates `baseline_obj` in place (appends this run's observation for every
    theme) — callers must persist it via baseline.save() afterwards.
    """
    themes = themes or []
    entity_aliases = entity_aliases or {}
    megacap_exclude = megacap_exclude or []

    baseline_window = int(emergence_cfg.get("baseline_window", 30))
    min_history_for_baseline = int(emergence_cfg.get("min_history_for_baseline", 3))
    novelty_window_days = int(emergence_cfg.get("novelty_window_days", 90))
    min_sources = int(emergence_cfg.get("min_sources", 2))
    theme_threshold = float(emergence_cfg.get("theme_threshold", 60))
    score_weights = emergence_cfg.get("score_weights", {}) or {}
    w_frequency = float(score_weights.get("frequency", 0.30))
    w_acceleration = float(score_weights.get("acceleration", 0.35))
    w_diversity = float(score_weights.get("diversity", 0.20))
    w_novelty = float(score_weights.get("novelty", 0.15))

    text_corpus = _theme_text_corpus(digest)
    today_str = date.today().isoformat()

    # Pass 1: compute raw per-theme counts, frequency, diversity, and
    # baseline-relative acceleration/novelty stats (baseline BEFORE append).
    per_theme_raw: dict[str, dict[str, Any]] = {}
    for theme in themes:
        theme_id = theme.get("id")
        if not theme_id:
            continue

        per_source_counts = _per_source_counts_for_theme(theme, digest, text_corpus)
        frequency = sum(per_source_counts.values())
        source_diversity = sum(1 for v in per_source_counts.values() if v > 0)

        history = baseline_mod.history_for(baseline_obj, theme_id)
        stats = baseline_mod.baseline_stats(history)

        mean = stats["mean"]
        std = stats["std"]
        acceleration_z = (frequency - mean) / max(std, 1.0)
        acceleration_ratio = frequency / max(mean, 1.0)
        novelty = _novelty(stats, min_history_for_baseline, novelty_window_days)

        per_theme_raw[theme_id] = {
            "theme": theme,
            "per_source_counts": per_source_counts,
            "frequency": frequency,
            "source_diversity": source_diversity,
            "acceleration_z": acceleration_z,
            "acceleration_ratio": acceleration_ratio,
            "novelty": novelty,
        }

    max_frequency = max((v["frequency"] for v in per_theme_raw.values()), default=0) or 1

    all_theme_scores: dict[str, float] = {}
    emergent_themes: list[dict[str, Any]] = []

    for theme_id, raw in per_theme_raw.items():
        norm_frequency = _clip01(raw["frequency"] / max_frequency)
        norm_acceleration = _clip01(raw["acceleration_z"] / 3.0)
        norm_diversity = _clip01(raw["source_diversity"] / len(THEME_SOURCES))
        norm_novelty = _clip01(raw["novelty"])

        emergence_score = 100.0 * (
            w_frequency * norm_frequency
            + w_acceleration * norm_acceleration
            + w_diversity * norm_diversity
            + w_novelty * norm_novelty
        )
        emergence_score = max(0.0, min(100.0, emergence_score))
        all_theme_scores[theme_id] = round(emergence_score, 1)

        is_emergent = emergence_score >= theme_threshold and raw["source_diversity"] >= min_sources

        if is_emergent:
            confirming_sources = [s for s, c in raw["per_source_counts"].items() if c > 0]
            emergent_themes.append(
                {
                    "theme_id": theme_id,
                    "name": raw["theme"].get("name", theme_id),
                    "emergence_score": round(emergence_score, 1),
                    "acceleration_z": round(raw["acceleration_z"], 2),
                    "source_diversity": raw["source_diversity"],
                    "novelty": round(raw["novelty"], 2),
                    "drivers": {
                        "frequency": raw["frequency"],
                        "acceleration_ratio": round(raw["acceleration_ratio"], 2),
                        "diversity": raw["source_diversity"],
                        "novelty": round(raw["novelty"], 2),
                    },
                    "confirming_sources": confirming_sources,
                }
            )

    emergent_themes.sort(key=lambda t: t["emergence_score"], reverse=True)

    # Entity detection: raw cross-source candidates, then filter to those
    # whose best-scoring candidate theme is actually emergent.
    emergent_theme_ids = {t["theme_id"] for t in emergent_themes}
    raw_entities = entities_mod.detect_entities(
        digest, themes, entity_aliases=entity_aliases, megacap_exclude=megacap_exclude, min_sources=min_sources
    )

    emergent_candidates: list[dict[str, Any]] = []
    for entity in raw_entities:
        candidate_theme_ids = [t for t in entity.get("candidate_themes", []) if t]
        if not candidate_theme_ids:
            continue
        # Pick the candidate theme with the highest emergence score.
        best_theme_id = max(candidate_theme_ids, key=lambda tid: all_theme_scores.get(tid, 0.0))
        if best_theme_id not in emergent_theme_ids:
            continue

        theme_score_info = next((t for t in emergent_themes if t["theme_id"] == best_theme_id), None)
        emergent_candidates.append(
            {
                "ticker": entity["ticker"],
                "theme_id": best_theme_id,
                "sources": entity["sources"],
                "emergence_score": all_theme_scores.get(best_theme_id, 0.0),
                "drivers": theme_score_info["drivers"] if theme_score_info else {},
                "first_seen": today_str,
                "in_watchlist": None,  # filled in by caller (main.py has watchlist context)
            }
        )

    emergent_candidates.sort(key=lambda c: c["emergence_score"], reverse=True)

    # Grow the baseline with this run's observation for every theme.
    for theme_id, raw in per_theme_raw.items():
        observation = {
            "date": today_str,
            "per_source_counts": raw["per_source_counts"],
            "frequency": raw["frequency"],
            "source_diversity": raw["source_diversity"],
        }
        baseline_mod.append_observation(baseline_obj, theme_id, observation, window=baseline_window)

    logger.info(
        "emergence.detector: %d/%d themes emergent, %d emergent candidates",
        len(emergent_themes),
        len(themes),
        len(emergent_candidates),
    )

    return {
        "emergent_themes": emergent_themes,
        "emergent_candidates": emergent_candidates,
        "all_theme_scores": all_theme_scores,
    }
