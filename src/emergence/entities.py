"""Entity (company) co-occurrence detection across the digest's free-text corpora.

Builds a `names_dictionary` mapping tokens (tickers themselves, plus optional
name aliases from config.yaml `entity_aliases`) to tickers, then scans the
free-text fields the existing collectors already expose:
  - github_trends.top_new_repos[].name / description / topics
  - arxiv_trends.sample_hot_titles
  - jobs_hn / hn_buzz do not currently expose raw text (only aggregated
    counts), so they are skipped defensively if absent.

A company becomes a *raw* candidate (not yet emergence-filtered — that is the
detector's job) when it is mentioned in >= min_sources distinct corpora and is
not in the megacap exclusion list.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _build_names_dictionary(
    themes: list[dict[str, Any]], entity_aliases: dict[str, list[str]]
) -> dict[str, set[str]]:
    """Return {ticker: {alias_or_ticker_lowercased, ...}} for every known ticker."""
    names: dict[str, set[str]] = {}
    for theme in themes:
        for ticker in theme.get("tickers", []):
            ticker = str(ticker).upper()
            names.setdefault(ticker, set()).add(ticker.lower())
    for ticker, aliases in (entity_aliases or {}).items():
        ticker = str(ticker).upper()
        bucket = names.setdefault(ticker, set())
        for alias in aliases:
            bucket.add(str(alias).lower())
    return names


def _theme_tickers_map(themes: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return {ticker: [theme_id, ...]} — a ticker may belong to multiple themes."""
    mapping: dict[str, list[str]] = {}
    for theme in themes:
        theme_id = theme.get("id")
        for ticker in theme.get("tickers", []):
            ticker = str(ticker).upper()
            mapping.setdefault(ticker, []).append(theme_id)
    return mapping


def _collect_corpora(digest: dict[str, Any]) -> dict[str, list[str]]:
    """Collect {source_name: [text_snippet, ...]} from whatever the digest offers.

    Defensive: any missing/malformed field is simply skipped, never raises.
    """
    corpora: dict[str, list[str]] = {}

    try:
        github = digest.get("github_trends") or {}
        repo_texts: list[str] = []
        for repo in github.get("top_new_repos", []) or []:
            parts = [
                str(repo.get("name") or ""),
                str(repo.get("description") or ""),
                " ".join(repo.get("topics") or []),
            ]
            repo_texts.append(" ".join(parts))
        if repo_texts:
            corpora["github_trends"] = repo_texts
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.entities: failed to build github corpus: %s", exc)

    try:
        arxiv = digest.get("arxiv_trends") or {}
        titles = [str(t) for t in (arxiv.get("sample_hot_titles") or []) if t]
        if titles:
            corpora["arxiv_trends"] = titles
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.entities: failed to build arxiv corpus: %s", exc)

    # jobs_hn / hn_buzz: current collectors only expose aggregated counts, no
    # raw comment/story text. Skip gracefully — if a future collector version
    # adds raw text fields (e.g. "sample_comments" / "sample_titles"), pick
    # them up here too.
    try:
        jobs = digest.get("jobs_hn") or {}
        raw_texts = jobs.get("sample_comments") or jobs.get("sample_texts")
        if raw_texts:
            corpora["jobs_hn"] = [str(t) for t in raw_texts if t]
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.entities: failed to build jobs_hn corpus: %s", exc)

    try:
        hn = digest.get("hn_buzz") or {}
        raw_titles = hn.get("sample_titles") or hn.get("sample_texts")
        if raw_titles:
            corpora["hn_buzz"] = [str(t) for t in raw_titles if t]
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.entities: failed to build hn_buzz corpus: %s", exc)

    try:
        edgar_fts = digest.get("edgar_fts") or {}
        # No raw text available from FTS (only counts); nothing to add here.
        _ = edgar_fts
    except Exception:  # noqa: BLE001
        pass

    return corpora


def _mentions_in_text(text_lower: str, ticker: str, aliases: set[str]) -> bool:
    """True if `ticker` (word-boundary) or any alias (substring) appears in text_lower."""
    ticker_pattern = r"\b" + re.escape(ticker.lower()) + r"\b"
    if re.search(ticker_pattern, text_lower):
        return True
    for alias in aliases:
        if alias == ticker.lower():
            continue  # already checked via word-boundary above
        if alias in text_lower:
            return True
    return False


def detect_entities(
    digest: dict[str, Any],
    themes: list[dict[str, Any]],
    entity_aliases: dict[str, list[str]] | None = None,
    megacap_exclude: list[str] | None = None,
    min_sources: int = 2,
) -> list[dict[str, Any]]:
    """Detect raw company candidates mentioned in >= min_sources digest corpora.

    Returns a list of:
      {"ticker", "sources": [...], "mention_counts": {source: count},
       "candidate_themes": [theme_id, ...]}

    Emergence filtering (is the theme actually emergent?) is NOT done here —
    that is the detector's responsibility. This function only returns the raw
    cross-source mention evidence.
    """
    entity_aliases = entity_aliases or {}
    megacap_exclude = {str(t).upper() for t in (megacap_exclude or [])}

    names_dictionary = _build_names_dictionary(themes, entity_aliases)
    theme_tickers = _theme_tickers_map(themes)
    corpora = _collect_corpora(digest)

    results: list[dict[str, Any]] = []

    for ticker, aliases in names_dictionary.items():
        if ticker in megacap_exclude:
            continue

        mention_counts: dict[str, int] = {}
        for source, texts in corpora.items():
            count = 0
            for text in texts:
                text_lower = text.lower()
                if _mentions_in_text(text_lower, ticker, aliases):
                    count += 1
            if count > 0:
                mention_counts[source] = count

        sources = sorted(mention_counts.keys())
        if len(sources) < min_sources:
            continue

        results.append(
            {
                "ticker": ticker,
                "sources": sources,
                "mention_counts": mention_counts,
                "candidate_themes": list(theme_tickers.get(ticker, [])),
            }
        )

    logger.info("emergence.entities: detected %d raw cross-source candidates", len(results))
    return results
