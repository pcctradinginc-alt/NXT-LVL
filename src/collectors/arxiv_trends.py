"""arXiv collector: research topic frequency in recent cs.AI/cs.LG/cs.RO papers.

Free Atom API, parsed with the stdlib xml.etree — a leading indicator for
what may reach products in 6-18 months.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any

from src.http_utils import get_text

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
MAX_RESULTS = 200
TOP_TITLES = 5


def _parse_feed(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    papers: list[dict[str, str]] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title_el = entry.find(f"{ATOM_NS}title")
        summary_el = entry.find(f"{ATOM_NS}summary")
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        papers.append({"title": title, "summary": summary})
    return papers


def _count_keywords(papers: list[dict[str, str]], stage_keywords: dict[int, list[str]]) -> dict[int, int]:
    counts: dict[int, int] = {stage_id: 0 for stage_id in stage_keywords}
    corpus = [(p["title"] + " " + p["summary"]).lower() for p in papers]
    for stage_id, keywords in stage_keywords.items():
        total = 0
        for keyword in keywords:
            kw = keyword.lower()
            total += sum(1 for text in corpus if kw in text)
        counts[stage_id] = total
    return counts


def collect(stage_keywords: dict[int, list[str]] | None = None) -> dict[str, Any]:
    """Collect per-stage arXiv paper counts plus a few sample hot titles.

    Returns:
      {
        "source": "arxiv_trends",
        "stage_paper_counts": {stage_id: count, ...},
        "sample_hot_titles": [title, ...],
      }
    """
    result: dict[str, Any] = {
        "source": "arxiv_trends",
        "stage_paper_counts": {},
        "sample_hot_titles": [],
    }

    if os.environ.get("NXT_OFFLINE") == "1":
        logger.info("arxiv_trends: NXT_OFFLINE=1, skipping network calls")
        return result

    stage_keywords = stage_keywords or {}

    try:
        xml_text = get_text(
            ARXIV_API_URL,
            params={
                "search_query": "cat:cs.AI OR cat:cs.LG OR cat:cs.RO",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": MAX_RESULTS,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("arxiv_trends: failed to fetch feed: %s", exc)
        return result

    try:
        papers = _parse_feed(xml_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("arxiv_trends: failed to parse Atom feed: %s", exc)
        return result

    result["stage_paper_counts"] = _count_keywords(papers, stage_keywords)
    result["sample_hot_titles"] = [p["title"] for p in papers[:TOP_TITLES] if p["title"]]

    logger.info("arxiv_trends: analyzed %d papers", len(papers))
    return result
