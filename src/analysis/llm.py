"""Anthropic (Claude) REST client: turns the compact signal digest into stage + candidates.

Uses the plain Anthropic Messages REST API (no SDK — consistent with the rest of
this project, which talks to every service over `requests` to keep the dependency
surface at just `requests` + `pyyaml`). A single call per pipeline run, with strict
JSON-schema validation of the response and one retry on parse failure. In
--dry-run mode, `analyze()` is bypassed entirely by main.py in favor of
`dry_run_stub()`.

Model: claude-haiku-4-5 (Haiku 4.5) — the cheapest current Claude model
($1 / $5 per million input/output tokens). At one call per day with a ~3-4k-token
digest, cost is a few cents per month.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.http_utils import request_json

logger = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 2048

REQUIRED_TOP_LEVEL_FIELDS = ["current_stage", "next_stage", "candidates"]
REQUIRED_CANDIDATE_FIELDS = ["ticker", "stage_id", "thesis", "source_evidence", "conviction"]

SYSTEM_INSTRUCTIONS = """You are an equity research assistant for an automated AI-infrastructure \
stage-rotation scanner. You will receive a compact JSON "signal digest" built from five free \
data sources (SEC EDGAR capex, GitHub developer trends, HN "who is hiring" job postings, arXiv \
research trends, HN story buzz), plus a static 7-stage value-chain model of the AI buildout:

1 Compute / Semiconductors (Training)
2 Datacenter Infrastructure
3 Energy / Cooling / Grid
4 Inference / Networking / Edge
5 Software / Agents / Data
6 Vertical AI Adoption
7 Robotics / Physical AI

Your job:
1. Determine the CURRENT stage that is most active right now based on the digest.
2. Determine the NEXT stage (3-12 month horizon) that should benefit next.
3. Propose 5-10 candidate tickers for the NEXT stage specifically. Explicitly avoid the current \
mega-cap winners (e.g. NVDA, MSFT, GOOGL, AMZN, META) — focus on names that are not yet fully \
priced in, preferring liquid mid-caps with tradable options relevant to the next stage.
4. For each candidate, provide: ticker, stage_id (int, the stage this candidate belongs to), a \
1-2 sentence thesis, source_evidence (list of source names from \
["edgar_capex","github_trends","jobs_hn","arxiv_trends","hn_buzz"] that support this candidate \
based on the digest), and conviction (float 0-1).

Respond with STRICT JSON only, matching exactly this schema, no markdown fences, no extra text:
{
  "current_stage": <int>,
  "next_stage": <int>,
  "reasoning": "<1-2 sentence summary in English>",
  "candidates": [
    {
      "ticker": "<string>",
      "stage_id": <int>,
      "thesis": "<string>",
      "source_evidence": ["<string>", ...],
      "conviction": <float 0-1>
    }
  ]
}
"""


class LLMResponseError(Exception):
    """Raised when the Claude response cannot be parsed/validated."""


def _validate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise LLMResponseError("Response is not a JSON object")

    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in payload:
            raise LLMResponseError(f"Missing required top-level field: {field_name}")

    candidates = payload["candidates"]
    if not isinstance(candidates, list):
        raise LLMResponseError("candidates must be a list")

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise LLMResponseError("Each candidate must be an object")
        for field_name in REQUIRED_CANDIDATE_FIELDS:
            if field_name not in candidate:
                raise LLMResponseError(f"Candidate missing required field: {field_name}")

    return payload


def _strip_fences(text: str) -> str:
    """Defensively strip Markdown code fences the model may add around JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json) ...
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        # ... and the closing fence, if present.
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _call_anthropic(api_key: str, system: str, user_text: str, model: str) -> str:
    """POST the digest to the Anthropic Messages API and return the raw text reply."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,
        "system": system,
        "messages": [{"role": "user", "content": user_text}],
    }
    response = request_json("POST", ANTHROPIC_URL, headers=headers, json_body=body)
    data = response.json()

    if data.get("stop_reason") == "refusal":
        raise LLMResponseError("Claude declined the request (stop_reason=refusal)")

    content = data.get("content", [])
    if not content:
        raise LLMResponseError("Claude response had no content blocks")
    text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    text = "".join(text_parts).strip()
    if not text:
        raise LLMResponseError("Claude response had no text content")
    return text


def analyze(digest: dict[str, Any], api_key: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Call Claude with the digest and return validated JSON, or a safe fallback.

    On any failure (network, parse, validation) after retry, returns a
    fallback dict with an empty candidate list so main.py can produce a
    "no signal" result instead of crashing.
    """
    fallback: dict[str, Any] = {
        "current_stage": None,
        "next_stage": None,
        "reasoning": "",
        "candidates": [],
    }

    if not api_key:
        logger.warning("llm.analyze: no ANTHROPIC_API_KEY configured, returning fallback")
        return fallback

    user_text = "Signal digest:\n" + json.dumps(digest, ensure_ascii=False)

    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            raw_text = _call_anthropic(api_key, SYSTEM_INSTRUCTIONS, user_text, model)
            payload = json.loads(_strip_fences(raw_text))
            validated = _validate(payload)
            logger.info(
                "llm.analyze: success (attempt %d), current_stage=%s next_stage=%s candidates=%d",
                attempt,
                validated.get("current_stage"),
                validated.get("next_stage"),
                len(validated.get("candidates", [])),
            )
            return validated
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("llm.analyze: attempt %d failed: %s", attempt, exc)
            if attempt == 1:
                user_text += (
                    "\n\nYour previous response could not be parsed as valid JSON matching the "
                    "schema. Reply again with STRICT JSON only, no markdown fences, no extra text."
                )

    logger.warning("llm.analyze: all attempts failed (%s), returning fallback", last_error)
    return fallback


def dry_run_stub() -> dict[str, Any]:
    """Deterministic stand-in for analyze() used by --dry-run (no API key needed)."""
    return {
        "current_stage": 2,
        "next_stage": 3,
        "reasoning": (
            "Dry-run stub: datacenter capex remains elevated while power/cooling constraints "
            "increasingly dominate build-out timelines, favoring energy & grid infrastructure next."
        ),
        "candidates": [
            {
                "ticker": "VRT",
                "stage_id": 3,
                "thesis": (
                    "Liquid cooling demand scales directly with GPU density; backlog growth "
                    "signals the next capex wave has not fully arrived yet."
                ),
                "source_evidence": ["edgar_capex", "github_trends"],
                "conviction": 0.75,
            },
            {
                "ticker": "MOD",
                "stage_id": 3,
                "thesis": (
                    "Modular datacenter and power infrastructure builder well positioned as "
                    "hyperscalers race to bring capacity online faster."
                ),
                "source_evidence": ["edgar_capex", "jobs_hn"],
                "conviction": 0.65,
            },
            {
                "ticker": "GEV",
                "stage_id": 3,
                "thesis": (
                    "Grid and power generation equipment demand is rising as datacenter energy "
                    "needs outpace current utility capacity."
                ),
                "source_evidence": ["edgar_capex", "hn_buzz", "arxiv_trends"],
                "conviction": 0.7,
            },
        ],
    }
