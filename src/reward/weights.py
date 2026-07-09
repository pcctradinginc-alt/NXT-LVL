"""Persistence and history log for adaptive feature weights & source reliability.

data/weights.json schema:
  {
    "feature_weights": {feature: weight, ...},   # e.g. breadth, momentum, ...
    "source_reliability": {source: multiplier, ...},
    "ledger": {
      "features": {feature: {"n", "wins", "sum_reward"}, ...},
      "sources": {source: {"n", "wins", "sum_reward"}, ...},
    },  # cumulative, consume-once attribution ledger (see src/reward/engine.py)
    "rewarded_evals": ["<signal_id>:<horizon>", ...],  # consumed eval keys
    "history": [
      {"date", "target", "old", "new", "delta", "reason", "evidence"}, ...
    ]
  }

Every adjustment (and every documented skip, logged once) is appended to
`history` and logged via the standard logging module — this is a rule-based,
fully explainable system, never a silent optimizer.

`ledger`/`rewarded_evals` make weight adaptation a CONVERGENT function of
cumulative evidence instead of a per-run nudge: each matured signal-horizon
evaluation is folded into the ledger exactly once (see
`engine.accumulate_ledger`), and `engine.recompute_weights` derives weights
as a deterministic function of that ledger's current totals, so repeated
runs on unchanged evidence stabilize instead of drifting to the bounds.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "data" / "weights.json"


def _today_str() -> str:
    return date.today().isoformat()


def load(
    path: Path | str = DEFAULT_WEIGHTS_PATH,
    defaults_feature: dict[str, float] | None = None,
    defaults_reliability: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Load weights.json, creating it from defaults if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        obj = {
            "feature_weights": dict(defaults_feature or {}),
            "source_reliability": dict(defaults_reliability or {}),
            "ledger": {"features": {}, "sources": {}},
            "rewarded_evals": [],
            "history": [],
        }
        save(obj, path)
        return obj

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("weights.json did not contain an object")
        data.setdefault("feature_weights", dict(defaults_feature or {}))
        data.setdefault("source_reliability", dict(defaults_reliability or {}))
        data.setdefault("ledger", {"features": {}, "sources": {}})
        data["ledger"].setdefault("features", {})
        data["ledger"].setdefault("sources", {})
        data.setdefault("rewarded_evals", [])
        data.setdefault("history", [])
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("reward.weights: failed to load weights.json, using defaults: %s", exc)
        return {
            "feature_weights": dict(defaults_feature or {}),
            "source_reliability": dict(defaults_reliability or {}),
            "ledger": {"features": {}, "sources": {}},
            "rewarded_evals": [],
            "history": [],
        }


def save(obj: dict[str, Any], path: Path | str = DEFAULT_WEIGHTS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def current_feature_weights(obj: dict[str, Any]) -> dict[str, float]:
    return dict(obj.get("feature_weights", {}))


def current_reliability(obj: dict[str, Any]) -> dict[str, float]:
    return dict(obj.get("source_reliability", {}))


def record_change(
    obj: dict[str, Any],
    target: str,
    old: float,
    new: float,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a history entry (adjustment or documented skip) and log it."""
    entry = {
        "date": _today_str(),
        "target": target,
        "old": old,
        "new": new,
        "delta": round(new - old, 6) if isinstance(old, (int, float)) and isinstance(new, (int, float)) else None,
        "reason": reason,
        "evidence": evidence or {},
    }
    obj.setdefault("history", []).append(entry)
    logger.info(
        "reward.weights: %s %s -> %s (%s) evidence=%s",
        target,
        old,
        new,
        reason,
        evidence,
    )
    return obj


def get_effective_weights(
    config_defaults: dict[str, float], path: Path | str = DEFAULT_WEIGHTS_PATH
) -> dict[str, float]:
    """Return the currently effective feature weights.

    Reads weights.json if present (and non-empty), else falls back to
    config_defaults. Used by main.py/scoring so learned adjustments become
    effective without touching config.yaml.
    """
    obj = load(path, defaults_feature=config_defaults)
    weights = current_feature_weights(obj)
    return weights if weights else dict(config_defaults)
