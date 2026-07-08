"""Rolling ThemeObservation history persisted to data/baseline.json.

Schema:
  {
    "themes": {
      theme_id: [
        {"date": "YYYY-MM-DD", "per_source_counts": {...}, "frequency": int,
         "source_diversity": int},
        ...
      ]
    }
  }

The list per theme is capped at `window` entries (oldest first, newest last).
Baseline statistics are always computed from the history BEFORE the current
run's observation is appended — callers (the detector) must call
`baseline_stats(history_for(obj, theme_id))` first, then
`append_observation(...)` afterwards, so the baseline never includes the very
observation it is being used to judge.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "baseline.json"


def load(path: Path | str = DEFAULT_BASELINE_PATH) -> dict[str, Any]:
    """Load the baseline object, or an empty skeleton if the file is missing/corrupt."""
    path = Path(path)
    if not path.exists():
        return {"themes": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and isinstance(data.get("themes"), dict):
            return data
        logger.warning("emergence.baseline: baseline.json malformed, starting fresh")
        return {"themes": {}}
    except Exception as exc:  # noqa: BLE001
        logger.warning("emergence.baseline: failed to load baseline.json: %s", exc)
        return {"themes": {}}


def save(obj: dict[str, Any], path: Path | str = DEFAULT_BASELINE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def history_for(obj: dict[str, Any], theme_id: str) -> list[dict[str, Any]]:
    """Return the (possibly empty) observation history list for a theme."""
    return list(obj.get("themes", {}).get(theme_id, []))


def append_observation(
    obj: dict[str, Any],
    theme_id: str,
    observation: dict[str, Any],
    window: int = 30,
) -> dict[str, Any]:
    """Append a new observation for a theme, capping the history at `window`.

    Mutates and returns `obj`.
    """
    themes = obj.setdefault("themes", {})
    history = themes.setdefault(theme_id, [])
    history.append(observation)
    if len(history) > window:
        del history[: len(history) - window]
    return obj


def baseline_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute baseline mean/std of frequency plus first/last-nonzero dates.

    `history` must be the observation list WITHOUT the current run's
    observation (see module docstring).
    """
    n = len(history)
    frequencies = [float(h.get("frequency", 0) or 0) for h in history]

    mean = statistics.fmean(frequencies) if frequencies else 0.0
    std = statistics.pstdev(frequencies) if len(frequencies) >= 2 else 0.0

    first_seen_date = history[0]["date"] if history else None

    last_nonzero_date = None
    for h in reversed(history):
        if (h.get("frequency", 0) or 0) > 0:
            last_nonzero_date = h.get("date")
            break

    return {
        "mean": mean,
        "std": std,
        "n": n,
        "first_seen_date": first_seen_date,
        "last_nonzero_date": last_nonzero_date,
    }
