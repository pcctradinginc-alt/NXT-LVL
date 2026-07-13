"""Read-only access to the production scoring calibration result.

CONCEPT_PROFIT.md Phase D: `src/backtest/optimize.py` (owned/finished, not
touched here) produces `data/scoring_calibration.json` — a walk-forward,
out-of-sample validated set of scoring weights plus a `validation.passed`
verdict. This module is the thin, fault-tolerant reader the PRODUCTION
pipeline (src/main.py) uses to decide whether it is allowed to emit a
tradeable signal at all: only when a calibration file exists AND its
`validation.passed` is exactly `True` does the system trust the calibrated
weights. Anything else (missing file, malformed JSON, `passed=False`) means
"no validated edge" and the caller must fall back to observation-only mode.

Never runs a calibration itself and never raises: a missing or corrupt
calibration file must never crash the production run, it must just mean
"stay in observation mode".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

CALIBRATION_PATH = DATA_DIR / "scoring_calibration.json"


def load_calibration(path: Path | str = CALIBRATION_PATH) -> dict[str, Any] | None:
    """Load the calibration JSON at `path`. Fault-tolerant: None if missing or malformed."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("calibration: failed to load %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("calibration: %s did not contain a JSON object, ignoring", path)
        return None
    return data


def is_validated(calib: dict[str, Any] | None) -> bool:
    """True iff `calib` is a dict AND calib['validation']['passed'] is exactly True."""
    if not isinstance(calib, dict):
        return False
    validation = calib.get("validation")
    if not isinstance(validation, dict):
        return False
    return validation.get("passed") is True


def validated_weights(calib: dict[str, Any] | None) -> dict[str, float] | None:
    """`calib['weights_final']` only when `is_validated(calib)`, else None.

    Never returns weights from an unvalidated (passed=False/missing) or
    malformed calibration, even if `weights_final` happens to be present —
    those weights are reported for transparency by optimize.py, not for
    production adoption (see optimize.py's own notes).
    """
    if not is_validated(calib):
        return None
    weights = calib.get("weights_final")
    if not isinstance(weights, dict):
        return None
    return weights
