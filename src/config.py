"""Loads config.yaml and environment variables into a single Settings object.

Usage:
    from src.config import load_settings
    settings = load_settings()
    settings.stages            # list of stage dicts from config.yaml
    settings.gemini_api_key    # from env
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class Settings:
    """Aggregated configuration: static config.yaml content + env secrets."""

    raw: dict[str, Any] = field(default_factory=dict)

    # Env-provided secrets / runtime knobs
    gemini_api_key: str = ""
    tradier_api_key: str = ""
    tradier_env: str = "prod"
    gmail_app_password: str = ""
    mail_from: str = ""
    mail_to: str = ""
    github_token: str = ""
    offline: bool = False

    @property
    def stages(self) -> list[dict[str, Any]]:
        return self.raw.get("stages", [])

    @property
    def capex_companies(self) -> list[str]:
        return self.raw.get("capex_companies", [])

    @property
    def scoring(self) -> dict[str, Any]:
        return self.raw.get("scoring", {})

    @property
    def scoring_weights(self) -> dict[str, float]:
        return self.scoring.get("weights", {})

    @property
    def signal_threshold(self) -> float:
        return float(self.scoring.get("signal_threshold", 70))

    @property
    def min_sources(self) -> int:
        return int(self.scoring.get("min_sources", 2))

    @property
    def cooldown_days(self) -> int:
        return int(self.scoring.get("cooldown_days", 14))

    @property
    def options_config(self) -> dict[str, Any]:
        return self.raw.get("options", {})

    @property
    def tracking_config(self) -> dict[str, Any]:
        return self.raw.get("tracking", {})

    @property
    def themes(self) -> list[dict[str, Any]]:
        return self.raw.get("themes", [])

    @property
    def entity_aliases(self) -> dict[str, list[str]]:
        return self.raw.get("entity_aliases", {})

    @property
    def megacap_exclude(self) -> list[str]:
        return self.raw.get("megacap_exclude", [])

    @property
    def emergence_config(self) -> dict[str, Any]:
        return self.raw.get("emergence", {})

    @property
    def reward_config(self) -> dict[str, Any]:
        return self.raw.get("reward", {})

    def stage_by_id(self, stage_id: int) -> dict[str, Any] | None:
        for stage in self.stages:
            if stage.get("id") == stage_id:
                return stage
        return None

    def all_keywords(self) -> dict[int, list[str]]:
        """Return {stage_id: [keywords]} for every configured stage."""
        return {stage["id"]: stage.get("keywords", []) for stage in self.stages}

    def watchlist_tickers(self) -> set[str]:
        """Return the set of all tickers across the 7-stage watchlist."""
        tickers: set[str] = set()
        for stage in self.stages:
            for ticker in stage.get("tickers", []):
                tickers.add(str(ticker).upper())
        return tickers


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping")
    return data


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load config.yaml plus environment variables into a Settings instance."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _load_yaml(path)

    return Settings(
        raw=raw,
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        tradier_api_key=os.environ.get("TRADIER_API_KEY", ""),
        tradier_env=os.environ.get("TRADIER_ENV", "prod") or "prod",
        gmail_app_password=os.environ.get("GMAIL_APP_PASSWORD", ""),
        mail_from=os.environ.get("MAIL_FROM", ""),
        mail_to=os.environ.get("MAIL_TO", ""),
        github_token=os.environ.get("GITHUB_TOKEN", ""),
        offline=os.environ.get("NXT_OFFLINE", "") == "1",
    )
