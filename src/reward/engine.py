"""Attribution ledgers + bounded, logged feature/source weight adaptation.

Pure rule-based reward logic — no hidden optimization. Every number in the
resulting adjustment traces back to a logged ledger entry: {n, wins, sum_alpha}
per feature and per source, built from each signal's `horizon_evals` at the
`primary_horizon`, weighted by `data_quality_score / 100`.

"Overheated" signals (emergence/score >= overheated_score_threshold at signal
time, but negative alpha at the primary horizon) count as a penalty for the
features/sources that drove that signal: they contribute to `n` but never to
`wins`, and their (negative) alpha counts in full toward `sum_alpha`.
"""

from __future__ import annotations

import logging
from typing import Any

from src.reward import weights as weights_mod

logger = logging.getLogger(__name__)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def build_ledgers(
    signals: list[dict[str, Any]],
    primary_horizon: int,
    overheated_threshold: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build {"features": {feature: {n, wins, sum_alpha}}, "sources": {...}}."""
    ledgers: dict[str, dict[str, dict[str, float]]] = {"features": {}, "sources": {}}
    horizon_key = str(primary_horizon)

    for signal in signals:
        horizon_evals = signal.get("horizon_evals") or {}
        evaluation = horizon_evals.get(horizon_key)
        if not evaluation:
            continue

        alpha = evaluation.get("alpha")
        if alpha is None:
            continue

        data_quality = signal.get("data_quality_score")
        quality_weight = (data_quality / 100.0) if isinstance(data_quality, (int, float)) else 1.0
        quality_weight = max(0.0, min(1.0, quality_weight))
        if quality_weight == 0:
            continue

        hit = bool(evaluation.get("hit"))

        # Overheated check: score at signal time >= threshold but negative alpha.
        signal_score = signal.get("score")
        is_overheated = (
            isinstance(signal_score, (int, float))
            and signal_score >= overheated_threshold
            and alpha < 0
        )

        feature_attribution = signal.get("feature_attribution") or {}
        for feature, contribution in feature_attribution.items():
            entry = ledgers["features"].setdefault(feature, {"n": 0.0, "wins": 0.0, "sum_alpha": 0.0})
            entry["n"] += quality_weight
            if hit and not is_overheated:
                entry["wins"] += quality_weight
            entry["sum_alpha"] += alpha * quality_weight

        source_attribution = signal.get("source_attribution") or []
        for source in source_attribution:
            entry = ledgers["sources"].setdefault(source, {"n": 0.0, "wins": 0.0, "sum_alpha": 0.0})
            entry["n"] += quality_weight
            if hit and not is_overheated:
                entry["wins"] += quality_weight
            entry["sum_alpha"] += alpha * quality_weight

    return ledgers


def _nudge(win_rate: float, avg_alpha: float, learning_rate: float, step_max: float) -> float:
    base = learning_rate * (win_rate - 0.5)
    alpha_term = 0.5 * learning_rate * _sign(avg_alpha) * min(1.0, abs(avg_alpha) / 10.0)
    nudge = base + alpha_term
    return max(-step_max, min(step_max, nudge))


def _update_ledger_group(
    obj: dict[str, Any],
    ledger: dict[str, dict[str, float]],
    current_values: dict[str, float],
    target_prefix: str,
    reward_cfg: dict[str, Any],
    bounds_min: float,
    bounds_max: float,
) -> dict[str, float]:
    min_samples = int(reward_cfg.get("min_samples", 5))
    learning_rate = float(reward_cfg.get("learning_rate", 0.04))
    step_max = float(reward_cfg.get("step_max", 0.02))

    updated = dict(current_values)

    for name, old_value in current_values.items():
        entry = ledger.get(name)
        n = entry["n"] if entry else 0.0

        if not entry or n < min_samples:
            weights_mod.record_change(
                obj,
                target=f"{target_prefix}:{name}",
                old=old_value,
                new=old_value,
                reason=f"skipped: n<min_samples (n={n:.1f})",
                evidence={"n": round(n, 2)},
            )
            continue

        win_rate = entry["wins"] / n if n else 0.0
        avg_alpha = entry["sum_alpha"] / n if n else 0.0

        nudge = _nudge(win_rate, avg_alpha, learning_rate, step_max)
        new_value = max(bounds_min, min(bounds_max, old_value + nudge))
        new_value = round(new_value, 4)

        updated[name] = new_value
        weights_mod.record_change(
            obj,
            target=f"{target_prefix}:{name}",
            old=old_value,
            new=new_value,
            reason=f"adjusted: win_rate={win_rate:.2f} avg_alpha={avg_alpha:.2f}",
            evidence={"n": round(n, 2), "win_rate": round(win_rate, 3), "avg_alpha": round(avg_alpha, 3)},
        )

    return updated


def update_weights(
    weights_obj: dict[str, Any],
    ledgers: dict[str, dict[str, dict[str, float]]],
    reward_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Apply bounded, logged nudges to feature weights and source reliability.

    Feature weights are renormalized to sum to 1 after all per-feature
    nudges; source reliabilities are only clipped (no renormalization).
    Returns the updated (and already-history-appended) weights_obj.
    """
    weight_bounds = reward_cfg.get("weight_bounds", {}) or {}
    w_min = float(weight_bounds.get("min", 0.05))
    w_max = float(weight_bounds.get("max", 0.45))

    reliability_bounds = reward_cfg.get("reliability_bounds", {}) or {}
    r_min = float(reliability_bounds.get("min", 0.5))
    r_max = float(reliability_bounds.get("max", 1.5))

    current_feature_weights = weights_mod.current_feature_weights(weights_obj)
    current_reliability = weights_mod.current_reliability(weights_obj)

    updated_features = _update_ledger_group(
        weights_obj,
        ledgers.get("features", {}),
        current_feature_weights,
        "feature",
        reward_cfg,
        w_min,
        w_max,
    )

    total = sum(updated_features.values())
    if total > 0:
        renormalized = {k: round(v / total, 4) for k, v in updated_features.items()}
        if renormalized != updated_features:
            weights_mod.record_change(
                weights_obj,
                target="__renormalize__",
                old=round(total, 4),
                new=1.0,
                reason="renormalized feature weights to sum=1 after adjustment",
                evidence={"before": updated_features, "after": renormalized},
            )
        updated_features = renormalized

    updated_sources = _update_ledger_group(
        weights_obj,
        ledgers.get("sources", {}),
        current_reliability,
        "source",
        reward_cfg,
        r_min,
        r_max,
    )

    weights_obj["feature_weights"] = updated_features
    weights_obj["source_reliability"] = updated_sources

    return weights_obj
