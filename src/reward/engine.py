"""Cumulative attribution ledger + convergent feature/source weight adaptation.

Pure rule-based reward logic — no hidden optimization. Every number in the
resulting adjustment traces back to a logged ledger entry: {n, wins,
sum_reward} per feature and per source, accumulated from each signal's
`horizon_evals` at the `primary_horizon`, weighted by `data_quality_score /
100`.

Design (fix 3): the ledger is CONSUME-ONCE and CUMULATIVE, and weights are
recomputed as a deterministic function of that cumulative ledger's current
totals ("target convergence") rather than being nudged a little further
every run. This fixes a real bug in the previous design: since
`build_ledgers` rebuilt from ALL signals every run and `update_weights`
nudged every run, the same static matured signals pushed weights a bit
further every single day until they saturated at the configured bounds —
even though no new evidence had arrived. Now:

  1. `accumulate_ledger` folds each signal-horizon evaluation into the
     cumulative ledger exactly once (tracked via `rewarded_evals`), so
     re-running on the same signals is a no-op.
  2. `recompute_weights` computes each weight as `current + clip(target -
     current, -step_max, +step_max)` where `target` is a deterministic
     function of the ledger's current totals. Because it moves toward a
     fixed target and stops there, repeated calls converge and stay
     converged — they never drift further just because time passed.

"Overheated" signals (score >= overheated_score_threshold at signal time,
but negative reward at the primary horizon) count as a penalty for the
features/sources that drove that signal: they contribute to `n` but never
to `wins`, and their (negative) reward counts in full toward `sum_reward`.

Reward magnitude per evaluation is `alpha` when the benchmark was available,
else `abs_return` (fix 2/3: a missing benchmark must not drop a signal from
the ledger — alpha being None only means "we don't know vs. SPY", not "no
data at all"). Because `hit` is now option-based (see
`src/reward/evaluator.py`), `wins` here tracks profitable OPTIONS, so the
reward engine converges feature/source weights toward what actually made
money, not just toward what made the stock price go up.
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


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def accumulate_ledger(
    weights_obj: dict[str, Any],
    signals: list[dict[str, Any]],
    primary_horizon: int,
    overheated_threshold: float,
) -> int:
    """Fold newly-matured signal evaluations into the cumulative ledger, once.

    For each signal, looks at `horizon_evals[str(primary_horizon)]`. Builds a
    key `f"{signal['id']}:{primary_horizon}"`; if already present in
    `weights_obj["rewarded_evals"]`, skips it (already consumed). Otherwise
    adds this eval's contribution once into `weights_obj["ledger"]` (per
    feature from `feature_attribution`, per source from `source_attribution`,
    weighted by `data_quality_score/100` clipped to [0,1]) and appends the
    key to `rewarded_evals`.

    Mutates `weights_obj` in place. Returns the number of newly consumed
    evals (0 if nothing new was due).
    """
    ledger = weights_obj.setdefault("ledger", {"features": {}, "sources": {}})
    ledger.setdefault("features", {})
    ledger.setdefault("sources", {})
    rewarded_evals: list[str] = weights_obj.setdefault("rewarded_evals", [])
    rewarded_set = set(rewarded_evals)

    horizon_key = str(primary_horizon)
    newly_consumed = 0

    for signal in signals:
        signal_id = signal.get("id")
        if not signal_id:
            continue
        eval_key = f"{signal_id}:{primary_horizon}"
        if eval_key in rewarded_set:
            continue  # already consumed on a previous run

        horizon_evals = signal.get("horizon_evals") or {}
        evaluation = horizon_evals.get(horizon_key)
        if not evaluation:
            continue  # not matured/evaluated yet -> not consumable this run

        alpha = evaluation.get("alpha")
        abs_return = evaluation.get("abs_return")
        reward = alpha if alpha is not None else abs_return
        if reward is None:
            continue  # no usable reward magnitude at all -> skip, retry later

        data_quality = signal.get("data_quality_score")
        quality_weight = (data_quality / 100.0) if isinstance(data_quality, (int, float)) else 1.0
        quality_weight = _clip(quality_weight, 0.0, 1.0)
        if quality_weight == 0:
            # Still consumed (mark as rewarded) — zero-quality data should not
            # be retried indefinitely, it just contributes nothing.
            rewarded_evals.append(eval_key)
            rewarded_set.add(eval_key)
            newly_consumed += 1
            continue

        hit = bool(evaluation.get("hit"))

        signal_score = signal.get("score")
        is_overheated = (
            isinstance(signal_score, (int, float))
            and signal_score >= overheated_threshold
            and reward < 0
        )

        feature_attribution = signal.get("feature_attribution") or {}
        for feature, _contribution in feature_attribution.items():
            entry = ledger["features"].setdefault(feature, {"n": 0.0, "wins": 0.0, "sum_reward": 0.0})
            entry["n"] += quality_weight
            if hit and not is_overheated:
                entry["wins"] += quality_weight
            entry["sum_reward"] += reward * quality_weight

        source_attribution = signal.get("source_attribution") or []
        for source in source_attribution:
            entry = ledger["sources"].setdefault(source, {"n": 0.0, "wins": 0.0, "sum_reward": 0.0})
            entry["n"] += quality_weight
            if hit and not is_overheated:
                entry["wins"] += quality_weight
            entry["sum_reward"] += reward * quality_weight

        rewarded_evals.append(eval_key)
        rewarded_set.add(eval_key)
        newly_consumed += 1

    if newly_consumed:
        logger.info("reward.engine: accumulated %d newly-matured signal evaluation(s) into the ledger", newly_consumed)

    return newly_consumed


def _target_value(
    entry: dict[str, float] | None,
    base_value: float,
    min_samples: int,
    learning_rate: float,
    bounds_min: float,
    bounds_max: float,
) -> tuple[float, float, float, float]:
    """Return (target, n, win_rate, avg_reward) for one ledger entry."""
    n = entry["n"] if entry else 0.0
    if not entry or n < min_samples:
        return base_value, n, 0.0, 0.0

    win_rate = entry["wins"] / n if n else 0.0
    avg_reward = entry["sum_reward"] / n if n else 0.0

    target = base_value * (
        1
        + learning_rate * ((win_rate - 0.5) * 2)
        + 0.5 * learning_rate * _sign(avg_reward) * min(1.0, abs(avg_reward) / 10.0)
    )
    target = _clip(target, bounds_min, bounds_max)
    return target, n, win_rate, avg_reward


def _recompute_group(
    weights_obj: dict[str, Any],
    ledger_group: dict[str, dict[str, float]],
    current_values: dict[str, float],
    base_values: dict[str, float],
    target_prefix: str,
    reward_cfg: dict[str, Any],
    bounds_min: float,
    bounds_max: float,
    renormalize: bool,
) -> dict[str, float]:
    min_samples = int(reward_cfg.get("min_samples", 5))
    learning_rate = float(reward_cfg.get("learning_rate", 0.04))
    step_max = float(reward_cfg.get("step_max", 0.02))

    logged_skip_targets = {
        h["target"] for h in weights_obj.get("history", []) if h.get("reason", "").startswith("skipped: n<min_samples")
    }

    updated = dict(current_values)
    targets: dict[str, float] = {}

    for name, old_value in current_values.items():
        base_value = base_values.get(name, old_value)
        entry = ledger_group.get(name)
        target, n, win_rate, avg_reward = _target_value(
            entry, base_value, min_samples, learning_rate, bounds_min, bounds_max
        )
        targets[name] = target

        target_id = f"{target_prefix}:{name}"

        if not entry or n < min_samples:
            # Avoid daily log spam: only log the "left at base" skip once per
            # target (i.e. the first time we observe n<min_samples for it).
            if target_id not in logged_skip_targets:
                weights_mod.record_change(
                    weights_obj,
                    target=target_id,
                    old=old_value,
                    new=old_value,
                    reason=f"skipped: n<min_samples (n={n:.1f})",
                    evidence={"n": round(n, 2)},
                )
            continue

        step = _clip(target - old_value, -step_max, step_max)
        new_value = round(old_value + step, 4)
        updated[name] = new_value

        if new_value != old_value:
            weights_mod.record_change(
                weights_obj,
                target=target_id,
                old=old_value,
                new=new_value,
                reason=(
                    f"converge: win_rate={win_rate:.2f} avg_reward={avg_reward:.2f} target={target:.4f}"
                ),
                evidence={"n": round(n, 2), "win_rate": round(win_rate, 3), "avg_reward": round(avg_reward, 3), "target": round(target, 4)},
            )
        # else: old == new -> pure no-op, skip logging to avoid spam.

    if renormalize:
        total = sum(updated.values())
        if total > 0:
            renormalized = {k: round(v / total, 4) for k, v in updated.items()}
            if renormalized != updated:
                weights_mod.record_change(
                    weights_obj,
                    target="__renormalize__",
                    old=round(total, 4),
                    new=1.0,
                    reason="renormalized feature weights to sum=1 after convergence step",
                    evidence={"before": updated, "after": renormalized},
                )
            updated = renormalized

    return updated


def recompute_weights(
    weights_obj: dict[str, Any],
    reward_cfg: dict[str, Any],
    base_feature_weights: dict[str, float],
    base_reliability: dict[str, float],
) -> dict[str, Any]:
    """Recompute feature weights and source reliability from the cumulative ledger.

    Deterministic and CONVERGENT: each value moves toward a fixed target
    derived from the ledger's current totals and stops there (bounded by
    `step_max` per call). Calling this repeatedly on an unchanged ledger is
    idempotent after the target is reached — it will NOT keep drifting to
    the configured bounds. Feature weights are renormalized to sum to 1;
    source reliabilities are only clipped (no renormalization), matching the
    previous engine's behavior.

    Returns the updated (and already-history-appended) weights_obj.
    """
    weight_bounds = reward_cfg.get("weight_bounds", {}) or {}
    w_min = float(weight_bounds.get("min", 0.05))
    w_max = float(weight_bounds.get("max", 0.45))

    reliability_bounds = reward_cfg.get("reliability_bounds", {}) or {}
    r_min = float(reliability_bounds.get("min", 0.5))
    r_max = float(reliability_bounds.get("max", 1.5))

    ledger = weights_obj.get("ledger", {}) or {}
    current_feature_weights = weights_mod.current_feature_weights(weights_obj)
    current_reliability = weights_mod.current_reliability(weights_obj)

    updated_features = _recompute_group(
        weights_obj,
        ledger.get("features", {}),
        current_feature_weights,
        base_feature_weights,
        "feature",
        reward_cfg,
        w_min,
        w_max,
        renormalize=True,
    )

    updated_sources = _recompute_group(
        weights_obj,
        ledger.get("sources", {}),
        current_reliability,
        base_reliability,
        "source",
        reward_cfg,
        r_min,
        r_max,
        renormalize=False,
    )

    weights_obj["feature_weights"] = updated_features
    weights_obj["source_reliability"] = updated_sources

    return weights_obj
