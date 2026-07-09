"""Forward Information-Coefficient (IC) calibration against the digest archive.

Purpose: once data/digest_history.jsonl (appended by src.main.run on every
run, see append_digest_history) has accumulated enough matured observations,
measure which scoring components in src/analysis/scoring.py actually
predicted forward returns. This is the honest, data-based answer to "is the
edge real?" — a true retroactive backtest of the collector-driven signal
logic is impossible (the underlying free data sources are point-in-time and
were never archived), so this is deliberately a FORWARD-looking tool that
gets more informative the longer the scanner has been running.

Information Coefficient (IC) here is the Spearman rank correlation between a
component's score at signal time and the ticker's realized forward return
over `horizon_days`. A component with |IC| near 0 ranked candidates no
better than chance; a strongly positive IC means high scores on that
component actually preceded better forward returns (and the reverse for a
strongly negative IC) — either result is actionable evidence for
recalibrating scoring.DEFAULT_WEIGHTS.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, load_settings
from src.options.tradier import TradierClient

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = DATA_DIR / "digest_history.jsonl"
MIN_OBSERVATIONS = 30

COMPONENTS = [
    "breadth",
    "momentum",
    "stage_fit",
    "divergence",
    "option_quality",
    "emergence",
    "total_score",
]


def _rank(values: list[float]) -> list[float]:
    """Rank values with ties resolved via the average-rank convention.

    Ranks are 1-based (standard Spearman convention); tied values receive
    the mean of the ranks they would otherwise occupy.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # +1 for 1-based ranks
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, stdlib-only (rank both, Pearson on ranks).

    Returns None if n < 3 or either series has zero variance (undefined
    correlation), rather than raising or returning a misleading 0.0.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    rx = _rank(list(xs))
    ry = _rank(list(ys))

    n = len(rx)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n

    var_x = sum((v - mean_x) ** 2 for v in rx)
    var_y = sum((v - mean_y) ** 2 for v in ry)
    if var_x == 0 or var_y == 0:
        return None

    cov = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    return cov / denom


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("calibrate: skipping malformed line in %s", path)
    except OSError as exc:
        logger.warning("calibrate: failed reading %s: %s", path, exc)
    return records


def _forward_return(
    tradier: TradierClient | None,
    ticker: str,
    run_date: str,
    horizon_days: int,
    forward_returns: dict[tuple[str, str], float] | None,
) -> float | None:
    if forward_returns is not None:
        return forward_returns.get((ticker, run_date))
    if tradier is None:
        return None
    try:
        start_dt = datetime.strptime(run_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    end_dt = start_dt + timedelta(days=horizon_days)
    fetch_start = (start_dt - timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        bars = tradier.get_history(ticker, start=fetch_start)
    except Exception as exc:  # noqa: BLE001
        logger.warning("calibrate: history fetch failed for %s: %s", ticker, exc)
        return None
    if not bars:
        return None

    def _bar_date(bar: dict[str, Any]) -> date | None:
        try:
            return datetime.strptime(str(bar["date"])[:10], "%Y-%m-%d").date()
        except (KeyError, ValueError, TypeError):
            return None

    priced = [(_bar_date(b), b) for b in bars]
    priced = [(d, b) for d, b in priced if d is not None]
    if not priced:
        return None
    priced.sort(key=lambda pair: pair[0])

    entry_candidates = [(d, b) for d, b in priced if d <= start_dt]
    if not entry_candidates:
        return None
    entry_date, entry_bar = entry_candidates[-1]

    exit_candidates = [(d, b) for d, b in priced if d <= end_dt]
    if not exit_candidates:
        return None
    exit_date, exit_bar = exit_candidates[-1]
    if exit_date <= entry_date:
        return None

    try:
        entry_close = float(entry_bar["close"])
        exit_close = float(exit_bar["close"])
    except (KeyError, TypeError, ValueError):
        return None
    if entry_close <= 0:
        return None
    return (exit_close - entry_close) / entry_close


def run_calibration(
    tradier: TradierClient | None,
    history_path: Path | str,
    *,
    horizon_days: int = 90,
    forward_returns: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    """Compute per-component Information Coefficients from the digest archive.

    Reads every archived run whose date is older than `horizon_days` (so the
    forward horizon has actually elapsed), fetches (or looks up, if
    `forward_returns` is injected for tests) each candidate's realized
    forward return, and computes Spearman IC between each scoring component
    and that forward return across all matured candidate-observations.

    Returns {"status": "insufficient_history", "n", "needed"} when fewer than
    MIN_OBSERVATIONS matured observations are available — an intentionally
    honest response rather than a false-precision correlation on too few
    points. Otherwise returns {"status": "ok", "n", "ic": {component: value},
    "horizon_days"}.
    """
    history_path = Path(history_path)
    records = _load_history(history_path)

    cutoff = date.today() - timedelta(days=horizon_days)

    paired: dict[str, list[tuple[float, float]]] = {c: [] for c in COMPONENTS}
    n_observations = 0

    for record in records:
        run_date_str = record.get("date")
        if not run_date_str:
            continue
        try:
            run_date = datetime.strptime(run_date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if run_date > cutoff:
            continue  # horizon has not elapsed yet for this run

        for candidate in record.get("candidates", []) or []:
            ticker = candidate.get("ticker")
            if not ticker:
                continue
            fwd_ret = _forward_return(tradier, ticker, run_date_str, horizon_days, forward_returns)
            if fwd_ret is None:
                continue

            n_observations += 1
            scores = candidate.get("scores") or {}
            for component in COMPONENTS:
                if component == "total_score":
                    value = candidate.get("total_score")
                else:
                    value = scores.get(component)
                if isinstance(value, (int, float)):
                    paired[component].append((float(value), fwd_ret))

    if n_observations < MIN_OBSERVATIONS:
        return {"status": "insufficient_history", "n": n_observations, "needed": MIN_OBSERVATIONS}

    ic: dict[str, float | None] = {}
    for component, pairs in paired.items():
        if len(pairs) < 3:
            ic[component] = None
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        value = spearman(xs, ys)
        ic[component] = round(value, 4) if value is not None else None

    return {"status": "ok", "n": n_observations, "ic": ic, "horizon_days": horizon_days}


def _print_report(report: dict[str, Any]) -> None:
    if report["status"] == "insufficient_history":
        print(
            "Insufficient forward-matured history for IC calibration: "
            f"{report['n']}/{report['needed']} candidate-observations available.\n"
            "This is expected early on — data/digest_history.jsonl needs to accumulate "
            "matured runs (older than the horizon) before a meaningful Information "
            "Coefficient can be computed. Run the daily scan for longer, then retry."
        )
        return

    print(f"Forward IC calibration — n={report['n']} matured candidate-observations, horizon={report['horizon_days']}d\n")
    print(f"{'component':<16} {'IC':>8}")
    for component, value in report["ic"].items():
        value_str = f"{value:+.4f}" if value is not None else "n/a"
        print(f"{component:<16} {value_str:>8}")
    print(
        "\nA high |IC| means that component ranked winners well; near-0 means it's "
        "noise and a candidate for down-weighting in config.yaml's scoring.weights."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NXT LVL — forward Information-Coefficient calibration from the digest archive"
    )
    parser.add_argument("--horizon", type=int, default=90, dest="horizon_days", help="Forward horizon in days (default 90)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    args = parse_args(argv)

    settings = load_settings()
    tradier = None
    if settings.tradier_api_key:
        tradier = TradierClient(settings.tradier_api_key, settings.tradier_env)
    else:
        logger.info("TRADIER_API_KEY not set; forward returns cannot be fetched (will report insufficient history)")

    report = run_calibration(tradier, DEFAULT_HISTORY_PATH, horizon_days=args.horizon_days)
    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
