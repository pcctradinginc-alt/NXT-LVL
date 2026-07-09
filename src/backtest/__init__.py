"""Validation & backtesting tools for NXT LVL.

Three honest pieces (see module docstrings for detail):
  price_backtest.py - mechanical price backtest of the option/divergence
                       building blocks on real historical equity prices.
  calibrate.py       - forward Information-Coefficient calibration against
                       the accumulating data/digest_history.jsonl archive.

A true retroactive backtest of the full collector-driven signal logic is not
possible: the free data sources (GitHub/EDGAR/HN/arXiv) are point-in-time and
were never archived historically. These tools instead validate what CAN be
validated today (option/divergence mechanics on real prices) and build the
substrate (digest_history.jsonl) for a real forward validation over time.
"""

from __future__ import annotations
