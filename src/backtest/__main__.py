"""Entry point for `python -m src.backtest` — delegates to price_backtest.main."""

from __future__ import annotations

import sys

from src.backtest.price_backtest import main

if __name__ == "__main__":
    sys.exit(main())
