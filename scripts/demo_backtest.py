"""Demo script: run both strategies over one symbol and print the results.

Usage:
    python scripts/demo_backtest.py --symbol AAPL
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable when this script runs directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.backtester.engine import run_backtest
from app.config import END_DATE, START_DATE, SYMBOLS
from app.data.database import init_db, load_prices, save_prices
from app.data.fetcher import load_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a demo backtest")
    parser.add_argument("--symbol", default=SYMBOLS[0])
    args = parser.parse_args()

    init_db()

    data = load_prices(args.symbol)
    if data.empty:
        data = load_from_csv(args.symbol)
        save_prices(args.symbol, data, source="csv")

    print(f"Demo backtest on {args.symbol.upper()}, {len(data)} bars\n")

    runs = [
        run_backtest(args.symbol, data, "moving_average", {"fast": 10, "slow": 50}),
        run_backtest(args.symbol, data, "volatility_breakout", {"window": 20, "num_std": 2.0}),
    ]

    summary = []
    for run in runs:
        m = run.metrics
        summary.append(
            {
                "strategy": run.strategy_name,
                "total_return": m["total_return"],
                "sharpe": m["sharpe_ratio"],
                "max_drawdown": m["max_drawdown"],
                "win_rate": m["win_rate"],
                "trades": m["num_trades"],
            }
        )

    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()