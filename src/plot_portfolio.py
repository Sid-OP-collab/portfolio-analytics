"""
Generate the headline chart: portfolio market value vs cost basis over time.

Saves to reports/portfolio_value.png -- this is the chart that goes in the
README.

Usage:
    python plot_portfolio.py data/my_transactions_clean.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORTS_DIR = Path("reports")


def plot_value_over_time(portfolio: pd.DataFrame, save_path: Path = None) -> Path:
    """Plot total_value and total_cost_basis over time, save as PNG, return
    the path saved to."""
    save_path = save_path or REPORTS_DIR / "portfolio_value.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio.index, portfolio["total_value"], label="Market value", linewidth=1.8)
    ax.plot(
        portfolio.index,
        portfolio["total_cost_basis"],
        label="Cost basis",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
    )
    ax.fill_between(
        portfolio.index,
        portfolio["total_value"],
        portfolio["total_cost_basis"],
        where=(portfolio["total_value"] >= portfolio["total_cost_basis"]),
        color="tab:green",
        alpha=0.15,
    )
    ax.fill_between(
        portfolio.index,
        portfolio["total_value"],
        portfolio["total_cost_basis"],
        where=(portfolio["total_value"] < portfolio["total_cost_basis"]),
        color="tab:red",
        alpha=0.15,
    )

    ax.set_title("Portfolio value vs cost basis over time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from ledger import build_ledger
    from loader import load_transactions
    from prices import get_prices
    from valuation import value_portfolio

    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
    txns = load_transactions(target)
    _, daily_holdings = build_ledger(txns)

    tickers = daily_holdings["ticker"].unique().tolist()
    price_data = get_prices(tickers, start=str(txns["date"].min().date()))
    portfolio = value_portfolio(daily_holdings, price_data)

    path = plot_value_over_time(portfolio)
    print(f"Saved chart to {path}")
