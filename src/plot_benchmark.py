"""
Chart the portfolio against benchmark indices over time.

Produces a two-panel figure:

  Top    -- absolute value. The portfolio alongside what the same cash,
            invested on the same dates, would be worth in SPY and QQQ.
            The dashed line is cumulative contributed capital, so the gap
            between any curve and that line is profit.

  Bottom -- percentage return on contributed capital, which strips out the
            growing size of the portfolio and shows relative performance
            directly. Early values are volatile and not very meaningful:
            a small gain on a small base is a large percentage.

Both panels use the same cash-flow-matched simulation as benchmark.py --
see the module docstring there for why that comparison is the honest one.

Usage:
    python plot_benchmark.py data/my_transactions_clean.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORTS_DIR = Path("reports")


def plot_vs_benchmarks(
    portfolio_value: pd.Series,
    benchmark_values: pd.DataFrame,
    contributed: pd.Series,
    labels: dict[str, str] | None = None,
    save_path: Path | None = None,
    currency: str = "$",
) -> Path:
    """Plot portfolio vs benchmarks, in both absolute and percentage terms.

    portfolio_value: daily total market value.
    benchmark_values: one column per benchmark, simulated on matched flows.
    contributed: cumulative contributed capital, same index.
    """
    labels = labels or {}
    save_path = save_path or REPORTS_DIR / "portfolio_vs_benchmarks.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_abs, ax_pct) = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    # --- Absolute value ---------------------------------------------------
    ax_abs.plot(
        portfolio_value.index, portfolio_value, label="Portfolio",
        linewidth=2.2, color="tab:blue", zorder=3,
    )
    for ticker in benchmark_values.columns:
        ax_abs.plot(
            benchmark_values.index, benchmark_values[ticker],
            label=f"{labels.get(ticker, ticker)} ({ticker})",
            linewidth=1.5, alpha=0.85,
        )
    ax_abs.plot(
        contributed.index, contributed, label="Contributed capital",
        linewidth=1.2, linestyle="--", color="grey", alpha=0.8,
    )
    ax_abs.fill_between(
        portfolio_value.index, portfolio_value, contributed,
        where=(portfolio_value >= contributed),
        color="tab:green", alpha=0.10,
    )
    ax_abs.fill_between(
        portfolio_value.index, portfolio_value, contributed,
        where=(portfolio_value < contributed),
        color="tab:red", alpha=0.10,
    )

    ax_abs.set_title("Portfolio vs benchmarks — same cash, same dates")
    ax_abs.set_ylabel(f"Value ({currency})")
    ax_abs.legend(loc="upper left", fontsize=9)
    ax_abs.grid(alpha=0.3)

    # --- Percentage return ------------------------------------------------
    # Guard against the first days, when contributed capital may be zero or
    # tiny -- dividing by it produces a meaningless spike that would squash
    # the rest of the chart into a flat line.
    meaningful = contributed > 0

    port_pct = (portfolio_value[meaningful] / contributed[meaningful]) - 1
    ax_pct.plot(port_pct.index, port_pct * 100, label="Portfolio",
                linewidth=2.2, color="tab:blue", zorder=3)

    for ticker in benchmark_values.columns:
        bench_pct = (benchmark_values[ticker][meaningful] / contributed[meaningful]) - 1
        ax_pct.plot(
            bench_pct.index, bench_pct * 100,
            label=f"{labels.get(ticker, ticker)} ({ticker})",
            linewidth=1.5, alpha=0.85,
        )

    ax_pct.axhline(0, color="grey", linewidth=1.0, linestyle="--", alpha=0.8)
    ax_pct.set_title("Return on contributed capital")
    ax_pct.set_ylabel("Return (%)")
    ax_pct.legend(loc="upper left", fontsize=9)
    ax_pct.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark import DEFAULT_BENCHMARKS, benchmark_value_series, net_cashflows
    from ledger import build_ledger
    from loader import load_transactions
    from prices import get_prices
    from valuation import value_portfolio

    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
    txns = load_transactions(target)
    _, daily_holdings = build_ledger(txns)

    tickers = daily_holdings["ticker"].unique().tolist()
    start = str(txns["date"].min().date())

    price_data = get_prices(tickers, start=start)
    portfolio = value_portfolio(daily_holdings, price_data)

    bench_prices = get_prices(list(DEFAULT_BENCHMARKS), start=start)
    index = portfolio.index

    bench_values = benchmark_value_series(txns, bench_prices, index)
    contributed = net_cashflows(txns).reindex(index, fill_value=0.0).cumsum()

    path = plot_vs_benchmarks(
        portfolio["total_value"], bench_values, contributed,
        labels=DEFAULT_BENCHMARKS,
    )
    print(f"Saved chart to {path}")
