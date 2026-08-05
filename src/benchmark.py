"""
Compare portfolio performance against benchmark indices.

The comparison implemented here is a *cash-flow-matched counterfactual*:
it simulates buying the benchmark with exactly the same money, on exactly
the same dates, that actually went into the portfolio. That is the honest
question -- "what if I'd bought the index instead?" -- and it is not the
same as comparing against the index's headline return.

Why the headline comparison misleads: an index that rose 12% over the
period did not rise 12% on money that only arrived last month. If
contributions were made gradually (the normal case), quoting the index's
full-period return overstates what an index investor with the same
deposit schedule would actually have earned. Matching the cash flows
removes that distortion from both sides.

Default benchmarks are ETFs rather than raw indices (SPY over ^GSPC,
QQQ over ^NDX) because ETF prices are what an investor could actually
have bought, and with auto_adjust they include dividends -- an index
price level does not, which would quietly flatter the portfolio.

Usage:
    from benchmark import compare_to_benchmarks

    table = compare_to_benchmarks(portfolio_value, transactions)
    print(table)
"""

import pandas as pd

DEFAULT_BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
}


def net_cashflows(transactions: pd.DataFrame) -> pd.Series:
    """Net external cash into the portfolio per day.

    Positive = money in (BUY), negative = money out (SELL). Dividends are
    excluded: they are internally generated, not new money the investor
    contributed, so an index investor with a matched deposit schedule
    would not have contributed them either.
    """
    trades = transactions[transactions["action"].isin(["BUY", "SELL"])]
    if trades.empty:
        return pd.Series(dtype=float)

    signed = trades.apply(
        lambda r: (r["quantity"] * r["price"] + r["fees"])
        if r["action"] == "BUY"
        else -(r["quantity"] * r["price"] - r["fees"]),
        axis=1,
    )
    return signed.groupby(trades["date"]).sum().sort_index()


def simulate_benchmark(
    transactions: pd.DataFrame, benchmark_prices: pd.Series, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Simulate investing the portfolio's cash flows into one benchmark.

    Returns a DataFrame indexed by date with `units` held, `value`, and
    cumulative `contributed`. Fractional units are assumed (as with any
    modern broker); no trading costs are modelled on the benchmark side,
    which if anything is generous to the benchmark.
    """
    prices = benchmark_prices.reindex(index).ffill().bfill()
    flows = net_cashflows(transactions).reindex(index, fill_value=0.0)

    # Units bought (or sold) each day at that day's price, accumulated.
    units_delta = flows / prices
    units = units_delta.cumsum()

    # A withdrawal larger than the simulated holding would imply shorting
    # the index, which is not a meaningful counterfactual -- floor at zero
    # and let the value series reflect a fully liquidated position.
    units = units.clip(lower=0.0)

    return pd.DataFrame(
        {
            "units": units,
            "price": prices,
            "value": units * prices,
            "contributed": flows.cumsum(),
        }
    )


def compare_to_benchmarks(
    portfolio_value: pd.Series,
    transactions: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a comparison table: the portfolio against each benchmark,
    on matched cash flows.

    portfolio_value: daily total market value (valuation.value_portfolio).
    benchmark_prices: one column per benchmark ticker, indexed by date.

    Returns a table with, for each row: final value, total contributed,
    absolute profit/loss, and percentage return on contributed capital.
    Rows are the portfolio plus one per benchmark, so the numbers sit
    side by side on identical inputs.
    """
    labels = labels or DEFAULT_BENCHMARKS
    index = portfolio_value.index

    flows = net_cashflows(transactions).reindex(index, fill_value=0.0)
    total_contributed = flows.cumsum().iloc[-1]

    rows = [
        {
            "name": "Portfolio",
            "final_value": portfolio_value.iloc[-1],
            "contributed": total_contributed,
            "profit_loss": portfolio_value.iloc[-1] - total_contributed,
        }
    ]

    for ticker in benchmark_prices.columns:
        sim = simulate_benchmark(transactions, benchmark_prices[ticker], index)
        final_value = sim["value"].iloc[-1]
        rows.append(
            {
                "name": f"{labels.get(ticker, ticker)} ({ticker})",
                "final_value": final_value,
                "contributed": sim["contributed"].iloc[-1],
                "profit_loss": final_value - sim["contributed"].iloc[-1],
            }
        )

    table = pd.DataFrame(rows)
    table["return_pct"] = table["profit_loss"] / table["contributed"]

    # Everything is measured against the same contributed capital, so the
    # difference in profit is directly comparable in currency terms.
    portfolio_pnl = table.loc[0, "profit_loss"]
    table["vs_portfolio"] = portfolio_pnl - table["profit_loss"]
    table.loc[0, "vs_portfolio"] = 0.0

    return table


def benchmark_value_series(
    transactions: pd.DataFrame, benchmark_prices: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Daily simulated value of each benchmark on matched cash flows --
    for plotting the portfolio against the benchmarks over time."""
    return pd.DataFrame(
        {
            ticker: simulate_benchmark(transactions, benchmark_prices[ticker], index)["value"]
            for ticker in benchmark_prices.columns
        }
    )


def format_comparison(table: pd.DataFrame, currency: str = "$") -> str:
    """Render the comparison table for terminal output."""
    lines = []
    width = max(len(name) for name in table["name"]) + 2

    header = (
        f"{'':<{width}}{'Final value':>15}{'Contributed':>15}"
        f"{'P/L':>15}{'Return':>10}{'vs Portfolio':>16}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, row in table.iterrows():
        # Build each cell as a finished string first; prefixing the currency
        # symbol onto an already-padded number pushes the columns apart.
        final = f"{currency}{row['final_value']:,.2f}"
        contributed = f"{currency}{row['contributed']:,.2f}"
        pnl = f"{currency}{row['profit_loss']:+,.2f}"
        vs = (
            ""
            if row["name"] == "Portfolio"
            else f"{currency}{row['vs_portfolio']:+,.2f}"
        )

        lines.append(
            f"{row['name']:<{width}}"
            f"{final:>15}{contributed:>15}{pnl:>15}"
            f"{row['return_pct']:>+10.2%}{vs:>16}"
        )

    lines.append("")
    lines.append(
        "Benchmarks simulate investing the same cash on the same dates. "
        "'vs Portfolio' is how much more (+) or less (-) the portfolio made."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
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

    bench_tickers = list(DEFAULT_BENCHMARKS)
    bench_prices = get_prices(bench_tickers, start=start)

    table = compare_to_benchmarks(portfolio["total_value"], txns, bench_prices)
    print(format_comparison(table))
