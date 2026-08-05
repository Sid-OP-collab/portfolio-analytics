"""
Turn daily holdings + daily prices into a portfolio value time series.

Usage:
    from loader import load_transactions
    from ledger import build_ledger
    from prices import get_prices
    from valuation import value_portfolio

    txns = load_transactions("data/my_transactions_clean.csv")
    _, daily_holdings = build_ledger(txns)

    tickers = daily_holdings["ticker"].unique().tolist()
    prices = get_prices(tickers, start=str(txns["date"].min().date()))

    portfolio_value = value_portfolio(daily_holdings, prices)
"""

import pandas as pd


def value_portfolio(daily_holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join holdings to prices and compute daily market value per position
    and in total.

    daily_holdings: output of ledger.build_ledger -- columns
        date, ticker, quantity, cost_basis, avg_cost
    prices: output of prices.get_prices -- index is date, one column per ticker

    Returns a DataFrame indexed by date with:
        total_value, total_cost_basis, total_unrealised_pnl
    plus one <TICKER>_value column per position, for the attribution step later.
    """
    holdings = daily_holdings.copy()
    holdings["date"] = pd.to_datetime(holdings["date"])

    # Long-format prices so we can merge on (date, ticker) directly, rather
    # than pivoting holdings wide -- this stays correct even as new tickers
    # enter or leave the portfolio over time.
    price_long = prices.reset_index().melt(
        id_vars=prices.index.name or "index", var_name="ticker", value_name="price"
    )
    price_long = price_long.rename(columns={price_long.columns[0]: "date"})
    price_long["date"] = pd.to_datetime(price_long["date"])

    merged = holdings.merge(price_long, on=["date", "ticker"], how="left")

    missing = merged[merged["price"].isna()]
    if not missing.empty:
        missing_tickers = missing["ticker"].unique().tolist()
        # Forward-fill within each ticker so a single missing trading day
        # (e.g. a holiday) doesn't zero out that day's value.
        merged["price"] = merged.groupby("ticker")["price"].ffill()

        # A leading gap can't be forward-filled -- it happens when a position
        # is opened on a weekend/holiday, before any trading-day price exists
        # yet for that ticker. Back-fill just that leading edge using the
        # next available price (typically the following Monday's close),
        # which is the best available estimate for "what was it worth" on
        # a day the market was closed.
        merged["price"] = merged.groupby("ticker")["price"].bfill()

        still_missing = merged["price"].isna().sum()
        if still_missing:
            print(
                f"Warning: {still_missing} rows still missing prices even after "
                f"forward- and back-fill, for tickers: {missing_tickers}. "
                "This means that ticker has no price data at all in the range "
                "requested -- check the ticker symbol and date range."
            )

    merged["market_value"] = merged["quantity"] * merged["price"]
    merged["unrealised_pnl"] = merged["market_value"] - merged["cost_basis"]

    daily_totals = (
        merged.groupby("date")[["market_value", "cost_basis", "unrealised_pnl"]]
        .sum()
        .rename(
            columns={
                "market_value": "total_value",
                "cost_basis": "total_cost_basis",
                "unrealised_pnl": "total_unrealised_pnl",
            }
        )
    )

    per_ticker_value = merged.pivot(index="date", columns="ticker", values="market_value")
    per_ticker_value.columns = [f"{t}_value" for t in per_ticker_value.columns]

    return daily_totals.join(per_ticker_value)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from ledger import build_ledger
    from loader import load_transactions
    from prices import get_prices

    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
    txns = load_transactions(target)
    _, daily_holdings = build_ledger(txns)

    tickers = daily_holdings["ticker"].unique().tolist()
    price_data = get_prices(tickers, start=str(txns["date"].min().date()))

    portfolio = value_portfolio(daily_holdings, price_data)
    print(portfolio[["total_value", "total_cost_basis", "total_unrealised_pnl"]].tail(10))
