"""
Compute time-weighted return (TWR) and money-weighted return (XIRR) for
the portfolio.

These answer two different questions, and the gap between them is often
the most interesting number in the whole project:

  TWR strips out the effect of *when* you added or removed money, so it
  answers "how did my stock-picking perform, independent of timing" --
  useful for comparing against a benchmark index.

  XIRR (money-weighted) answers "what annualised return did I actually
  earn, given when I put money in and took it out" -- if you happened to
  add a lot of cash right before a rally, your XIRR will look better than
  your TWR even though your stock selection didn't change.

Known limitation: dividends are not modelled as a cash balance anywhere
in this project (see valuation.py), so they're excluded from both
calculations below. In practice this slightly understates both TWR and
XIRR relative to a broker's own reported returns, which typically assume
dividends are reinvested or held as cash.

Usage:
    from returns import compute_twr, compute_xirr

    twr = compute_twr(portfolio_value, transactions)
    xirr = compute_xirr(transactions, current_value, as_of=portfolio_value.index[-1])
"""

from datetime import datetime

import pandas as pd
from scipy.optimize import brentq


def compute_external_cashflows(transactions: pd.DataFrame) -> pd.Series:
    """Net cash contributed into the portfolio per day.

    Positive = investor put money in (a BUY). Negative = investor took
    money out (a SELL). Indexed by date; days with no trades are absent
    (not zero), since a Series aligned to a value index is expected to
    reindex/fillna(0) on the caller's side.
    """
    trades = transactions[transactions["action"].isin(["BUY", "SELL"])].copy()
    trades["cash_flow"] = trades.apply(
        lambda r: (r["quantity"] * r["price"] + r["fees"])
        if r["action"] == "BUY"
        else -(r["quantity"] * r["price"] - r["fees"]),
        axis=1,
    )
    return trades.groupby("date")["cash_flow"].sum()


def compute_twr(portfolio_value: pd.Series, transactions: pd.DataFrame) -> float:
    """Time-weighted return over the full span of `portfolio_value`.

    portfolio_value: daily total market value, indexed by date (e.g. the
        `total_value` column from valuation.value_portfolio).
    """
    cashflows = compute_external_cashflows(transactions).reindex(portfolio_value.index, fill_value=0.0)

    daily_returns = []
    for i in range(1, len(portfolio_value)):
        prev_value = portfolio_value.iloc[i - 1]
        curr_value = portfolio_value.iloc[i]
        cf = cashflows.iloc[i]

        if prev_value <= 0:
            # No position was held yet the day before -- there's no basis
            # to compute a return against, so this day is skipped rather
            # than treated as an undefined or infinite return.
            continue

        daily_return = (curr_value - prev_value - cf) / prev_value
        daily_returns.append(daily_return)

    linked = 1.0
    for r in daily_returns:
        linked *= 1 + r
    return linked - 1


def compute_xirr(
    transactions: pd.DataFrame, current_value: float, as_of: pd.Timestamp
) -> float | None:
    """Money-weighted (annualised) return: the single constant rate that
    discounts every historical cash flow, plus a hypothetical "sell
    everything today" flow of `current_value` on `as_of`, to zero.

    Returns None if no rate in a wide search range satisfies the equation
    (can happen with unusual cash-flow patterns, e.g. all flows on one day).
    """
    trades = transactions[transactions["action"].isin(["BUY", "SELL"])]

    flows = []
    for _, r in trades.iterrows():
        amount = (
            -(r["quantity"] * r["price"] + r["fees"])
            if r["action"] == "BUY"
            else (r["quantity"] * r["price"] - r["fees"])
        )
        flows.append((r["date"], amount))
    flows.append((pd.Timestamp(as_of), current_value))

    day0 = min(date for date, _ in flows)

    def npv(rate: float) -> float:
        return sum(
            amount / (1 + rate) ** ((date - day0).days / 365.0) for date, amount in flows
        )

    # Search a wide bracket since portfolios can plausibly show anywhere
    # from a near-total loss to a very high annualised return. Very short
    # holding periods (a few days) can genuinely require an extreme rate
    # to annualise correctly -- that's not a bug, it's what "annualised"
    # means when compounded out from a tiny window -- so the bracket is
    # deliberately generous rather than tight.
    low, high = -0.9999, 1_000_000.0
    try:
        if npv(low) * npv(high) > 0:
            return None
        return brentq(npv, low, high)
    except (ValueError, RuntimeError):
        return None


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
    price_data = get_prices(tickers, start=str(txns["date"].min().date()))
    portfolio = value_portfolio(daily_holdings, price_data)

    twr = compute_twr(portfolio["total_value"], txns)
    current_value = portfolio["total_value"].iloc[-1]
    xirr = compute_xirr(txns, current_value, as_of=portfolio.index[-1])

    print(f"Time-weighted return (TWR):   {twr:+.2%}")
    print(f"Money-weighted return (XIRR): {xirr:+.2%}" if xirr is not None else "XIRR: could not solve")
    print(f"Current value: {current_value:.2f}")
