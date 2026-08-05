"""
Build a daily holdings ledger from a transaction log, using FIFO lot matching.

Given a transaction DataFrame (see loader.py for schema), this produces:
  1. A trade log where every SELL is broken down into which BUY lot(s) it
     consumed, with realised profit/loss computed per lot.
  2. A daily snapshot of holdings per ticker: shares held, cost basis,
     and (once merged with prices) market value and unrealised P&L.

FIFO means: when you sell, the shares sold are assumed to come from your
*oldest* still-open purchase lots first. This is a choice, not the only
correct answer (average cost is the common alternative) -- FIFO is used
here because it matches how tax authorities in most jurisdictions treat
disposals by default, and because it makes "which lot did this sale close"
an unambiguous, auditable question.

Usage:
    from loader import load_transactions
    from ledger import build_ledger

    transactions = load_transactions("data/sample_transactions.csv")
    trade_log, daily_holdings = build_ledger(transactions)
"""

from collections import defaultdict, deque

import pandas as pd


class Lot:
    """A single open purchase lot: some quantity bought at some price."""

    __slots__ = ("date", "quantity", "price")

    def __init__(self, date: pd.Timestamp, quantity: float, price: float):
        self.date = date
        self.quantity = quantity
        self.price = price


def build_ledger(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process transactions in date order and return (trade_log, daily_holdings).

    trade_log: one row per transaction, annotated with realised P&L for sells.
    daily_holdings: one row per (date, ticker) with quantity held and cost basis,
        covering every calendar day from the first transaction to the last
        (holdings carried forward on days with no activity for that ticker).
    """
    open_lots: dict[str, deque[Lot]] = defaultdict(deque)
    trade_rows = []

    for _, txn in transactions.iterrows():
        ticker = txn["ticker"]

        if txn["action"] == "BUY":
            open_lots[ticker].append(Lot(txn["date"], txn["quantity"], txn["price"]))
            trade_rows.append({**txn, "realised_pnl": None, "cost_basis_sold": None})

        elif txn["action"] == "SELL":
            realised, cost_basis_sold = _consume_fifo(
                open_lots[ticker], txn["quantity"], txn["price"], txn["fees"]
            )
            trade_rows.append(
                {**txn, "realised_pnl": realised, "cost_basis_sold": cost_basis_sold}
            )

        elif txn["action"] == "SPLIT":
            ratio = txn["quantity"]
            for lot in open_lots[ticker]:
                lot.quantity *= ratio
                lot.price /= ratio
            trade_rows.append({**txn, "realised_pnl": None, "cost_basis_sold": None})

        elif txn["action"] == "DIV":
            trade_rows.append({**txn, "realised_pnl": None, "cost_basis_sold": None})

    trade_log = pd.DataFrame(trade_rows)
    daily_holdings = _snapshot_daily(transactions, open_lots, trade_log)
    return trade_log, daily_holdings


def _consume_fifo(
    lots: deque[Lot], sell_qty: float, sell_price: float, fees: float
) -> tuple[float, float]:
    """Remove sell_qty shares from the oldest lots first (mutating `lots`).

    Returns (realised_pnl, cost_basis_of_shares_sold). Raises ValueError if
    there isn't enough held -- this should already be caught by the loader's
    oversell check, so hitting it here means the two checks have drifted
    out of sync, which is itself worth knowing about.
    """
    remaining = sell_qty
    cost_basis_sold = 0.0

    while remaining > 1e-9:
        if not lots:
            raise ValueError(f"Oversold: {remaining} shares with no open lots left")
        lot = lots[0]
        take = min(lot.quantity, remaining)
        cost_basis_sold += take * lot.price
        lot.quantity -= take
        remaining -= take
        if lot.quantity <= 1e-9:
            lots.popleft()

    proceeds = sell_qty * sell_price - fees
    realised_pnl = proceeds - cost_basis_sold
    return realised_pnl, cost_basis_sold


def _snapshot_daily(
    transactions: pd.DataFrame,
    final_open_lots: dict[str, deque[Lot]],
    trade_log: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct, for every calendar day and every ticker ever traded,
    the shares held and their cost basis -- by replaying transactions
    day by day rather than relying on the already-mutated `final_open_lots`.
    """
    tickers = transactions["ticker"].unique()
    all_days = pd.date_range(transactions["date"].min(), transactions["date"].max(), freq="D")

    # Replay independently of build_ledger's mutation, so this function
    # is correct even if called on its own.
    running_lots: dict[str, deque[Lot]] = defaultdict(deque)
    rows = []

    txns_by_day = transactions.groupby("date")

    for day in all_days:
        if day in txns_by_day.groups:
            for _, txn in txns_by_day.get_group(day).iterrows():
                ticker = txn["ticker"]
                if txn["action"] == "BUY":
                    running_lots[ticker].append(Lot(txn["date"], txn["quantity"], txn["price"]))
                elif txn["action"] == "SELL":
                    _consume_fifo(running_lots[ticker], txn["quantity"], txn["price"], txn["fees"])
                elif txn["action"] == "SPLIT":
                    ratio = txn["quantity"]
                    for lot in running_lots[ticker]:
                        lot.quantity *= ratio
                        lot.price /= ratio
                # DIV affects cash, not share count -- no lot change here.

        for ticker in tickers:
            lots = running_lots[ticker]
            qty = sum(lot.quantity for lot in lots)
            cost_basis = sum(lot.quantity * lot.price for lot in lots)
            if qty > 1e-9:
                rows.append(
                    {
                        "date": day,
                        "ticker": ticker,
                        "quantity": qty,
                        "cost_basis": cost_basis,
                        "avg_cost": cost_basis / qty,
                    }
                )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from loader import load_transactions

    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
    txns = load_transactions(target)
    trades, daily = build_ledger(txns)

    print("--- Trade log (sells with realised P&L) ---")
    sells = trades[trades["action"] == "SELL"]
    print(sells[["date", "ticker", "quantity", "price", "realised_pnl"]])

    print("\n--- Latest holdings ---")
    latest_day = daily["date"].max()
    print(daily[daily["date"] == latest_day][["ticker", "quantity", "avg_cost", "cost_basis"]])
