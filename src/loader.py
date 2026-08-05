"""
Load and validate a transaction CSV in the project's canonical schema.

Canonical schema:
    date, ticker, action, quantity, price, fees
    - date     : YYYY-MM-DD
    - ticker   : e.g. "AAPL"
    - action   : BUY | SELL | DIV | SPLIT
    - quantity : shares for BUY/SELL/SPLIT; 0 for DIV
    - price    : per-share fill price for BUY/SELL; cash amount for DIV;
                 split ratio (e.g. 2 for a 2-for-1) for SPLIT
    - fees     : commission, in the same currency as price

Usage:
    from loader import load_transactions
    df = load_transactions("data/sample_transactions.csv")
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["date", "ticker", "action", "quantity", "price", "fees"]
VALID_ACTIONS = {"BUY", "SELL", "DIV", "SPLIT"}


class TransactionValidationError(ValueError):
    """Raised when the transaction file fails validation. Message lists every
    problem found, not just the first, so one fix-and-rerun cycle catches
    everything instead of playing whack-a-mole with one error at a time."""


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read a transaction CSV, validate it, and return it sorted by date.

    Raises TransactionValidationError listing every row-level problem found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No transaction file at {path}")

    df = pd.read_csv(path)
    errors = _check_columns(df)
    if errors:
        raise TransactionValidationError("\n".join(errors))

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["action"] = df["action"].astype(str).str.strip().str.upper()

    errors += _check_rows(df)
    if errors:
        raise TransactionValidationError("\n".join(errors))

    return df.sort_values("date").reset_index(drop=True)


def _check_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required column(s): {missing}"]
    return []


def _check_rows(df: pd.DataFrame) -> list[str]:
    """Row-level checks. Returns one message per problem, prefixed with the
    1-indexed row number as it would appear if opened in a spreadsheet
    (i.e. accounting for the header row), so it's easy to go find it."""
    errors = []

    for i, row in df.iterrows():
        line = i + 2  # +1 for 0-index, +1 for the header row

        if pd.isna(row["date"]):
            errors.append(f"Row {line}: unparseable date")

        if row["action"] not in VALID_ACTIONS:
            errors.append(f"Row {line}: unknown action '{row['action']}'")
            continue  # further checks assume a known action

        if row["action"] in {"BUY", "SELL"}:
            if pd.isna(row["quantity"]) or row["quantity"] <= 0:
                errors.append(f"Row {line}: {row['action']} needs quantity > 0")
            if pd.isna(row["price"]) or row["price"] < 0:
                errors.append(f"Row {line}: {row['action']} needs price >= 0")

        if row["action"] == "SPLIT" and (pd.isna(row["quantity"]) or row["quantity"] <= 0):
            errors.append(f"Row {line}: SPLIT needs a positive ratio in quantity")

    errors += _check_oversells(df)
    return errors


def _check_oversells(df: pd.DataFrame) -> list[str]:
    """Walk each ticker in date order and flag any SELL exceeding running
    holdings. Ignores SPLIT for simplicity at this stage — split-adjusted
    holdings are handled later in the ledger builder, not here."""
    errors = []
    for ticker, group in df[df["action"].isin(["BUY", "SELL"])].groupby("ticker"):
        held = 0.0
        for i, row in group.sort_values("date").iterrows():
            line = i + 2
            if row["action"] == "BUY":
                held += row["quantity"]
            else:
                if row["quantity"] > held + 1e-9:
                    errors.append(
                        f"Row {line}: SELL of {row['quantity']} {ticker} "
                        f"exceeds holding of {held:.6f}"
                    )
                held -= row["quantity"]
    return errors


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/sample_transactions.csv"
    result = load_transactions(target)
    print(f"Loaded {len(result)} transactions from {target}")
    print(result.head())
