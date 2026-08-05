"""
Convert a raw broker transaction export into the project's canonical schema.

Raw schema (broker export):
    Symbol, Side, Qty, Fill Price, Commission, Closing Time
    e.g. "NASDAQ:IREN, Buy, 1.371721, 39.08, 0, 02-07-2026 15:52"

Canonical schema (used by everything downstream):
    date, ticker, action, quantity, price, fees
    - date     : ISO YYYY-MM-DD
    - ticker   : exchange prefix stripped
    - action   : BUY | SELL | DIV
    - quantity : shares for BUY/SELL; 0 for DIV
    - price    : per-share fill price for BUY/SELL; total cash received for DIV
    - fees     : commission

Usage:
    python clean_transactions.py data/my_real_transactions.csv data/my_real_transactions_clean.csv
"""

import sys
from pathlib import Path

import pandas as pd

ACTION_MAP = {"buy": "BUY", "sell": "SELL", "dividend": "DIV"}


def load_raw(path: Path) -> pd.DataFrame:
    """Read the broker export, leaving all values as strings for safe parsing."""
    return pd.read_csv(path, dtype=str, skip_blank_lines=True).dropna(how="all")


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the broker export onto the canonical schema."""
    df = pd.DataFrame()

    # "NASDAQ:IREN" -> "IREN". Tickers without a prefix pass through unchanged.
    df["ticker"] = raw["Symbol"].str.strip().str.split(":").str[-1]

    df["action"] = raw["Side"].str.strip().str.lower().map(ACTION_MAP)
    unknown = raw.loc[df["action"].isna(), "Side"].unique()
    if len(unknown):
        raise ValueError(f"Unrecognised Side values: {list(unknown)}")

    # Broker timestamps are DD-MM-YYYY HH:MM; we only keep the calendar date.
    ts = pd.to_datetime(raw["Closing Time"].str.strip(), format="%d-%m-%Y %H:%M")
    df["date"] = ts.dt.strftime("%Y-%m-%d")

    qty = pd.to_numeric(raw["Qty"], errors="coerce").fillna(0.0)
    price = pd.to_numeric(raw["Fill Price"], errors="coerce").fillna(0.0)
    fees = pd.to_numeric(raw["Commission"], errors="coerce").fillna(0.0)

    is_div = df["action"].eq("DIV")

    # For dividends the broker puts the cash amount in Qty and leaves Fill Price
    # blank. Canonical form carries that cash in `price` and zeroes `quantity`,
    # so no downstream code mistakes a dividend for a change in share count.
    df["quantity"] = qty.where(~is_div, 0.0)
    df["price"] = price.where(~is_div, qty)
    df["fees"] = fees

    df = df[["date", "ticker", "action", "quantity", "price", "fees"]]
    return df.sort_values(["date", "ticker", "action"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable warnings. Does not raise."""
    warnings: list[str] = []

    trades = df[df["action"].isin(["BUY", "SELL"])]
    if (trades["quantity"] <= 0).any():
        bad = trades[trades["quantity"] <= 0]
        warnings.append(f"{len(bad)} trade row(s) with non-positive quantity")

    suspicious = trades[(trades["action"] == "BUY") & (trades["price"] < 1)]
    for _, row in suspicious.iterrows():
        warnings.append(
            f"{row['date']} {row['ticker']}: BUY at {row['price']} "
            "- unusually low, check if this is a dividend reinvestment"
        )

    # Walk each ticker chronologically and flag any sale exceeding holdings.
    for ticker, group in df.groupby("ticker"):
        held = 0.0
        for _, row in group.sort_values("date").iterrows():
            if row["action"] == "BUY":
                held += row["quantity"]
            elif row["action"] == "SELL":
                if row["quantity"] > held + 1e-9:
                    warnings.append(
                        f"{row['date']} {ticker}: SELL of {row['quantity']:.6f} "
                        f"exceeds holding of {held:.6f}"
                    )
                held -= row["quantity"]

    return warnings


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    clean = transform(load_raw(src))
    clean.to_csv(dest, index=False)

    print(f"Wrote {len(clean)} rows to {dest}")
    print(clean["action"].value_counts().to_string())
    print(f"Date range: {clean['date'].min()} to {clean['date'].max()}")
    print(f"Tickers: {clean['ticker'].nunique()}")

    for warning in validate(clean):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
