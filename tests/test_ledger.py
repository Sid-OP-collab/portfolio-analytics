"""Tests for ledger.build_ledger -- the FIFO cost-basis engine.

Every expected value here is computed by hand in the test's docstring or
comments, so a failure tells you what the answer should have been rather
than just that two numbers differed. This is the module where a silent
error would be most damaging (wrong cost basis propagates into every
downstream metric), so the cases are deliberately arithmetic-heavy.
"""

import pandas as pd
import pytest

from ledger import build_ledger


def make_txns(rows: list[dict]) -> pd.DataFrame:
    """Build a transaction frame in the shape the loader would produce."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def test_single_buy_creates_holding():
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-03"))

    last = daily[daily["date"] == pd.Timestamp("2026-01-03")].iloc[0]
    assert last["quantity"] == 10
    assert last["cost_basis"] == pytest.approx(1000.0)
    assert last["avg_cost"] == pytest.approx(100.0)


def test_fifo_consumes_oldest_lot_first():
    """Buy 10 @ 100, then 10 @ 200, then sell 10.

    Under FIFO the sale must consume the *first* lot (the 100s), leaving
    10 shares at a 200 cost basis. Under average cost it would instead
    leave 10 shares at 150 -- so this test is what actually pins the
    accounting method down.
    """
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 200.0, "fees": 0.0},
        {"date": "2026-01-03", "ticker": "X", "action": "SELL",
         "quantity": 10, "price": 150.0, "fees": 0.0},
    ])
    trades, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-03"))

    remaining = daily[daily["date"] == pd.Timestamp("2026-01-03")].iloc[0]
    assert remaining["quantity"] == pytest.approx(10)
    assert remaining["avg_cost"] == pytest.approx(200.0)

    # Sold 10 @ 150 against a cost of 10 @ 100 -> +500 realised.
    sell_row = trades[trades["action"] == "SELL"].iloc[0]
    assert sell_row["realised_pnl"] == pytest.approx(500.0)
    assert sell_row["cost_basis_sold"] == pytest.approx(1000.0)


def test_partial_lot_consumption():
    """Buy 10 @ 100, buy 10 @ 200, sell 15.

    The sale eats all 10 of the first lot plus 5 of the second:
      cost basis sold = 10*100 + 5*200 = 2000
      proceeds        = 15 * 180      = 2700
      realised        = +700
    Remaining: 5 shares @ 200.
    """
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 200.0, "fees": 0.0},
        {"date": "2026-01-03", "ticker": "X", "action": "SELL",
         "quantity": 15, "price": 180.0, "fees": 0.0},
    ])
    trades, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-03"))

    sell_row = trades[trades["action"] == "SELL"].iloc[0]
    assert sell_row["cost_basis_sold"] == pytest.approx(2000.0)
    assert sell_row["realised_pnl"] == pytest.approx(700.0)

    remaining = daily[daily["date"] == pd.Timestamp("2026-01-03")].iloc[0]
    assert remaining["quantity"] == pytest.approx(5)
    assert remaining["avg_cost"] == pytest.approx(200.0)


def test_fees_reduce_realised_pnl():
    """Buy 10 @ 100 (no fee), sell 10 @ 110 with a 5.00 fee.
    Proceeds 1100 - 5 = 1095; cost 1000; realised = +95, not +100.
    """
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "SELL",
         "quantity": 10, "price": 110.0, "fees": 5.0},
    ])
    trades, _ = build_ledger(txns, as_of=pd.Timestamp("2026-01-02"))
    sell_row = trades[trades["action"] == "SELL"].iloc[0]
    assert sell_row["realised_pnl"] == pytest.approx(95.0)


def test_full_exit_removes_position():
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "SELL",
         "quantity": 10, "price": 110.0, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-05"))
    # After a full exit the ticker should not appear on later dates at all.
    assert daily[daily["date"] > pd.Timestamp("2026-01-02")].empty


def test_split_preserves_total_cost_basis():
    """A 2-for-1 split doubles share count and halves per-share cost, but
    total cost basis must be unchanged -- a split is not an economic event.
    """
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "SPLIT",
         "quantity": 2, "price": 0.0, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-02"))
    after = daily[daily["date"] == pd.Timestamp("2026-01-02")].iloc[0]

    assert after["quantity"] == pytest.approx(20)
    assert after["avg_cost"] == pytest.approx(50.0)
    assert after["cost_basis"] == pytest.approx(1000.0)


def test_dividend_does_not_change_share_count():
    """DIV rows carry cash, not shares. Getting this wrong would inflate
    holdings by the dividend amount -- a subtle, plausible bug."""
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "DIV",
         "quantity": 0, "price": 25.0, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-02"))
    after = daily[daily["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    assert after["quantity"] == pytest.approx(10)
    assert after["cost_basis"] == pytest.approx(1000.0)


def test_holdings_carry_forward_past_last_transaction():
    """A position you haven't sold is still held during a quiet period --
    the ledger must not simply stop at the final transaction date."""
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-06-01"))
    assert daily["date"].max() == pd.Timestamp("2026-06-01")
    assert daily[daily["date"] == pd.Timestamp("2026-06-01")].iloc[0]["quantity"] == 10


def test_as_of_never_truncates_real_history():
    """If as_of is earlier than the last transaction, real history still wins;
    silently dropping trades would be far worse than an over-long series."""
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-05-01", "ticker": "X", "action": "BUY",
         "quantity": 5, "price": 120.0, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-02-01"))
    assert daily["date"].max() >= pd.Timestamp("2026-05-01")


def test_multiple_tickers_are_independent():
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-01", "ticker": "B", "action": "BUY",
         "quantity": 5, "price": 200.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "A", "action": "SELL",
         "quantity": 10, "price": 110.0, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-02"))
    day2 = daily[daily["date"] == pd.Timestamp("2026-01-02")]

    assert set(day2["ticker"]) == {"B"}
    assert day2.iloc[0]["cost_basis"] == pytest.approx(1000.0)


def test_fractional_shares_fully_exit_without_residue():
    """Real broker data is full of long fractional quantities. Selling the
    exact amount held must close the position cleanly rather than leaving
    a floating-point dust position behind.
    """
    qty = 1.371721
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": qty, "price": 39.08, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "SELL",
         "quantity": qty, "price": 41.00, "fees": 0.0},
    ])
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-03"))
    assert daily[daily["date"] == pd.Timestamp("2026-01-03")].empty
