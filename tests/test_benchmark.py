"""Tests for benchmark.py.

The comparison only means anything if the simulated benchmark receives
exactly the same money on exactly the same dates as the portfolio did.
These tests pin that down, and pin down the cases where a naive
implementation would quietly flatter one side or the other.
"""

import pandas as pd
import pytest

from benchmark import (
    compare_to_benchmarks,
    net_cashflows,
    simulate_benchmark,
)


def txns(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def dates(n: int, start: str = "2026-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


def test_cashflows_exclude_dividends():
    """Dividends are internally generated, not contributed capital. Counting
    them would credit the benchmark with money the investor never added."""
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "A", "action": "DIV",
         "quantity": 0, "price": 50.0, "fees": 0.0},
    ])
    flows = net_cashflows(trades)
    assert flows.sum() == pytest.approx(1000.0)


def test_cashflow_signs_and_fees():
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 5.0},
        {"date": "2026-01-02", "ticker": "A", "action": "SELL",
         "quantity": 5, "price": 100.0, "fees": 5.0},
    ])
    flows = net_cashflows(trades)
    assert flows.iloc[0] == pytest.approx(1005.0)   # fee increases money in
    assert flows.iloc[1] == pytest.approx(-495.0)   # fee reduces money out


def test_single_deposit_tracks_index_return():
    """1000 invested at a price of 100 buys 10 units. If the index then
    rises to 130, the simulated holding is worth 1300."""
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    idx = dates(3)
    prices = pd.Series([100.0, 110.0, 130.0], index=idx)

    sim = simulate_benchmark(trades, prices, idx)
    assert sim["units"].iloc[0] == pytest.approx(10.0)
    assert sim["value"].iloc[-1] == pytest.approx(1300.0)
    assert sim["contributed"].iloc[-1] == pytest.approx(1000.0)


def test_later_deposit_does_not_earn_earlier_gains():
    """The central correctness property. Money deposited on day 3 must not
    receive the index's day 1-2 gains -- which is exactly what comparing
    against the index's headline return would wrongly do.

    Deposit 1000 at price 100 -> 10 units.
    Index rises to 200 by day 3; deposit another 1000 -> 5 units.
    Total 15 units at 200 = 3000, on 2000 contributed. Profit 1000.
    A headline comparison would claim the index doubled, implying 2000
    profit on the same money.
    """
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-03", "ticker": "A", "action": "BUY",
         "quantity": 5, "price": 200.0, "fees": 0.0},
    ])
    idx = dates(3)
    prices = pd.Series([100.0, 150.0, 200.0], index=idx)

    sim = simulate_benchmark(trades, prices, idx)
    assert sim["units"].iloc[-1] == pytest.approx(15.0)
    assert sim["value"].iloc[-1] == pytest.approx(3000.0)
    assert sim["contributed"].iloc[-1] == pytest.approx(2000.0)


def test_withdrawal_reduces_simulated_units():
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "A", "action": "SELL",
         "quantity": 5, "price": 100.0, "fees": 0.0},
    ])
    idx = dates(2)
    prices = pd.Series([100.0, 100.0], index=idx)

    sim = simulate_benchmark(trades, prices, idx)
    # 1000 in buys 10 units; 500 out at the same price sells 5.
    assert sim["units"].iloc[-1] == pytest.approx(5.0)
    assert sim["value"].iloc[-1] == pytest.approx(500.0)


def test_units_never_go_negative():
    """Withdrawing more than was contributed would imply shorting the index,
    which isn't a meaningful counterfactual."""
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "A", "action": "SELL",
         "quantity": 1, "price": 500.0, "fees": 0.0},
    ])
    idx = dates(2)
    prices = pd.Series([100.0, 100.0], index=idx)

    sim = simulate_benchmark(trades, prices, idx)
    assert (sim["units"] >= 0).all()


def test_benchmark_uses_weekend_carry_forward():
    """Cash flows can land on days the market was closed; the simulation
    must use the nearest available price rather than dropping the flow."""
    idx = dates(3, start="2026-01-03")  # Sat, Sun, Mon
    trades = txns([
        {"date": "2026-01-03", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    # Only Monday has a price.
    prices = pd.Series([120.0], index=[pd.Timestamp("2026-01-05")])

    sim = simulate_benchmark(trades, prices, idx)
    assert sim["value"].notna().all()
    assert sim["units"].iloc[-1] == pytest.approx(1000.0 / 120.0)


def test_comparison_table_shape_and_arithmetic():
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    idx = dates(2)
    portfolio_value = pd.Series([1000.0, 1500.0], index=idx)
    bench = pd.DataFrame({"SPY": [100.0, 110.0]}, index=idx)

    table = compare_to_benchmarks(portfolio_value, trades, bench)

    portfolio_row = table[table["name"] == "Portfolio"].iloc[0]
    assert portfolio_row["profit_loss"] == pytest.approx(500.0)
    assert portfolio_row["return_pct"] == pytest.approx(0.50)

    spy_row = table[table["name"].str.contains("SPY")].iloc[0]
    assert spy_row["profit_loss"] == pytest.approx(100.0)
    assert spy_row["return_pct"] == pytest.approx(0.10)
    # Portfolio made 500 vs the benchmark's 100 on identical capital.
    assert spy_row["vs_portfolio"] == pytest.approx(400.0)


def test_comparison_reports_underperformance_as_negative():
    """Sign convention: `vs_portfolio` must go negative when the benchmark
    wins, so the column reads as 'how much better the portfolio did'."""
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    idx = dates(2)
    portfolio_value = pd.Series([1000.0, 1050.0], index=idx)   # +5%
    bench = pd.DataFrame({"SPY": [100.0, 130.0]}, index=idx)   # +30%

    table = compare_to_benchmarks(portfolio_value, trades, bench)
    spy_row = table[table["name"].str.contains("SPY")].iloc[0]

    assert spy_row["profit_loss"] == pytest.approx(300.0)
    assert spy_row["vs_portfolio"] == pytest.approx(-250.0)


def test_multiple_benchmarks_are_independent():
    trades = txns([
        {"date": "2026-01-01", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    idx = dates(2)
    portfolio_value = pd.Series([1000.0, 1200.0], index=idx)
    bench = pd.DataFrame(
        {"SPY": [100.0, 110.0], "QQQ": [50.0, 60.0]}, index=idx
    )

    table = compare_to_benchmarks(portfolio_value, trades, bench)
    assert len(table) == 3  # portfolio + two benchmarks

    qqq_row = table[table["name"].str.contains("QQQ")].iloc[0]
    # 1000 at 50 -> 20 units; at 60 -> 1200. Profit 200.
    assert qqq_row["profit_loss"] == pytest.approx(200.0)
