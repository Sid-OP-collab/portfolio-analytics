"""Tests for valuation.value_portfolio.

Most of the risk here is in the join: holdings exist for every calendar day,
but prices only exist for trading days. Getting that wrong produces a value
series with holes or zeroes on weekends, which then corrupts every downstream
volatility and drawdown figure.
"""

import pandas as pd
import pytest

from valuation import value_portfolio


def holdings(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_market_value_is_quantity_times_price():
    held = holdings([
        {"date": "2026-01-01", "ticker": "A",
         "quantity": 10, "cost_basis": 1000.0, "avg_cost": 100.0},
    ])
    prices = pd.DataFrame({"A": [120.0]}, index=pd.to_datetime(["2026-01-01"]))

    result = value_portfolio(held, prices)
    assert result.loc[pd.Timestamp("2026-01-01"), "total_value"] == pytest.approx(1200.0)


def test_unrealised_pnl_is_value_minus_cost():
    held = holdings([
        {"date": "2026-01-01", "ticker": "A",
         "quantity": 10, "cost_basis": 1000.0, "avg_cost": 100.0},
    ])
    prices = pd.DataFrame({"A": [120.0]}, index=pd.to_datetime(["2026-01-01"]))

    result = value_portfolio(held, prices)
    row = result.loc[pd.Timestamp("2026-01-01")]
    assert row["total_cost_basis"] == pytest.approx(1000.0)
    assert row["total_unrealised_pnl"] == pytest.approx(200.0)


def test_totals_sum_across_multiple_positions():
    held = holdings([
        {"date": "2026-01-01", "ticker": "A",
         "quantity": 10, "cost_basis": 1000.0, "avg_cost": 100.0},
        {"date": "2026-01-01", "ticker": "B",
         "quantity": 5, "cost_basis": 500.0, "avg_cost": 100.0},
    ])
    prices = pd.DataFrame(
        {"A": [120.0], "B": [80.0]}, index=pd.to_datetime(["2026-01-01"])
    )

    result = value_portfolio(held, prices)
    # 10*120 + 5*80 = 1600
    assert result.loc[pd.Timestamp("2026-01-01"), "total_value"] == pytest.approx(1600.0)
    assert result.loc[pd.Timestamp("2026-01-01"), "total_unrealised_pnl"] == pytest.approx(100.0)


def test_weekend_carries_friday_price_forward():
    """Markets are closed at the weekend but the position is still held.
    Saturday and Sunday must inherit Friday's close rather than going blank."""
    friday = pd.Timestamp("2026-01-02")
    held = holdings([
        {"date": d, "ticker": "A", "quantity": 10,
         "cost_basis": 1000.0, "avg_cost": 100.0}
        for d in ["2026-01-02", "2026-01-03", "2026-01-04"]
    ])
    prices = pd.DataFrame({"A": [120.0]}, index=[friday])

    result = value_portfolio(held, prices)
    assert result["total_value"].notna().all()
    assert result.loc[pd.Timestamp("2026-01-04"), "total_value"] == pytest.approx(1200.0)


def test_position_opened_before_first_trading_price_is_backfilled():
    """Regression test. A position opened on a Saturday has no earlier price
    to carry forward, so forward-fill alone leaves it blank. The next
    trading day's price is the best available estimate for the weekend."""
    saturday = pd.Timestamp("2026-01-03")
    monday = pd.Timestamp("2026-01-05")
    held = holdings([
        {"date": saturday, "ticker": "A", "quantity": 10,
         "cost_basis": 1000.0, "avg_cost": 100.0},
        {"date": monday, "ticker": "A", "quantity": 10,
         "cost_basis": 1000.0, "avg_cost": 100.0},
    ])
    prices = pd.DataFrame({"A": [130.0]}, index=[monday])

    result = value_portfolio(held, prices)
    assert result.loc[saturday, "total_value"] == pytest.approx(1300.0)


def test_per_ticker_value_columns_are_emitted():
    """Attribution downstream depends on these columns existing per position."""
    held = holdings([
        {"date": "2026-01-01", "ticker": "A",
         "quantity": 10, "cost_basis": 1000.0, "avg_cost": 100.0},
        {"date": "2026-01-01", "ticker": "B",
         "quantity": 5, "cost_basis": 500.0, "avg_cost": 100.0},
    ])
    prices = pd.DataFrame(
        {"A": [120.0], "B": [80.0]}, index=pd.to_datetime(["2026-01-01"])
    )

    result = value_portfolio(held, prices)
    assert "A_value" in result.columns
    assert "B_value" in result.columns
    assert result.loc[pd.Timestamp("2026-01-01"), "A_value"] == pytest.approx(1200.0)


def test_value_tracks_price_changes_over_time():
    held = holdings([
        {"date": d, "ticker": "A", "quantity": 10,
         "cost_basis": 1000.0, "avg_cost": 100.0}
        for d in ["2026-01-01", "2026-01-02", "2026-01-03"]
    ])
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 90.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )

    result = value_portfolio(held, prices)
    assert list(result["total_value"]) == pytest.approx([1000.0, 1100.0, 900.0])
