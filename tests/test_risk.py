"""Tests for risk.py.

Two of these encode bugs that were caught during development and should
stay caught: the Sharpe-on-a-flat-series case (floating-point noise made a
zero-volatility series produce an astronomically large ratio), and the
attribution case where positions whose start/end dates fell on a weekend
were silently dropped from the table.
"""

import numpy as np
import pandas as pd
import pytest

from risk import (
    annualised_volatility,
    daily_returns,
    drawdown_duration,
    index_series,
    max_drawdown,
    position_contribution,
    rolling_beta,
    sharpe_ratio,
)

TRADING_DAYS = 252


def series(values, start="2026-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_daily_returns_drops_first_row():
    value = series([100, 110, 121])
    returns = daily_returns(value)
    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(0.10)


def test_volatility_matches_manual_calculation():
    """Alternating +2%/-2% days, checked against numpy's sample stdev."""
    pattern = [0.02, -0.02] * 5
    values = [100.0]
    for r in pattern:
        values.append(values[-1] * (1 + r))

    expected = np.std(pattern, ddof=1) * np.sqrt(TRADING_DAYS)
    assert annualised_volatility(series(values)) == pytest.approx(expected)


def test_volatility_of_constant_growth_is_zero():
    """Steady 1%/day compounding has no dispersion, so no volatility."""
    value = series([100 * 1.01**i for i in range(10)])
    assert annualised_volatility(value) == pytest.approx(0.0, abs=1e-9)


def test_sharpe_of_flat_series_is_nan():
    """Regression test. Zero volatility produces 1e-16 rather than exactly
    0.0 in floating point, so an `== 0` guard silently passed and returned
    a Sharpe ratio in the quadrillions. The guard needs a tolerance."""
    value = series([100 * 1.01**i for i in range(10)])
    assert np.isnan(sharpe_ratio(value))


def test_sharpe_rises_when_risk_free_rate_falls():
    """Sanity check on direction: a lower hurdle rate can only help."""
    rng = np.random.default_rng(0)
    value = series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 200)))
    assert sharpe_ratio(value, risk_free_rate=0.0) > sharpe_ratio(
        value, risk_free_rate=0.05
    )


def test_max_drawdown_peak_to_trough():
    """Rises to 120, falls to 70: 70/120 - 1 = -41.67%."""
    value = series([100, 110, 120, 100, 84, 70, 80, 90])
    dd, peak, trough = max_drawdown(value)

    assert dd == pytest.approx(-0.416667, abs=1e-5)
    assert peak == pd.Timestamp("2026-01-03")
    assert trough == pd.Timestamp("2026-01-06")


def test_max_drawdown_of_monotonic_rise_is_zero():
    dd, peak, trough = max_drawdown(series([100, 110, 120, 130]))
    assert dd == 0.0
    assert peak is None
    assert trough is None


def test_drawdown_duration_ends_at_recovery():
    """Dips below 100 on day 2 and regains it on day 4. Recovery counts as
    reaching the prior peak, not exceeding it, so the underwater stretch
    runs 01-02 to 01-04: two days."""
    value = series([100, 80, 90, 100, 105])
    assert drawdown_duration(value) == 2


def test_drawdown_duration_counts_unrecovered_tail():
    """A portfolio still underwater at the end has still spent that time
    underwater; the stretch must not be discarded for lacking a recovery.
    Underwater from 01-02 through the final day 01-04: two days."""
    value = series([100, 90, 85, 80])
    assert drawdown_duration(value) == 2


def test_rolling_beta_of_double_leverage_is_two():
    rng = np.random.default_rng(0)
    bench_returns = rng.normal(0, 0.01, 100)
    port_returns = bench_returns * 2.0

    bench = series(100 * np.cumprod(1 + bench_returns))
    port = series(100 * np.cumprod(1 + port_returns))

    beta = rolling_beta(port, bench, window=60)
    assert beta.dropna().iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_rolling_beta_of_identical_series_is_one():
    rng = np.random.default_rng(1)
    value = series(100 * np.cumprod(1 + rng.normal(0, 0.01, 100)))
    beta = rolling_beta(value, value, window=60)
    assert beta.dropna().iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_position_contribution_weights_by_starting_size():
    """Two equal positions, one doubles and one is flat: total contribution
    is 50%, split as 50 points from the winner and 0 from the other."""
    holdings = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "ticker": "A",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
        {"date": pd.Timestamp("2026-01-10"), "ticker": "A",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
        {"date": pd.Timestamp("2026-01-01"), "ticker": "B",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
        {"date": pd.Timestamp("2026-01-10"), "ticker": "B",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
    ])
    prices = pd.DataFrame(
        {"A": [100, 200], "B": [100, 100]},
        index=pd.to_datetime(["2026-01-01", "2026-01-10"]),
    )

    contrib = position_contribution(holdings, prices)
    by_ticker = contrib.set_index("ticker")

    assert by_ticker.loc["A", "weight"] == pytest.approx(0.5)
    assert by_ticker.loc["A", "position_return"] == pytest.approx(1.0)
    assert by_ticker.loc["A", "contribution"] == pytest.approx(0.5)
    assert by_ticker.loc["B", "contribution"] == pytest.approx(0.0)
    assert contrib["contribution"].sum() == pytest.approx(0.5)


def test_position_contribution_keeps_positions_ending_on_a_weekend():
    """Regression test. A position sold on a Monday was last *held* on the
    preceding Sunday, which has no market price. The original lookup
    required an exact date match and silently dropped every such position
    -- which was most of them on real data.
    """
    saturday = pd.Timestamp("2026-01-03")
    holdings = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "ticker": "A",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
        {"date": saturday, "ticker": "A",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
    ])
    # Weekday-only prices, exactly as a market data feed would supply them.
    prices = pd.DataFrame(
        {"A": [100, 120]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )

    contrib = position_contribution(holdings, prices)
    assert list(contrib["ticker"]) == ["A"]
    assert contrib.iloc[0]["position_return"] == pytest.approx(0.20)


def test_position_contribution_skips_ticker_with_no_price_data():
    holdings = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "ticker": "GHOST",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
        {"date": pd.Timestamp("2026-01-02"), "ticker": "GHOST",
         "quantity": 10, "cost_basis": 1000, "avg_cost": 100},
    ])
    prices = pd.DataFrame(
        {"A": [100, 120]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    assert position_contribution(holdings, prices).empty


# --- Cash-flow adjustment -------------------------------------------------
#
# These pin down the fix for the bug where risk metrics were computed on raw
# portfolio value. A deposit lifts total value instantly; treating that as a
# one-day return inflates volatility, flatters Sharpe, and can erase a real
# drawdown by making the line go up. For a portfolio funded by regular
# contributions -- which is the normal case -- the unadjusted figures are
# not merely imprecise, they measure the wrong thing entirely.


def txns(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_deposit_is_not_counted_as_a_return():
    """Value doubles purely because cash was added; prices never moved."""
    value = series([1000.0, 2000.0])
    trades = txns([
        {"date": "2026-01-02", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    assert daily_returns(value, trades).iloc[0] == pytest.approx(0.0, abs=1e-12)
    # Without the adjustment the same series reads as a 100% gain.
    assert daily_returns(value).iloc[0] == pytest.approx(1.0)


def test_deposit_does_not_create_volatility():
    """A flat market with one deposit has zero volatility, not a spike."""
    value = series([1000.0, 2000.0, 2000.0, 2000.0])
    trades = txns([
        {"date": "2026-01-02", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    assert annualised_volatility(value, trades) == pytest.approx(0.0, abs=1e-9)
    assert annualised_volatility(value) > 1.0  # the unadjusted figure is huge


def test_withdrawal_is_not_counted_as_a_loss():
    value = series([2000.0, 1000.0])
    trades = txns([
        {"date": "2026-01-02", "ticker": "A", "action": "SELL",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    assert daily_returns(value, trades).iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_deposit_does_not_erase_a_real_drawdown():
    """Prices fall 20%, then a deposit lifts total value back above the old
    peak. On raw value that looks like a full recovery; the investor still
    lost 20% on what they held."""
    value = series([1000.0, 800.0, 1800.0])
    trades = txns([
        {"date": "2026-01-03", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    dd_adjusted, _, _ = max_drawdown(value, trades)
    assert dd_adjusted == pytest.approx(-0.20, abs=1e-9)

    # The adjusted series must still sit below its opening peak at the end;
    # the deposit lifted raw value to a new high but recovered nothing.
    adjusted = index_series(value, trades)
    assert adjusted.iloc[-1] == pytest.approx(80.0, abs=1e-9)
    assert adjusted.iloc[0] == pytest.approx(100.0)


def test_index_series_tracks_market_only():
    """Two 10% market gains with a deposit in between compound to +21% on
    the index, regardless of how much cash was added along the way."""
    # 100 -> 110 (+10%), then +1000 deposited with no price move,
    # then 1110 -> 1221 (+10%). Index: 100 * 1.1 * 1.0 * 1.1 = 121.
    value = series([100.0, 110.0, 1110.0, 1221.0])
    trades = txns([
        {"date": "2026-01-03", "ticker": "A", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 0.0},
    ])
    idx = index_series(value, trades, base=100.0)
    assert idx.iloc[-1] == pytest.approx(121.0, abs=1e-6)


def test_returns_skip_days_before_any_position_exists():
    value = series([0.0, 0.0, 100.0, 110.0])
    trades = txns([
        {"date": "2026-01-03", "ticker": "A", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    returns = daily_returns(value, trades)
    assert len(returns) == 1
    assert returns.iloc[0] == pytest.approx(0.10)
