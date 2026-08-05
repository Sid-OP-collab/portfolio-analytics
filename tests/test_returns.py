"""Tests for returns.compute_twr and returns.compute_xirr.

The headline case is test_twr_and_xirr_diverge_with_deposit_timing: TWR and
XIRR are *supposed* to disagree when contributions are unevenly timed, and
a version of this code that made them agree would be quietly wrong. Several
tests here exist to pin that divergence down rather than paper over it.
"""

import pandas as pd
import pytest

from returns import compute_external_cashflows, compute_twr, compute_xirr


def make_txns(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def series(values: list[float], dates: list[str]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates))


def test_cashflow_sign_convention():
    """Buys are money in (positive), sells are money out (negative), and
    fees always work against the investor in both directions."""
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 10, "price": 100.0, "fees": 1.0},
        {"date": "2026-01-02", "ticker": "X", "action": "SELL",
         "quantity": 5, "price": 110.0, "fees": 1.0},
    ])
    flows = compute_external_cashflows(txns)

    assert flows[pd.Timestamp("2026-01-01")] == pytest.approx(1001.0)
    assert flows[pd.Timestamp("2026-01-02")] == pytest.approx(-549.0)


def test_twr_simple_gain_with_no_later_cashflows():
    """100 -> 110 with no further deposits is unambiguously +10%."""
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    value = series([100, 110], ["2026-01-01", "2026-01-02"])
    assert compute_twr(value, txns) == pytest.approx(0.10)


def test_twr_chains_periods_multiplicatively():
    """Two consecutive +10% periods compound to +21%, not +20%."""
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
        {"date": "2026-07-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 110.0, "fees": 0.0},
    ])
    value = series([100, 220, 242], ["2026-01-01", "2026-07-01", "2027-01-01"])
    assert compute_twr(value, txns) == pytest.approx(0.21)


def test_twr_excludes_deposits_from_return():
    """Doubling the portfolio by depositing more cash is not a 100% return.

    Buy 1 @ 100, then next day deposit another 100 with prices flat. The
    value doubles to 200 but the return is 0%.
    """
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    value = series([100, 200], ["2026-01-01", "2026-01-02"])
    assert compute_twr(value, txns) == pytest.approx(0.0, abs=1e-9)


def test_twr_handles_loss():
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    value = series([100, 75], ["2026-01-01", "2026-01-02"])
    assert compute_twr(value, txns) == pytest.approx(-0.25)


def test_xirr_one_year_ten_percent():
    """The canonical check: 100 in, 110 out one year later -> ~10% annualised."""
    txns = make_txns([
        {"date": "2025-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    result = compute_xirr(txns, current_value=110.0, as_of=pd.Timestamp("2026-01-01"))
    assert result == pytest.approx(0.10, abs=1e-3)


def test_xirr_annualises_a_half_year_gain_upward():
    """+10% earned over six months annualises to roughly +21%, since it
    would compound twice in a year."""
    txns = make_txns([
        {"date": "2025-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    result = compute_xirr(txns, current_value=110.0, as_of=pd.Timestamp("2025-07-02"))
    assert result == pytest.approx(0.21, abs=0.01)


def test_xirr_handles_loss():
    txns = make_txns([
        {"date": "2025-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    result = compute_xirr(txns, current_value=50.0, as_of=pd.Timestamp("2026-01-01"))
    assert result == pytest.approx(-0.50, abs=1e-3)


def test_twr_and_xirr_diverge_with_deposit_timing():
    """The central claim the project makes about these two metrics.

    Scenario: a small position is flat for six months, then a large deposit
    lands right before a sharp rally. Stock selection (TWR) looks modest;
    the investor's actual money-weighted outcome (XIRR) looks much better,
    purely because most of the capital was present for the good part.
    """
    txns = make_txns([
        {"date": "2025-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
        {"date": "2025-07-01", "ticker": "X", "action": "BUY",
         "quantity": 9, "price": 100.0, "fees": 0.0},
    ])
    # Flat first half; +50% second half on a now-10x-larger position.
    value = series([100, 100, 1000, 1500],
                   ["2025-01-01", "2025-06-30", "2025-07-01", "2026-01-01"])

    twr = compute_twr(value, txns)
    xirr = compute_xirr(txns, current_value=1500.0, as_of=pd.Timestamp("2026-01-01"))

    assert twr == pytest.approx(0.50, abs=1e-6)
    assert xirr > twr


def test_xirr_returns_none_for_degenerate_short_window():
    """Annualising a two-day return has no stable solution. Returning None
    is the honest answer; returning a huge number would look like a result."""
    txns = make_txns([
        {"date": "2026-01-01", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
        {"date": "2026-01-02", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 110.0, "fees": 0.0},
    ])
    result = compute_xirr(txns, current_value=242.0, as_of=pd.Timestamp("2026-01-03"))
    assert result is None


def test_twr_skips_days_with_no_prior_position():
    """Before any position exists there is no base to compute a return
    against; those days must be skipped rather than treated as infinite."""
    txns = make_txns([
        {"date": "2026-01-03", "ticker": "X", "action": "BUY",
         "quantity": 1, "price": 100.0, "fees": 0.0},
    ])
    value = series([0, 0, 100, 110],
                   ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
    assert compute_twr(value, txns) == pytest.approx(0.10)
