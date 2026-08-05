"""End-to-end test of the full pipeline.

Unit tests cover each module in isolation; this one checks that they still
compose correctly -- that the shape ledger emits is the shape valuation
expects, and so on. Prices are stubbed rather than downloaded so the suite
stays deterministic and runs without network access.
"""

import pandas as pd
import pytest

from ledger import build_ledger
from loader import load_transactions
from returns import compute_twr
from risk import annualised_volatility, max_drawdown, position_contribution
from valuation import value_portfolio

# Transaction prices deliberately match the stubbed market prices below.
# If they didn't, selling above the prevailing market price would register
# as a genuine gain in TWR -- correct behaviour, but it would make this
# fixture a test of something other than "flat market, zero return".
CSV = (
    "date,ticker,action,quantity,price,fees\n"
    "2026-01-01,AAA,BUY,10,100.00,0\n"
    "2026-01-02,BBB,BUY,5,200.00,0\n"
    "2026-01-05,AAA,SELL,4,100.00,0\n"
    "2026-01-06,AAA,DIV,0,3.50,0\n"
)


@pytest.fixture
def pipeline(tmp_path):
    """Run the whole pipeline against a fixed, hand-checkable dataset."""
    path = tmp_path / "txns.csv"
    path.write_text(CSV)

    txns = load_transactions(path)
    _, daily = build_ledger(txns, as_of=pd.Timestamp("2026-01-08"))

    dates = pd.date_range("2026-01-01", "2026-01-08", freq="D")
    prices = pd.DataFrame(
        {"AAA": [100.0] * len(dates), "BBB": [200.0] * len(dates)},
        index=dates,
    )

    portfolio = value_portfolio(daily, prices)
    return txns, daily, prices, portfolio


def test_pipeline_runs_end_to_end(pipeline):
    _, _, _, portfolio = pipeline
    assert not portfolio.empty
    assert portfolio["total_value"].notna().all()


def test_final_holdings_match_hand_calculation(pipeline):
    """Bought 10 AAA and 5 BBB, sold 4 AAA. Final: 6 AAA, 5 BBB.
    Cost basis: 6*100 + 5*200 = 1600. At flat prices, value is also 1600."""
    _, daily, _, portfolio = pipeline

    final = daily[daily["date"] == pd.Timestamp("2026-01-08")].set_index("ticker")
    assert final.loc["AAA", "quantity"] == pytest.approx(6)
    assert final.loc["BBB", "quantity"] == pytest.approx(5)

    last = portfolio.loc[pd.Timestamp("2026-01-08")]
    assert last["total_cost_basis"] == pytest.approx(1600.0)
    assert last["total_value"] == pytest.approx(1600.0)
    assert last["total_unrealised_pnl"] == pytest.approx(0.0)


def test_flat_prices_produce_zero_return_and_volatility(pipeline):
    """With prices constant throughout, every price-driven metric must be
    zero. Anything non-zero here means cash flows are leaking into the
    return calculation, which is exactly what TWR exists to prevent."""
    txns, _, _, portfolio = pipeline

    assert compute_twr(portfolio["total_value"], txns) == pytest.approx(0.0, abs=1e-9)
    assert annualised_volatility(portfolio["total_value"], txns) == pytest.approx(
        0.0, abs=1e-9
    )

    dd, _, _ = max_drawdown(portfolio["total_value"], txns)
    assert dd == pytest.approx(0.0, abs=1e-9)


def test_attribution_covers_every_held_ticker(pipeline):
    _, daily, prices, _ = pipeline
    contrib = position_contribution(daily, prices)
    assert set(contrib["ticker"]) == {"AAA", "BBB"}


def test_dividend_row_does_not_inflate_holdings(pipeline):
    """The DIV row carries 3.50 of cash. If it were treated as shares, AAA
    would show 9.5 rather than 6."""
    _, daily, _, _ = pipeline
    final = daily[daily["date"] == pd.Timestamp("2026-01-08")].set_index("ticker")
    assert final.loc["AAA", "quantity"] == pytest.approx(6)
