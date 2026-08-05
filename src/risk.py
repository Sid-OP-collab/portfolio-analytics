"""
Risk metrics computed from a daily portfolio value series.

All of these are computed on daily returns derived from `total_value`,
NOT time-weighted returns -- for risk metrics (volatility, drawdown) you
want the return series as actually experienced day to day, including the
effect of cash flows, because that's the risk you actually lived through.
This is a deliberate difference from returns.py's TWR, which exists
specifically to strip cash flow timing out. Mixing the two purposes into
one return series would misrepresent both.

Usage:
    from risk import (
        annualised_volatility, sharpe_ratio, max_drawdown,
        rolling_beta, position_contribution,
    )
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def daily_returns(value: pd.Series) -> pd.Series:
    """Simple day-over-day percentage change. First entry is dropped (no
    prior value to compare against)."""
    return value.pct_change().dropna()


def annualised_volatility(value: pd.Series) -> float:
    """Standard deviation of daily returns, scaled to an annual figure by
    the sqrt(time) rule -- this assumes returns are roughly independent
    day to day, which is a standard simplification, not a guarantee."""
    returns = daily_returns(value)
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(value: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio: excess return over the risk-free rate,
    divided by annualised volatility.

    risk_free_rate: annualised, e.g. 0.045 for 4.5%. Defaults to 0 -- in
    a near-zero-rate environment this barely matters, but it stops being
    a safe default once rates are meaningfully above zero, so it's worth
    passing the current rate explicitly rather than relying on the default.
    """
    returns = daily_returns(value)
    vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if vol < 1e-10:
        return float("nan")  # undefined, not zero -- a flat series has no ratio
    annualised_return = returns.mean() * TRADING_DAYS_PER_YEAR
    return (annualised_return - risk_free_rate) / vol


def max_drawdown(value: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Largest peak-to-trough decline.

    Returns (drawdown_pct, peak_date, trough_date). drawdown_pct is
    negative (e.g. -0.23 for a 23% decline). If the series never declines,
    returns (0.0, None, None).
    """
    running_peak = value.cummax()
    drawdown = value / running_peak - 1.0

    trough_date = drawdown.idxmin()
    trough_value = drawdown.loc[trough_date]

    if trough_value >= 0:
        return 0.0, None, None

    # The peak is the most recent all-time-high date at or before the trough.
    peak_date = value.loc[:trough_date].idxmax()
    return trough_value, peak_date, trough_date


def drawdown_duration(value: pd.Series) -> int:
    """Length, in calendar days, of the longest stretch spent below a prior
    peak (i.e. time from a peak until the value fully recovers past it).
    Returns 0 if the series is non-decreasing throughout. If the series
    ends still in a drawdown, that unfinished stretch counts too -- it's
    real underwater time even without a recovery date yet.
    """
    running_peak = value.cummax()
    underwater = value < running_peak

    longest = 0
    current_start = None
    for date, is_underwater in underwater.items():
        if is_underwater and current_start is None:
            current_start = date
        elif not is_underwater and current_start is not None:
            longest = max(longest, (date - current_start).days)
            current_start = None
    if current_start is not None:  # still underwater at the end of the series
        longest = max(longest, (value.index[-1] - current_start).days)

    return longest


def rolling_beta(
    portfolio_value: pd.Series, benchmark_value: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling beta of the portfolio against a benchmark over `window` days.

    Both series must share a comparable date index; they're aligned by
    inner join before computing, so mismatched date ranges don't silently
    produce NaNs everywhere.
    """
    port_returns = daily_returns(portfolio_value)
    bench_returns = daily_returns(benchmark_value)
    aligned = pd.concat([port_returns, bench_returns], axis=1, join="inner")
    aligned.columns = ["portfolio", "benchmark"]

    rolling_cov = aligned["portfolio"].rolling(window).cov(aligned["benchmark"])
    rolling_var = aligned["benchmark"].rolling(window).var()
    return rolling_cov / rolling_var


def position_contribution(
    daily_holdings: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Each position's contribution to total return over the full period:
    (start-of-period weight) x (that position's own return), summed across
    positions equals (approximately) the portfolio's total return.

    This is a standard single-period Brinson-style attribution -- it
    approximates because weights actually drift daily as prices move and
    trades happen, but for a summary "what drove the return" table this
    approximation is the industry-standard starting point.
    """
    first_day = daily_holdings["date"].min()
    last_day = daily_holdings["date"].max()

    # Forward/back-fill prices across the full holdings date range so a
    # lookup for "today" still resolves to the most recent close, rather
    # than requiring an exact date match (which fails whenever today's
    # close isn't published yet -- silently dropping every still-open
    # position from this table while closed ones, whose end date is safely
    # in the past, would appear fine).
    full_range = pd.date_range(min(first_day, prices.index.min()), max(last_day, prices.index.max()), freq="D")
    prices_filled = prices.reindex(full_range).ffill().bfill()

    rows = []
    for ticker, group in daily_holdings.groupby("ticker"):
        if ticker not in prices_filled.columns:
            continue

        first_qty_rows = group[group["date"] == group["date"].min()]
        start_date = first_qty_rows["date"].iloc[0]
        start_qty = first_qty_rows["quantity"].iloc[0]

        last_qty_rows = group[group["date"] == group["date"].max()]
        end_date = last_qty_rows["date"].iloc[0]
        end_qty = last_qty_rows["quantity"].iloc[0]

        start_price = prices_filled.loc[start_date, ticker]
        end_price = prices_filled.loc[end_date, ticker]

        if pd.isna(start_price) or pd.isna(end_price):
            continue  # genuinely no price data anywhere near this ticker's range

        start_value = start_qty * start_price
        end_value = end_qty * end_price
        position_return = (end_price / start_price) - 1.0

        rows.append(
            {
                "ticker": ticker,
                "start_value": start_value,
                "end_value": end_value,
                "position_return": position_return,
            }
        )

    contrib = pd.DataFrame(rows)
    total_start_value = contrib["start_value"].sum()
    contrib["weight"] = contrib["start_value"] / total_start_value
    contrib["contribution"] = contrib["weight"] * contrib["position_return"]
    return contrib.sort_values("contribution", ascending=False).reset_index(drop=True)


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

    vol = annualised_volatility(portfolio["total_value"])
    sharpe = sharpe_ratio(portfolio["total_value"])
    dd, peak, trough = max_drawdown(portfolio["total_value"])
    dd_days = drawdown_duration(portfolio["total_value"])

    print(f"Annualised volatility: {vol:.2%}")
    print(f"Sharpe ratio:          {sharpe:.2f}")
    print(f"Max drawdown:          {dd:.2%} (peak {peak}, trough {trough})")
    print(f"Longest drawdown:      {dd_days} days")

    print("\nPer-position contribution to return:")
    contrib = position_contribution(daily_holdings, price_data)
    print(contrib[["ticker", "weight", "position_return", "contribution"]])
