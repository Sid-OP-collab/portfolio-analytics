"""
Risk metrics computed from a daily portfolio value series.

IMPORTANT: these must be computed on a *cash-flow-adjusted* return series,
not on raw portfolio value. Raw value jumps whenever money is deposited or
withdrawn, and a naive pct_change() reads a deposit as an enormous one-day
gain. For a portfolio funded by regular contributions that inflates
volatility, distorts Sharpe, and masks real drawdowns -- it measures the
deposit schedule rather than market risk.

The fix is `index_series`: a synthetic unit price (like a fund's NAV per
share) that starts at 100 and moves only when markets move. Pass
`transactions` to any function here to get the adjusted figure; omit it
only when the series is already known to be free of external cash flows.

Usage:
    from risk import annualised_volatility, sharpe_ratio, max_drawdown

    vol = annualised_volatility(portfolio["total_value"], transactions)
    dd, peak, trough = max_drawdown(portfolio["total_value"], transactions)
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _external_cashflows(transactions: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Net cash added to the portfolio per day, aligned to `index`.

    Positive means the investor put money in (a BUY); negative means they
    took money out (a SELL). Mirrors returns.compute_external_cashflows;
    duplicated here so risk.py doesn't depend on returns.py.
    """
    trades = transactions[transactions["action"].isin(["BUY", "SELL"])]
    if trades.empty:
        return pd.Series(0.0, index=index)

    signed = trades.apply(
        lambda r: (r["quantity"] * r["price"] + r["fees"])
        if r["action"] == "BUY"
        else -(r["quantity"] * r["price"] - r["fees"]),
        axis=1,
    )
    per_day = signed.groupby(trades["date"]).sum()
    return per_day.reindex(index, fill_value=0.0)


def daily_returns(
    value: pd.Series, transactions: pd.DataFrame | None = None
) -> pd.Series:
    """Day-over-day return, excluding the effect of deposits and withdrawals.

    With `transactions`, each day's return is (change in value minus net
    cash added) / previous value -- so contributing money is correctly
    treated as a zero-return event rather than a gain. Without it, this
    falls back to a plain pct_change(), which is only valid when no
    external cash flows occurred.
    """
    if transactions is None:
        return value.pct_change().dropna()

    flows = _external_cashflows(transactions, value.index)
    prev = value.shift(1)
    adjusted = (value - prev - flows) / prev

    # Days before any position existed have no base to measure against.
    adjusted = adjusted[prev > 0]
    return adjusted.dropna()


def index_series(
    value: pd.Series, transactions: pd.DataFrame | None = None, base: float = 100.0
) -> pd.Series:
    """A synthetic unit-price series: what one 'share' of this portfolio
    would be worth if it started at `base` and no money were ever added
    or removed. Drawdowns measured on this reflect market losses rather
    than the shape of the contribution schedule.
    """
    returns = daily_returns(value, transactions)
    compounded = base * (1 + returns).cumprod()

    # Prepend the opening level. Without it the series starts at its first
    # *post-return* value, so a portfolio that only ever fell would show no
    # drawdown at all -- there'd be no peak above it to measure from.
    if len(returns) == 0:
        return pd.Series([base], index=value.index[:1])

    first_date = value.index[value.index.get_loc(returns.index[0]) - 1]
    opening = pd.Series([base], index=[first_date])
    return pd.concat([opening, compounded])


def annualised_volatility(
    value: pd.Series, transactions: pd.DataFrame | None = None
) -> float:
    """Standard deviation of daily returns, scaled to an annual figure by
    the sqrt(time) rule -- this assumes returns are roughly independent
    day to day, which is a standard simplification, not a guarantee.

    Pass `transactions` so deposits aren't counted as volatility.
    """
    returns = daily_returns(value, transactions)
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(
    value: pd.Series,
    transactions: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio: excess return over the risk-free rate,
    divided by annualised volatility.

    risk_free_rate: annualised, e.g. 0.045 for 4.5%. Defaults to 0 -- in
    a near-zero-rate environment this barely matters, but it stops being
    a safe default once rates are meaningfully above zero, so it's worth
    passing the current rate explicitly rather than relying on the default.
    """
    returns = daily_returns(value, transactions)
    vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if vol < 1e-10:
        return float("nan")  # undefined, not zero -- a flat series has no ratio
    annualised_return = returns.mean() * TRADING_DAYS_PER_YEAR
    return (annualised_return - risk_free_rate) / vol


def max_drawdown(
    value: pd.Series, transactions: pd.DataFrame | None = None
) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Largest peak-to-trough decline.

    Returns (drawdown_pct, peak_date, trough_date). drawdown_pct is
    negative (e.g. -0.23 for a 23% decline). If the series never declines,
    returns (0.0, None, None).

    Pass `transactions` so that a deposit isn't mistaken for a recovery:
    on raw value, adding cash lifts the line and can erase a drawdown that
    the investor genuinely experienced.
    """
    if transactions is not None:
        value = index_series(value, transactions)

    running_peak = value.cummax()
    drawdown = value / running_peak - 1.0

    trough_date = drawdown.idxmin()
    trough_value = drawdown.loc[trough_date]

    if trough_value >= 0:
        return 0.0, None, None

    # The peak is the most recent all-time-high date at or before the trough.
    peak_date = value.loc[:trough_date].idxmax()
    return trough_value, peak_date, trough_date


def drawdown_duration(
    value: pd.Series, transactions: pd.DataFrame | None = None
) -> int:
    """Length, in calendar days, of the longest stretch spent below a prior
    peak (i.e. time from a peak until the value recovers to it).
    Returns 0 if the series is non-decreasing throughout. If the series
    ends still in a drawdown, that unfinished stretch counts too -- it's
    real underwater time even without a recovery date yet.

    Pass `transactions` for the same reason as max_drawdown.
    """
    if transactions is not None:
        value = index_series(value, transactions)

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
    portfolio_value: pd.Series,
    benchmark_value: pd.Series,
    window: int = 60,
    transactions: pd.DataFrame | None = None,
) -> pd.Series:
    """Rolling beta of the portfolio against a benchmark over `window` days.

    Both series must share a comparable date index; they're aligned by
    inner join before computing, so mismatched date ranges don't silently
    produce NaNs everywhere. The benchmark is a price series with no cash
    flows, so only the portfolio side needs adjusting.
    """
    port_returns = daily_returns(portfolio_value, transactions)
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
    if contrib.empty:
        # No position had usable price data. Return the expected shape rather
        # than an empty frame with no columns, so callers can filter, sort,
        # or display the result without special-casing this.
        return pd.DataFrame(
            columns=[
                "ticker",
                "start_value",
                "end_value",
                "position_return",
                "weight",
                "contribution",
            ]
        )

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

    # Passing `txns` is what makes these figures measure market risk rather
    # than the shape of the contribution schedule.
    vol = annualised_volatility(portfolio["total_value"], txns)
    sharpe = sharpe_ratio(portfolio["total_value"], txns)
    dd, peak, trough = max_drawdown(portfolio["total_value"], txns)
    dd_days = drawdown_duration(portfolio["total_value"], txns)

    print(f"Annualised volatility: {vol:.2%}")
    print(f"Sharpe ratio:          {sharpe:.2f}")
    print(f"Max drawdown:          {dd:.2%} (peak {peak}, trough {trough})")
    print(f"Longest drawdown:      {dd_days} days")

    print("\nPer-position contribution to return:")
    contrib = position_contribution(daily_holdings, price_data)
    print(contrib[["ticker", "weight", "position_return", "contribution"]])
