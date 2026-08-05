"""
Fetch and cache daily closing prices for a set of tickers.

Prices are cached to a local parquet file so re-running the pipeline doesn't
re-download from Yahoo Finance every time -- useful both for speed and
because yfinance will rate-limit or flake out under repeated hammering.

Usage:
    from prices import get_prices

    prices = get_prices(["AAPL", "MSFT"], start="2025-11-01", end="2026-08-05")
    # prices is a DataFrame indexed by date, one column per ticker
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

CACHE_PATH = Path("data/price_cache.csv")
CACHE_MAX_AGE = timedelta(days=1)


def get_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_path: Path = CACHE_PATH,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily closing prices for `tickers` between `start` and `end`.

    Reads from the parquet cache if it's fresh and covers the requested
    tickers and date range; otherwise re-downloads via yfinance and
    rewrites the cache. Set force_refresh=True to always re-download.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cached = None if force_refresh else _load_cache(cache_path)

    if cached is not None and _cache_covers(cached, tickers, start, end):
        return cached.loc[start:end, tickers]

    downloaded = _download(tickers, start, end)

    if cached is not None:
        # Merge rather than overwrite, so tickers/dates fetched in earlier
        # runs aren't lost just because this run asked for a narrower set.
        combined = downloaded.combine_first(cached)
    else:
        combined = downloaded

    _save_cache(combined, cache_path)
    return combined.loc[start:end, tickers]


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance returns a single-level column index only when given one ticker;
    # requesting as a list keeps the shape consistent regardless of count.
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        # Single ticker: raw.columns is a flat Index like ["Open", "High", ...]
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    closes.index = pd.to_datetime(closes.index)
    return closes


def _load_cache(cache_path: Path) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    if age > CACHE_MAX_AGE:
        return None
    try:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    except Exception:
        return None


def _cache_covers(cached: pd.DataFrame, tickers: list[str], start: str, end: str) -> bool:
    if not set(tickers).issubset(cached.columns):
        return False
    if cached.index.min() > pd.Timestamp(start):
        return False
    if cached.index.max() < pd.Timestamp(end) - pd.Timedelta(days=3):
        # Allow a few days of slack for weekends/holidays with no trading.
        return False
    return True


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)


if __name__ == "__main__":
    import sys

    tickers = sys.argv[1:] or ["AAPL", "MSFT"]
    result = get_prices(tickers, start="2025-11-01")
    print(result.tail())
