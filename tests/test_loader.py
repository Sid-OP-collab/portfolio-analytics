"""Tests for loader.load_transactions.

The loader is the project's only guard against bad input reaching the
FIFO engine, so these tests focus on the failure cases rather than the
happy path -- a loader that accepts a malformed file silently is worse
than one that crashes.
"""

import pandas as pd
import pytest

from loader import TransactionValidationError, load_transactions

HEADER = "date,ticker,action,quantity,price,fees\n"


def write_csv(tmp_path, body: str, name: str = "txns.csv"):
    path = tmp_path / name
    path.write_text(HEADER + body)
    return path


def test_loads_valid_file(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-01,AAPL,BUY,10,100.00,1.00\n"
        "2026-01-05,AAPL,SELL,4,110.00,1.00\n",
    )
    df = load_transactions(path)

    assert len(df) == 2
    assert list(df.columns[:6]) == ["date", "ticker", "action", "quantity", "price", "fees"]
    assert df["date"].dtype.kind == "M"  # parsed to datetime, not left as string


def test_sorts_by_date(tmp_path):
    """Rows may arrive newest-first (as broker exports often do); the FIFO
    engine depends on chronological order, so the loader must not trust
    the file's ordering."""
    path = write_csv(
        tmp_path,
        "2026-03-01,AAPL,BUY,1,100.00,0\n"
        "2026-01-01,AAPL,BUY,1,90.00,0\n"
        "2026-02-01,AAPL,BUY,1,95.00,0\n",
    )
    df = load_transactions(path)
    assert list(df["date"]) == sorted(df["date"])


def test_normalises_ticker_and_action_case(tmp_path):
    path = write_csv(tmp_path, "2026-01-01,aapl,buy,10,100.00,0\n")
    df = load_transactions(path)
    assert df.loc[0, "ticker"] == "AAPL"
    assert df.loc[0, "action"] == "BUY"


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,ticker,action,quantity\n2026-01-01,AAPL,BUY,10\n")
    with pytest.raises(TransactionValidationError, match="Missing required column"):
        load_transactions(path)


def test_unknown_action_raises(tmp_path):
    path = write_csv(tmp_path, "2026-01-01,AAPL,TRANSFER,10,100.00,0\n")
    with pytest.raises(TransactionValidationError, match="unknown action"):
        load_transactions(path)


def test_negative_quantity_raises(tmp_path):
    path = write_csv(tmp_path, "2026-01-01,AAPL,BUY,-5,100.00,0\n")
    with pytest.raises(TransactionValidationError, match="quantity > 0"):
        load_transactions(path)


def test_overselling_raises(tmp_path):
    """Selling more than you hold is the error most likely to silently
    corrupt cost-basis maths downstream, so it must be caught at load."""
    path = write_csv(
        tmp_path,
        "2026-01-01,AAPL,BUY,5,100.00,0\n"
        "2026-01-02,AAPL,SELL,10,110.00,0\n",
    )
    with pytest.raises(TransactionValidationError, match="exceeds holding"):
        load_transactions(path)


def test_oversell_check_is_per_ticker(tmp_path):
    """Holding 10 AAPL must not license selling 10 MSFT."""
    path = write_csv(
        tmp_path,
        "2026-01-01,AAPL,BUY,10,100.00,0\n"
        "2026-01-02,MSFT,SELL,10,110.00,0\n",
    )
    with pytest.raises(TransactionValidationError, match="MSFT"):
        load_transactions(path)


def test_selling_entire_position_is_allowed(tmp_path):
    """Boundary case: selling exactly what you hold is valid, and must not
    be tripped up by floating-point comparison on fractional shares."""
    path = write_csv(
        tmp_path,
        "2026-01-01,AAPL,BUY,1.371721,100.00,0\n"
        "2026-01-02,AAPL,SELL,1.371721,110.00,0\n",
    )
    df = load_transactions(path)
    assert len(df) == 2


def test_reports_all_errors_not_just_first(tmp_path):
    path = write_csv(
        tmp_path,
        "2026-01-01,AAPL,TRANSFER,10,100.00,0\n"
        "2026-01-02,MSFT,BUY,-5,100.00,0\n",
    )
    with pytest.raises(TransactionValidationError) as exc:
        load_transactions(path)
    message = str(exc.value)
    assert "TRANSFER" in message
    assert "quantity > 0" in message


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "does_not_exist.csv")
