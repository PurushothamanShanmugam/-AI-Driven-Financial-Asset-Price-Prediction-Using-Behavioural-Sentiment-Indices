"""
tests/test_data_loader.py
=========================
Unit tests for src/data_loader.py
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_all_assets


class TestLoadAllAssets:

    def test_returns_dict(self):
        result = load_all_assets()
        assert isinstance(result, dict), "load_all_assets() must return a dict"

    def test_all_five_assets_present(self):
        result = load_all_assets()
        expected = {"SPY", "QQQ", "GLD", "TLT", "BTC_USD"}
        assert set(result.keys()) == expected, f"Expected {expected}, got {set(result.keys())}"

    def test_each_asset_is_dataframe(self):
        result = load_all_assets()
        for name, df in result.items():
            assert isinstance(df, pd.DataFrame), f"{name} should be a DataFrame"

    def test_date_column_exists(self):
        result = load_all_assets()
        for name, df in result.items():
            assert "Date" in df.columns, f"{name} missing 'Date' column"

    def test_date_column_is_datetime(self):
        result = load_all_assets()
        for name, df in result.items():
            assert pd.api.types.is_datetime64_any_dtype(df["Date"]), \
                f"{name} 'Date' column should be datetime"

    def test_ohlcv_columns_present(self):
        result = load_all_assets()
        required = {"Open", "High", "Low", "Close", "Volume"}
        for name, df in result.items():
            missing = required - set(df.columns)
            assert not missing, f"{name} missing OHLCV columns: {missing}"

    def test_no_duplicate_dates_per_asset(self):
        result = load_all_assets()
        for name, df in result.items():
            dupes = df["Date"].duplicated().sum()
            assert dupes == 0, f"{name} has {dupes} duplicate dates"

    def test_dates_are_sorted(self):
        result = load_all_assets()
        for name, df in result.items():
            assert df["Date"].is_monotonic_increasing, \
                f"{name} dates are not sorted ascending"

    def test_minimum_row_count(self):
        result = load_all_assets()
        for name, df in result.items():
            assert len(df) >= 1000, \
                f"{name} has only {len(df)} rows — expected at least 1000"

    def test_asset_column_set_correctly(self):
        result = load_all_assets()
        for name, df in result.items():
            assert "asset" in df.columns, f"{name} missing 'asset' column"
            assert df["asset"].iloc[0] == name, \
                f"{name} asset column value mismatch"

    def test_close_prices_positive(self):
        result = load_all_assets()
        for name, df in result.items():
            assert (df["Close"] > 0).all(), \
                f"{name} has non-positive Close prices"

    def test_volume_non_negative(self):
        result = load_all_assets()
        for name, df in result.items():
            assert (df["Volume"] >= 0).all(), \
                f"{name} has negative Volume values"

    def test_btc_has_fear_greed(self):
        result = load_all_assets()
        btc = result["BTC_USD"]
        assert "fg_value" in btc.columns, \
            "BTC_USD should have 'fg_value' (Fear & Greed) column"

    def test_spy_has_vix(self):
        result = load_all_assets()
        spy = result["SPY"]
        vix_cols = [c for c in spy.columns if "vix" in c.lower()]
        assert vix_cols, "SPY should have VIX columns"