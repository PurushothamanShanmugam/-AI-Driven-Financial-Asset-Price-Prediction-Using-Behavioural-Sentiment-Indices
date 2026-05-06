"""
tests/test_preprocess.py
========================
Unit tests for src/preprocess.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_all_assets
from src.preprocess  import prepare_asset_dataframe

TARGETS = [
    "target_high_tplus1",
    "target_low_tplus1",
    "target_high_tplus10",
    "target_low_tplus10",
]


@pytest.fixture(scope="module")
def prepared_spy():
    raw = load_all_assets()
    return prepare_asset_dataframe(raw["SPY"], "SPY")


@pytest.fixture(scope="module")
def prepared_btc():
    raw = load_all_assets()
    return prepare_asset_dataframe(raw["BTC_USD"], "BTC_USD")


class TestPrepareAssetDataFrame:

    def test_returns_dataframe(self, prepared_spy):
        assert isinstance(prepared_spy, pd.DataFrame)

    def test_all_targets_present(self, prepared_spy):
        for tgt in TARGETS:
            assert tgt in prepared_spy.columns, f"Missing target: {tgt}"

    def test_no_string_columns_in_features(self, prepared_spy):
        exclude = {"Date", "asset"}
        for col in prepared_spy.columns:
            if col not in exclude and col not in TARGETS:
                assert pd.api.types.is_numeric_dtype(prepared_spy[col]), \
                    f"Feature column '{col}' is not numeric"

    def test_date_column_present(self, prepared_spy):
        assert "Date" in prepared_spy.columns

    def test_minimum_feature_count(self, prepared_spy):
        feature_cols = [c for c in prepared_spy.columns
                        if c not in TARGETS and c not in {"Date", "asset"}
                        and pd.api.types.is_numeric_dtype(prepared_spy[c])]
        assert len(feature_cols) >= 50, \
            f"Expected ≥50 features, got {len(feature_cols)}"

    def test_no_all_nan_feature_columns(self, prepared_spy):
        feature_cols = [c for c in prepared_spy.columns
                        if c not in TARGETS and c not in {"Date", "asset"}
                        and pd.api.types.is_numeric_dtype(prepared_spy[c])]
        for col in feature_cols:
            nan_pct = prepared_spy[col].isna().mean()
            assert nan_pct < 0.95, \
                f"Column '{col}' is {nan_pct:.0%} NaN — likely broken feature"

    def test_target_values_positive(self, prepared_spy):
        for tgt in TARGETS:
            col = prepared_spy[tgt].dropna()
            assert (col > 0).all(), f"Target '{tgt}' has non-positive values"

    def test_vol_norm_column_present(self, prepared_spy):
        assert "vol_norm" in prepared_spy.columns, \
            "vol_norm column required for volatility-based prediction intervals"

    def test_shift_guard_no_leakage(self, prepared_spy):
        """
        Verify leakage fix: lag_1 features must be the previous day's value,
        not the same day. We check that lag_1 col = Close.shift(1).
        """
        if "Close_lag_1" in prepared_spy.columns:
            expected = prepared_spy["Close"].shift(1)
            actual   = prepared_spy["Close_lag_1"]
            # Allow for NaN in first row
            match = expected.iloc[1:].round(4).equals(actual.iloc[1:].round(4))
            assert match, "Close_lag_1 does not match Close.shift(1) — potential leakage"

    def test_targets_created_from_future_prices(self, prepared_spy):
        """
        target_high_tplus1 should equal next day's High (forward-looking).
        """
        if "High" in prepared_spy.columns:
            expected = prepared_spy["High"].shift(-1)
            actual   = prepared_spy["target_high_tplus1"]
            # Compare middle rows (avoid NaN edges)
            mid = slice(10, -10)
            assert np.allclose(
                expected.iloc[mid].dropna().values,
                actual.iloc[mid].dropna().values,
                rtol=1e-3
            ), "target_high_tplus1 does not match High.shift(-1)"

    def test_btc_fg_classification_dropped(self, prepared_btc):
        """
        fg_classification is a string column — must be dropped or
        one-hot encoded before reaching the feature matrix.
        """
        feature_cols = [c for c in prepared_btc.columns
                        if c not in TARGETS and c not in {"Date", "asset"}]
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(prepared_btc[col]), \
                f"Non-numeric column '{col}' found — fg_classification not cleaned"

    def test_row_count_preserved(self, prepared_spy):
        raw = load_all_assets()
        original_len = len(raw["SPY"])
        # After dropping NaN targets, row count should be close to original
        assert len(prepared_spy) >= original_len * 0.85, \
            f"Too many rows dropped: {original_len} → {len(prepared_spy)}"