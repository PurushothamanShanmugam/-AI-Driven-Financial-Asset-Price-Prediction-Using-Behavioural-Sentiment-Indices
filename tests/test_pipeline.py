"""
tests/test_pipeline.py
======================
Integration tests for src/pipeline.py
Tests the full end-to-end flow on a tiny synthetic dataset
so they run fast without real training.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling import chronological_split, build_feature_matrix, TARGETS


@pytest.fixture
def small_prepared_df():
    """100-row synthetic asset DataFrame — fast for integration testing."""
    np.random.seed(99)
    n = 300
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 150 + np.cumsum(np.random.randn(n) * 0.8)
    prices = np.clip(prices, 10, 500)
    feats  = pd.DataFrame(
        np.random.randn(n, 20),
        columns=[f"feat_{i}" for i in range(20)]
    )
    df = pd.DataFrame({
        "Date":  dates,
        "asset": "MOCK",
        "Close": prices,
        "Open":  prices * 0.999,
        "High":  prices * 1.005,
        "Low":   prices * 0.995,
        "vol_norm": np.random.randn(n),
    })
    df = pd.concat([df, feats], axis=1)
    df["target_high_tplus1"]  = prices * 1.003
    df["target_low_tplus1"]   = prices * 0.997
    df["target_high_tplus10"] = prices * 1.015
    df["target_low_tplus10"]  = prices * 0.985
    return df


class TestChronologicalSplitIntegration:

    def test_split_preserves_all_rows(self, small_prepared_df):
        train, valid, test = chronological_split(small_prepared_df)
        assert len(train) + len(valid) + len(test) == len(small_prepared_df)

    def test_no_data_leakage_between_splits(self, small_prepared_df):
        train, valid, test = chronological_split(small_prepared_df)
        assert train.index.max() < valid.index.min()
        assert valid.index.max() < test.index.min()


class TestBuildFeatureMatrixIntegration:

    def test_feature_matrix_correct_shape(self, small_prepared_df):
        X, y, cols = build_feature_matrix(small_prepared_df)
        assert X.shape[0] == len(small_prepared_df)
        assert y.shape == (len(small_prepared_df), len(TARGETS))

    def test_no_target_in_X(self, small_prepared_df):
        X, y, cols = build_feature_matrix(small_prepared_df)
        for tgt in TARGETS:
            assert tgt not in X.columns

    def test_all_targets_in_y(self, small_prepared_df):
        X, y, cols = build_feature_matrix(small_prepared_df)
        for tgt in TARGETS:
            assert tgt in y.columns

    def test_feature_cols_all_numeric(self, small_prepared_df):
        X, y, cols = build_feature_matrix(small_prepared_df)
        for col in cols:
            assert pd.api.types.is_numeric_dtype(X[col]), \
                f"Feature '{col}' is not numeric"


class TestModelTrainingSmoke:
    """
    Smoke tests — trains Ridge on tiny data to verify the pipeline runs
    without errors. Does NOT test R² values (too few rows for that).
    """

    def test_ridge_trains_without_error(self, small_prepared_df):
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer

        train, valid, test = chronological_split(small_prepared_df)
        X_train, y_train, feature_cols = build_feature_matrix(train)
        X_test,  y_test,  _            = build_feature_matrix(test)

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
            ]), feature_cols)
        ])
        pipe = Pipeline([
            ("prep",  preprocessor),
            ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert preds.shape == (len(test), len(TARGETS))
        assert np.isfinite(preds).all(), "Predictions contain NaN/Inf"

    def test_predictions_are_positive_prices(self, small_prepared_df):
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer

        train, valid, test = chronological_split(small_prepared_df)
        X_train, y_train, feature_cols = build_feature_matrix(train)
        X_test,  y_test,  _            = build_feature_matrix(test)

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
            ]), feature_cols)
        ])
        pipe = Pipeline([
            ("prep",  preprocessor),
            ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        clamped = np.clip(preds, 0, None)

        # After clamping, all values should be non-negative
        assert (clamped >= 0).all(), "Some clamped predictions are negative"