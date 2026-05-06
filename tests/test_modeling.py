"""
tests/test_modeling.py
======================
Unit tests for src/modeling.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling import (
    chronological_split,
    build_feature_matrix,
    evaluate_predictions,
    compute_prediction_intervals,
    model_spaces,
    TARGETS,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_df():
    """Minimal DataFrame that mimics a prepared asset DataFrame."""
    np.random.seed(42)
    n = 200
    dates  = pd.date_range("2015-01-01", periods=n, freq="B")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Date":   dates,
        "asset":  "TEST",
        "Close":  prices,
        "Open":   prices * 0.999,
        "High":   prices * 1.005,
        "Low":    prices * 0.995,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        "vol_norm": np.random.randn(n),
        "feat_1":   np.random.randn(n),
        "feat_2":   np.random.randn(n),
        "feat_3":   np.random.randn(n),
        "target_high_tplus1":  prices * 1.002,
        "target_low_tplus1":   prices * 0.998,
        "target_high_tplus10": prices * 1.010,
        "target_low_tplus10":  prices * 0.990,
    })
    return df


@pytest.fixture
def split_dfs(dummy_df):
    train, valid, test = chronological_split(dummy_df)
    return train, valid, test


# ── chronological_split ───────────────────────────────────────────────────────

class TestChronologicalSplit:

    def test_returns_three_splits(self, dummy_df):
        result = chronological_split(dummy_df)
        assert len(result) == 3

    def test_sizes_sum_to_total(self, dummy_df):
        train, valid, test = chronological_split(dummy_df)
        assert len(train) + len(valid) + len(test) == len(dummy_df)

    def test_default_70_15_15(self, dummy_df):
        n = len(dummy_df)
        train, valid, test = chronological_split(dummy_df)
        assert abs(len(train) / n - 0.70) < 0.02
        assert abs(len(valid) / n - 0.15) < 0.02

    def test_no_overlap(self, dummy_df):
        train, valid, test = chronological_split(dummy_df)
        train_idx = set(train.index)
        valid_idx = set(valid.index)
        test_idx  = set(test.index)
        assert train_idx.isdisjoint(valid_idx)
        assert train_idx.isdisjoint(test_idx)
        assert valid_idx.isdisjoint(test_idx)

    def test_chronological_order(self, dummy_df):
        train, valid, test = chronological_split(dummy_df)
        assert train["Date"].max() < valid["Date"].min()
        assert valid["Date"].max() < test["Date"].min()

    def test_custom_split_ratio(self, dummy_df):
        train, valid, test = chronological_split(dummy_df, train=0.6, valid=0.2)
        n = len(dummy_df)
        assert abs(len(train) / n - 0.60) < 0.02


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:

    def test_returns_three_values(self, dummy_df):
        result = build_feature_matrix(dummy_df)
        assert len(result) == 3

    def test_targets_excluded_from_features(self, dummy_df):
        X, y, cols = build_feature_matrix(dummy_df)
        for tgt in TARGETS:
            assert tgt not in cols, f"Target '{tgt}' should not be in feature columns"

    def test_date_excluded(self, dummy_df):
        X, y, cols = build_feature_matrix(dummy_df)
        assert "Date" not in cols

    def test_asset_excluded(self, dummy_df):
        X, y, cols = build_feature_matrix(dummy_df)
        assert "asset" not in cols

    def test_y_has_all_targets(self, dummy_df):
        X, y, cols = build_feature_matrix(dummy_df)
        for tgt in TARGETS:
            assert tgt in y.columns

    def test_X_shape_matches_df(self, dummy_df):
        X, y, cols = build_feature_matrix(dummy_df)
        assert len(X) == len(dummy_df)

    def test_only_numeric_features(self, dummy_df):
        df_with_str = dummy_df.copy()
        df_with_str["string_col"] = "text"
        X, y, cols = build_feature_matrix(df_with_str)
        assert "string_col" not in cols


# ── evaluate_predictions ─────────────────────────────────────────────────────

class TestEvaluatePredictions:

    def test_returns_list_of_dicts(self, dummy_df):
        _, y, _ = build_feature_matrix(dummy_df)
        y_pred  = y.values * 1.01
        rows    = evaluate_predictions(y, y_pred, "TestModel", "test", "TEST")
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_one_row_per_target(self, dummy_df):
        _, y, _ = build_feature_matrix(dummy_df)
        y_pred  = y.values * 1.01
        rows    = evaluate_predictions(y, y_pred, "TestModel", "test", "TEST")
        assert len(rows) == len(TARGETS)

    def test_r2_close_to_one_for_near_perfect_preds(self, dummy_df):
        _, y, _ = build_feature_matrix(dummy_df)
        y_pred  = y.values * 1.0001   # nearly perfect
        rows    = evaluate_predictions(y, y_pred, "TestModel", "test", "TEST")
        for row in rows:
            assert row["R2"] > 0.99, f"Expected R²≈1 for near-perfect preds, got {row['R2']}"

    def test_r2_low_for_bad_preds(self, dummy_df):
        _, y, _ = build_feature_matrix(dummy_df)
        np.random.seed(0)
        y_pred  = np.random.randn(*y.shape) * 1000  # terrible predictions
        rows    = evaluate_predictions(y, y_pred, "TestModel", "test", "TEST")
        for row in rows:
            assert row["R2"] < 0.5

    def test_required_keys_present(self, dummy_df):
        _, y, _ = build_feature_matrix(dummy_df)
        y_pred  = y.values
        rows    = evaluate_predictions(y, y_pred, "TestModel", "test", "TEST")
        required_keys = {"asset", "split", "model", "target", "R2", "MAE", "RMSE"}
        for row in rows:
            assert required_keys.issubset(row.keys())

    def test_mae_non_negative(self, dummy_df):
        _, y, _ = build_feature_matrix(dummy_df)
        y_pred  = y.values * 1.05
        rows    = evaluate_predictions(y, y_pred, "TestModel", "test", "TEST")
        for row in rows:
            assert row["MAE"] >= 0

    def test_handles_nan_rows(self, dummy_df):
        _, y, _ = build_feature_matrix(dummy_df)
        y_pred  = y.values.copy().astype(float)
        y_pred[0, :] = np.nan   # introduce NaN
        # Should not raise
        rows = evaluate_predictions(y, y_pred, "TestModel", "test", "TEST")
        assert len(rows) > 0


# ── compute_prediction_intervals ─────────────────────────────────────────────

class TestComputePredictionIntervals:

    def test_returns_two_arrays(self, dummy_df):
        y_pred = np.ones((len(dummy_df), 4)) * 100
        lower, upper = compute_prediction_intervals(dummy_df, y_pred)
        assert lower.shape == y_pred.shape
        assert upper.shape == y_pred.shape

    def test_lower_less_than_upper(self, dummy_df):
        y_pred = np.ones((len(dummy_df), 4)) * 100
        lower, upper = compute_prediction_intervals(dummy_df, y_pred)
        assert (lower <= upper).all()

    def test_intervals_widen_with_high_vol(self, dummy_df):
        y_pred  = np.ones((len(dummy_df), 4)) * 100
        low_vol = dummy_df.copy(); low_vol["vol_norm"]  = -1.0
        high_vol = dummy_df.copy(); high_vol["vol_norm"] =  3.0
        _, upper_low  = compute_prediction_intervals(low_vol,  y_pred)
        _, upper_high = compute_prediction_intervals(high_vol, y_pred)
        assert (upper_high > upper_low).all(), \
            "Intervals should widen with higher volatility"

    def test_base_interval_approximately_1_5pct(self, dummy_df):
        df_zero_vol = dummy_df.copy(); df_zero_vol["vol_norm"] = 0.0
        y_pred = np.ones((len(dummy_df), 4)) * 100
        lower, upper = compute_prediction_intervals(df_zero_vol, y_pred)
        # At vol_norm=0, scale=0.015, so upper ≈ 101.5, lower ≈ 98.5
        assert np.allclose(upper, 101.5, atol=0.01)
        assert np.allclose(lower, 98.5,  atol=0.01)


# ── model_spaces ─────────────────────────────────────────────────────────────

class TestModelSpaces:

    def test_returns_dict(self):
        spaces = model_spaces()
        assert isinstance(spaces, dict)

    def test_expected_models_present(self):
        spaces = model_spaces()
        expected = {"Ridge", "Lasso", "ElasticNet"}
        assert set(spaces.keys()) == expected

    def test_rf_and_gb_not_present(self):
        """RF and GB deliberately excluded due to negative R² on financial data."""
        spaces = model_spaces()
        assert "RandomForest"     not in spaces
        assert "GradientBoosting" not in spaces

    def test_each_model_has_pipe_and_grid(self):
        spaces = model_spaces()
        for name, (pipe, grid) in spaces.items():
            assert pipe is not None,       f"{name}: pipeline is None"
            assert isinstance(grid, dict), f"{name}: param_grid is not a dict"
            assert len(grid) > 0,          f"{name}: param_grid is empty"

    def test_alpha_grids_have_multiple_values(self):
        spaces = model_spaces()
        for name, (_, grid) in spaces.items():
            if "model__estimator__alpha" in grid:
                alphas = grid["model__estimator__alpha"]
                assert len(alphas) >= 5, \
                    f"{name}: alpha grid has only {len(alphas)} values — too narrow"

    def test_all_alphas_positive(self):
        spaces = model_spaces()
        for name, (_, grid) in spaces.items():
            if "model__estimator__alpha" in grid:
                for a in grid["model__estimator__alpha"]:
                    assert a > 0, f"{name}: alpha={a} is not positive"