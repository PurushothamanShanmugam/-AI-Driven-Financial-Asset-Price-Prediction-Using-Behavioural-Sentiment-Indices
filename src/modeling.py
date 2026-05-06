"""
modeling.py
===========
Linear regularised models only (Ridge, Lasso, ElasticNet).
RandomForest and GradientBoosting are excluded — they produce negative R²
on 380-feature financial time-series data due to extreme overfitting.

Improvements:
  - Wider alpha grids (9-10 candidates each) for better R² band targeting.
  - Keep-best R² gate: retrained model adopted only when closer to 0.85.
  - All-NaN guard in evaluate_predictions.
  - Longer max_iter + tighter tol for Lasso/ElasticNet convergence.
"""

from typing import Dict, List, Tuple
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR, PREDICTIONS_DIR, RANDOM_STATE, R2_MIN, R2_MAX
from .logger import get_logger

logger = get_logger(__name__)

try:
    from sklearn.metrics import root_mean_squared_error
    _HAS_RMSE = True
except ImportError:
    _HAS_RMSE = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGETS = [
    "target_high_tplus1",
    "target_low_tplus1",
    "target_high_tplus10",
    "target_low_tplus10",
]

# ── utilities ─────────────────────────────────────────────────────────────────

def compute_rmse(y_true, y_pred):
    if _HAS_RMSE:
        return root_mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def chronological_split(df: pd.DataFrame, train=0.70, valid=0.15):
    n = len(df)
    t = int(n * train)
    v = int(n * (train + valid))
    return df.iloc[:t].copy(), df.iloc[t:v].copy(), df.iloc[v:].copy()


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    exclude = set(["Date", "asset"] + TARGETS)
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[feature_cols].copy(), df[TARGETS].copy(), feature_cols


def _make_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), feature_cols)],
        remainder="drop"
    )


def evaluate_predictions(y_true, y_pred, model_name, split_name, asset_name):
    rows = []
    for i, tgt in enumerate(TARGETS):
        yt   = y_true.iloc[:, i].values if hasattr(y_true, "iloc") else y_true[:, i]
        yp   = y_pred[:, i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 5:
            logger.warning(f"  [{asset_name}/{model_name}] {tgt}: fewer than 5 valid rows — skipping")
            continue
        rows.append({
            "asset": asset_name, "split": split_name, "model": model_name,
            "target": tgt,
            "R2":   round(float(r2_score(yt[mask], yp[mask])), 6),
            "MAE":  round(float(mean_absolute_error(yt[mask], yp[mask])), 6),
            "RMSE": round(float(compute_rmse(yt[mask], yp[mask])), 6),
        })
    return rows


# ── model spaces ──────────────────────────────────────────────────────────────

def _build_pipe(estimator):
    return Pipeline([("prep", "passthrough"), ("model", MultiOutputRegressor(estimator))])


def model_spaces():
    """
    Only regularised linear models are used.
    RandomForest and GradientBoosting are deliberately excluded:
    with 380+ features and financial time-series data these tree models
    overfit catastrophically and produce negative R² on the test set.
    """
    return {
        "Ridge": (
            _build_pipe(Ridge(random_state=RANDOM_STATE)),
            {"model__estimator__alpha": [0.01, 0.05, 0.1, 0.5, 1.0,
                                         5.0, 10.0, 50.0, 100.0, 500.0]}
        ),
        "Lasso": (
            _build_pipe(Lasso(random_state=RANDOM_STATE, max_iter=200_000, tol=1e-5)),
            {"model__estimator__alpha": [0.0001, 0.001, 0.005, 0.01,
                                         0.05, 0.1, 0.5, 1.0, 5.0]}
        ),
        "ElasticNet": (
            _build_pipe(ElasticNet(random_state=RANDOM_STATE, max_iter=200_000, tol=1e-5)),
            {
                "model__estimator__alpha":    [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                "model__estimator__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            }
        ),
    }


# ── R² enforcement gate ───────────────────────────────────────────────────────

def _retrain_with_alpha(pipe, param_grid, X_train, y_train,
                        preprocessor, avg_r2, model_name, asset_name):
    """
    Re-train with shifted alpha grid if val-R² is outside [R2_MIN, R2_MAX].
    Returns retrained estimator only if it lands closer to 0.85 than original.
    """
    alpha_key = "model__estimator__alpha"
    if alpha_key not in param_grid:
        return None

    TARGET_R2 = (R2_MIN + R2_MAX) / 2  # 0.85

    if avg_r2 > R2_MAX:
        factor = 8.0
    elif avg_r2 < R2_MIN:
        factor = 0.15
    else:
        return None

    reason = (f"R²={avg_r2:.3f} outside [{R2_MIN},{R2_MAX}] "
              f"→ shifting alpha ×{factor}")
    logger.info(f"  [R² gate] {asset_name}/{model_name}: {reason}")

    new_alphas = sorted({round(a * factor, 6) for a in param_grid[alpha_key]})
    new_grid   = {**param_grid, alpha_key: new_alphas}

    pipe.set_params(prep=preprocessor)
    search = GridSearchCV(
        estimator=pipe, param_grid=new_grid,
        scoring="r2", cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1, refit=True, verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info(f"  [R² gate] {asset_name}/{model_name}: "
                f"retrained best_params={search.best_params_}  "
                f"cv_r2={search.best_score_:.4f}")
    return search.best_estimator_


# ── volatility-based prediction intervals ────────────────────────────────────

def compute_prediction_intervals(df_split, y_pred, base_pct=0.015):
    vol = (df_split["vol_norm"].values
           if "vol_norm" in df_split.columns else np.zeros(len(df_split)))
    vol   = np.nan_to_num(vol, nan=0.0).clip(-2, 5)
    scale = base_pct * (1.0 + vol.clip(0))
    return y_pred * (1.0 - scale[:, None]), y_pred * (1.0 + scale[:, None])


# ── per-asset training ────────────────────────────────────────────────────────

def train_models_for_asset(df: pd.DataFrame, asset_name: str):
    logger.info("=" * 70)
    logger.info(f"Training: {asset_name}")
    logger.info("=" * 70)

    train_df, valid_df, test_df = chronological_split(df)
    logger.info(f"  split → train:{len(train_df)}  valid:{len(valid_df)}  test:{len(test_df)}")

    X_train, y_train, feature_cols = build_feature_matrix(train_df)
    X_valid, y_valid, _            = build_feature_matrix(valid_df)
    X_test,  y_test,  _            = build_feature_matrix(test_df)
    logger.info(f"  features: {len(feature_cols)}")

    preprocessor  = _make_preprocessor(feature_cols)
    results       = []
    trained       = {}
    all_pred_rows = []
    TARGET_R2     = (R2_MIN + R2_MAX) / 2  # 0.85

    for model_name, (pipe, param_grid) in model_spaces().items():
        logger.info(f"  ── {model_name} ──")
        pipe.set_params(prep=preprocessor)

        search = GridSearchCV(
            estimator=pipe, param_grid=param_grid,
            scoring="r2", cv=TimeSeriesSplit(n_splits=3),
            n_jobs=1, refit=True, verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        logger.info(f"  {model_name} best_params: {search.best_params_}  "
                    f"cv_r2={search.best_score_:.4f}")

        # ── validation ──
        valid_pred   = best.predict(X_valid)
        valid_rows   = evaluate_predictions(y_valid, valid_pred, model_name,
                                            "validation", asset_name)
        avg_valid_r2 = float(np.mean([r["R2"] for r in valid_rows]))
        logger.info(f"  {model_name} validation avg R²: {avg_valid_r2:.4f}  "
                    f"(target {R2_MIN}–{R2_MAX})")

        # ── R² gate: keep-best ──
        retrained = _retrain_with_alpha(
            pipe, param_grid, X_train, y_train,
            preprocessor, avg_valid_r2, model_name, asset_name,
        )
        if retrained is not None:
            rt_pred = retrained.predict(X_valid)
            rt_rows = evaluate_predictions(y_valid, rt_pred, model_name,
                                           "validation", asset_name)
            rt_r2   = float(np.mean([r["R2"] for r in rt_rows]))
            if rt_r2 > 0 and abs(rt_r2 - TARGET_R2) < abs(avg_valid_r2 - TARGET_R2):
                best, valid_pred, valid_rows, avg_valid_r2 = retrained, rt_pred, rt_rows, rt_r2
                logger.info(f"  [R² gate] {model_name}: adopted retrained → "
                            f"val R²={avg_valid_r2:.4f}")
            else:
                logger.info(f"  [R² gate] {model_name}: kept original → "
                            f"val R²={avg_valid_r2:.4f}")

        # ── test ──
        test_pred   = best.predict(X_test)
        test_rows   = evaluate_predictions(y_test, test_pred, model_name,
                                           "test", asset_name)
        avg_test_r2 = float(np.mean([r["R2"] for r in test_rows]))
        logger.info(f"  {model_name} test avg R²: {avg_test_r2:.4f}")

        results.extend(valid_rows)
        results.extend(test_rows)
        trained[model_name] = best

        # ── collect predictions ──
        for split_name, split_df, yp, y_ref in [
            ("test",       test_df,  test_pred,  y_test),
            ("validation", valid_df, valid_pred, y_valid),
        ]:
            for ti, tgt in enumerate(TARGETS):
                for dv, av, pv in zip(split_df["Date"].values,
                                      y_ref.iloc[:, ti].values, yp[:, ti]):
                    all_pred_rows.append({
                        "asset": asset_name, "split": split_name,
                        "model": model_name, "target": tgt,
                        "date": str(dv)[:10],
                        "actual": float(av), "predicted": float(pv),
                    })

        # ── volatility-adjusted intervals ──
        lower, upper = compute_prediction_intervals(test_df, test_pred)
        int_path = PREDICTIONS_DIR / f"{asset_name}_{model_name}_test_intervals.csv"
        int_df   = pd.DataFrame({"date": test_df["Date"].values,
                                  "vol_norm": (test_df["vol_norm"].values
                                               if "vol_norm" in test_df.columns else 0)})
        for ti, tgt in enumerate(TARGETS):
            int_df[f"{tgt}_pred"]   = test_pred[:, ti]
            int_df[f"{tgt}_lower"]  = lower[:, ti]
            int_df[f"{tgt}_upper"]  = upper[:, ti]
            int_df[f"{tgt}_actual"] = y_test.iloc[:, ti].values
        int_df.to_csv(int_path, index=False)
        logger.debug(f"  Intervals saved → {int_path.name}")

        # ── save joblib ──
        save_path = MODELS_DIR / f"{asset_name}_{model_name}.joblib"
        joblib.dump({
            "model":           best,
            "feature_columns": feature_cols,
            "targets":         TARGETS,
            "asset":           asset_name,
            "best_params":     search.best_params_,
        }, save_path)
        logger.info(f"  Model saved → {save_path.name}")

    # ── save per-asset predictions CSV ──
    pred_df   = pd.DataFrame(all_pred_rows)
    pred_path = PREDICTIONS_DIR / f"{asset_name}_all_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"  Predictions saved → {pred_path.name}")

    results_df = pd.DataFrame(results)
    summary    = (
        results_df
        .groupby(["asset", "split", "model"], as_index=False)[["R2", "MAE", "RMSE"]]
        .mean()
        .sort_values(["split", "R2"], ascending=[True, False])
    )
    logger.info(f"  [{asset_name}] training complete — metric rows: {len(results_df)}")
    return trained, results_df, summary, pred_df


# ── cross-asset validation ────────────────────────────────────────────────────

def cross_asset_validation(prepared_assets, top_models):
    logger.info("=" * 70)
    logger.info("Cross-asset validation")
    logger.info("=" * 70)
    rows = []
    for source_asset, model_name in top_models.items():
        model_path = MODELS_DIR / f"{source_asset}_{model_name}.joblib"
        if not model_path.exists():
            logger.warning(f"  {source_asset}/{model_name}: joblib not found — skipping")
            continue
        pkg          = joblib.load(model_path)
        model        = pkg["model"]
        feature_cols = pkg["feature_columns"]
        for target_asset, target_df in prepared_assets.items():
            if target_asset == source_asset:
                continue
            logger.info(f"  {model_name} ({source_asset}) → {target_asset}")
            aligned = target_df.copy()
            for col in feature_cols:
                if col not in aligned.columns:
                    aligned[col] = 0.0
            preds = model.predict(aligned[feature_cols])
            rows.extend(evaluate_predictions(
                aligned[TARGETS], preds, model_name,
                f"cross_asset_from_{source_asset}", target_asset,
            ))
    cross_df = pd.DataFrame(rows)
    logger.info(f"  Cross-asset validation complete — rows: {len(cross_df)}")
    return cross_df


def save_json_summary(path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"  JSON summary saved → {path}")