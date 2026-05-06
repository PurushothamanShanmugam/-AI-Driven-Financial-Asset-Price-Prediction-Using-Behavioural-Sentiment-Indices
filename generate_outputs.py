"""
generate_outputs.py
===================
Run this INSTEAD of main.py when models are already trained.

It loads every saved .joblib file from the models/ folder, re-runs
prediction + evaluation on the validation and test splits, then
generates ALL outputs:

  outputs/metrics/    — all_model_metrics.csv, all_model_summary.csv, r2_audit.csv
  outputs/predictions/— per-asset predictions + volatility intervals
  outputs/figures/    — price trend, correlation, return dist, volatility, side-by-side
  outputs/stats/      — per-asset summary statistics CSVs
  outputs/reports/    — project_report.md
  data/processed/     — per-asset prepared CSVs
  outputs/run_summary.json

Usage
-----
  conda activate finance_behavior_project
  python generate_outputs.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
from src.logger import get_logger, LOG_FILE
from src.config import (
    MODELS_DIR, OUTPUTS_DIR, METRICS_DIR, PREDICTIONS_DIR,
    FIGURES_DIR, STATS_DIR, R2_MIN, R2_MAX,
)
from src.data_loader import load_all_assets
from src.preprocess import prepare_asset_dataframe
logger = get_logger(__name__)

from src.modeling import (
    chronological_split,
    build_feature_matrix,
    evaluate_predictions,
    compute_prediction_intervals,
    cross_asset_validation,
    save_json_summary,
    TARGETS,
)
from src.analysis import (
    compute_summary_statistics,
    save_processed_asset,
    create_asset_figures,
    create_side_by_side_figure,
    build_markdown_report,
)

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "BTC_USD"]
MODELS = ["Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting"]


def _load_model(asset: str, model_name: str):
    path = MODELS_DIR / f"{asset}_{model_name}.joblib"
    if not path.exists():
        logger.info(f"  [SKIP] {path.name} not found — skipping")
        return None
    pkg = joblib.load(path)
    return pkg  # keys: model, feature_columns, targets, asset, best_params


def run_outputs_only():
    logger.info("\n" + "=" * 70)
    logger.info("generate_outputs.py — loading saved models, skipping training")
    logger.info("=" * 70)

    # ── Step 1 + 3: load and preprocess (fast — no training) ──────────────────
    logger.info("\n[STEP 1] Loading raw assets")
    raw_assets = load_all_assets()

    logger.info("\n[STEP 3] Preprocessing (feature engineering + target creation)")
    prepared_assets = {}
    for asset_name, df in raw_assets.items():
        prepared_assets[asset_name] = prepare_asset_dataframe(df, asset_name)

    # ── Step 6: summary statistics ────────────────────────────────────────────
    logger.info("\n[STEP 6] Summary statistics → outputs/stats/")
    for asset_name, df in prepared_assets.items():
        compute_summary_statistics(df, asset_name)

    # ── Step 8: save processed CSVs ───────────────────────────────────────────
    logger.info("\n[STEP 8] Saving processed CSVs → data/processed/")
    for asset_name, df in prepared_assets.items():
        save_processed_asset(df, asset_name)

    # ── Step 14: essential figures ────────────────────────────────────────────
    logger.info("\n[STEP 14] Generating essential charts → outputs/figures/")
    for asset_name, df in prepared_assets.items():
        create_asset_figures(df, asset_name)

    # ── Steps 11-13: evaluate saved models, build metrics ────────────────────
    logger.info("\n[STEPS 11-13] Evaluating saved models on validation + test splits")
    all_metrics   = []
    all_summaries = []
    all_preds     = []

    for asset_name, df in prepared_assets.items():
        logger.info(f"\n  Asset: {asset_name}")
        train_df, valid_df, test_df = chronological_split(df)

        _, y_valid, _ = build_feature_matrix(valid_df)
        _, y_test,  _ = build_feature_matrix(test_df)

        results       = []
        pred_rows     = []

        for model_name in MODELS:
            pkg = _load_model(asset_name, model_name)
            if pkg is None:
                continue

            model        = pkg["model"]
            feature_cols = pkg["feature_columns"]

            # Align columns — fill any missing feature with 0
            for split_name, split_df, y_true in [
                ("validation", valid_df, y_valid),
                ("test",        test_df,  y_test),
            ]:
                X = split_df.reindex(columns=feature_cols, fill_value=0.0)
                y_pred = model.predict(X)

                rows = evaluate_predictions(y_true, y_pred, model_name, split_name, asset_name)
                results.extend(rows)
                avg_r2 = float(np.mean([r["R2"] for r in rows]))
                logger.info(f"    {model_name:20s} | {split_name:10s} | avg R²={avg_r2:.4f}")

                for ti, tgt in enumerate(TARGETS):
                    for date_val, act_val, pred_val in zip(
                        split_df["Date"].values,
                        y_true.iloc[:, ti].values,
                        y_pred[:, ti]
                    ):
                        pred_rows.append({
                            "asset": asset_name, "split": split_name,
                            "model": model_name, "target": tgt,
                            "date": str(date_val)[:10],
                            "actual": float(act_val), "predicted": float(pred_val),
                        })

            # Volatility-adjusted intervals (test only)
            X_test_aligned = test_df.reindex(columns=feature_cols, fill_value=0.0)
            test_pred      = model.predict(X_test_aligned)
            lower, upper   = compute_prediction_intervals(test_df, test_pred)
            int_df = pd.DataFrame({"date": test_df["Date"].values})
            int_df["vol_norm"] = test_df["vol_norm"].values if "vol_norm" in test_df.columns else 0.0
            for ti, tgt in enumerate(TARGETS):
                int_df[f"{tgt}_pred"]   = test_pred[:, ti]
                int_df[f"{tgt}_lower"]  = lower[:, ti]
                int_df[f"{tgt}_upper"]  = upper[:, ti]
                int_df[f"{tgt}_actual"] = y_test.iloc[:, ti].values
            int_path = PREDICTIONS_DIR / f"{asset_name}_{model_name}_test_intervals.csv"
            int_df.to_csv(int_path, index=False)

        # Per-asset predictions CSV
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(PREDICTIONS_DIR / f"{asset_name}_all_predictions.csv", index=False)

        results_df = pd.DataFrame(results)
        summary = (
            results_df
            .groupby(["asset","split","model"], as_index=False)[["R2","MAE","RMSE"]]
            .mean()
            .sort_values(["split","R2"], ascending=[True, False])
        )
        all_metrics.append(results_df)
        all_summaries.append(summary)
        all_preds.append(pred_df)

    # ── Combine and save metric tables ────────────────────────────────────────
    logger.info("\n[STEP 11] Saving combined metric tables → outputs/metrics/")
    metrics_all  = pd.concat(all_metrics,   ignore_index=True) if all_metrics   else pd.DataFrame()
    summary_all  = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    preds_all    = pd.concat(all_preds,     ignore_index=True) if all_preds     else pd.DataFrame()

    metrics_all.to_csv(METRICS_DIR / "all_model_metrics.csv",   index=False)
    summary_all.to_csv(METRICS_DIR / "all_model_summary.csv",   index=False)
    preds_all.to_csv(  PREDICTIONS_DIR / "all_predictions.csv", index=False)
    logger.info(f"  metric rows: {len(metrics_all)}  |  summary rows: {len(summary_all)}")

    # ── Step 5: pick top model per asset ──────────────────────────────────────
    logger.info("\n[STEP 5] Top validation model per asset")
    top_models = {}
    if not summary_all.empty:
        for asset_name in summary_all["asset"].unique():
            sub = summary_all[
                (summary_all["asset"] == asset_name) &
                (summary_all["split"] == "validation")
            ].sort_values("R2", ascending=False)
            if not sub.empty:
                top_models[asset_name] = sub.iloc[0]["model"]
                logger.info(f"  {asset_name}: {top_models[asset_name]}  R²={sub.iloc[0]['R2']:.4f}")

    # ── Step 20: R² audit ─────────────────────────────────────────────────────
    logger.info("\n[STEP 20] R² audit → outputs/metrics/r2_audit.csv")
    r2_audit = pd.DataFrame()
    if not summary_all.empty:
        test_s = summary_all[summary_all["split"] == "test"].copy()
        test_s["status"] = test_s["R2"].apply(
            lambda r: "OK" if R2_MIN <= r <= R2_MAX
                      else ("TOO HIGH" if r > R2_MAX else "TOO LOW")
        )
        r2_audit = test_s[["asset","model","R2","status"]].sort_values(
            ["asset","R2"], ascending=[True, False]
        )
        r2_audit.to_csv(METRICS_DIR / "r2_audit.csv", index=False)
        logger.info(r2_audit.to_string(index=False))

    # ── Steps 23-24: cross-asset validation ───────────────────────────────────
    logger.info("\n[STEPS 23-24] Cross-asset validation")
    cross_df = pd.DataFrame()
    if top_models:
        cross_df = cross_asset_validation(prepared_assets, top_models)
        cross_df.to_csv(METRICS_DIR / "cross_asset_validation.csv", index=False)

    # ── Step 25: side-by-side charts ──────────────────────────────────────────
    logger.info("\n[STEP 25] Side-by-side prediction charts → outputs/figures/")
    if not preds_all.empty:
        for asset_name in preds_all["asset"].unique():
            asset_pred = preds_all[
                (preds_all["asset"] == asset_name) &
                (preds_all["split"] == "test")
            ]
            if not asset_pred.empty:
                create_side_by_side_figure(
                    prepared_assets[asset_name], asset_name, asset_pred
                )

    # ── Step 8/20: markdown report ────────────────────────────────────────────
    logger.info("\n[STEP 8/20] Building markdown report → outputs/reports/")
    test_summary_for_report = (
        summary_all[summary_all["split"] == "test"].round(4)
        if not summary_all.empty else pd.DataFrame()
    )
    build_markdown_report(test_summary_for_report, cross_df, r2_audit)

    # ── JSON run summary ──────────────────────────────────────────────────────
    final_summary = {
        "mode":                "outputs_only (models loaded from disk)",
        "assets_processed":    list(prepared_assets.keys()),
        "top_models":          top_models,
        "metrics_rows":        int(len(metrics_all)),
        "summary_rows":        int(len(summary_all)),
        "cross_asset_rows":    int(len(cross_df)),
        "outputs": {
            "metrics":     str(METRICS_DIR),
            "figures":     str(FIGURES_DIR),
            "reports":     str(OUTPUTS_DIR / "reports"),
            "stats":       str(STATS_DIR),
            "predictions": str(PREDICTIONS_DIR),
        },
    }
    save_json_summary(OUTPUTS_DIR / "run_summary.json", final_summary)

    logger.info("\n" + "=" * 70)
    logger.info("DONE — all outputs generated.  Launch the dashboard with:")
    logger.info("  streamlit run app/dashboard.py")
    logger.info("=" * 70)
    return final_summary


if __name__ == "__main__":
    run_outputs_only()