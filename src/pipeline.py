"""
pipeline.py
===========
Orchestrates all 26 steps end-to-end:
  Step 1  — load assets
  Step 3  — preprocess (clean, features, targets)
  Step 6  — summary statistics   → outputs/stats/
  Step 8  — save processed CSVs  → data/processed/
  Step 14 — essential figures    → outputs/figures/
  Step 9-13 — train + compare all models
  Step 20 — R² audit table
  Step 23-24 — cross-asset validation
  Step 25 — side-by-side prediction charts
  Step 8/20 — markdown report    → outputs/reports/
"""

import json
from pathlib import Path

import pandas as pd

from .logger import get_logger
from .config import OUTPUTS_DIR, METRICS_DIR, PREDICTIONS_DIR
from .data_loader import load_all_assets
from .preprocess import prepare_asset_dataframe
from .analysis import (
    compute_summary_statistics,
    save_processed_asset,
    create_asset_figures,
    create_side_by_side_figure,
    build_markdown_report,
)
from .modeling import (
    train_models_for_asset,
    cross_asset_validation,
    save_json_summary,
    TARGETS,
)

logger = get_logger(__name__)


def run_full_pipeline():
    logger.info("\n" + "=" * 70)
    logger.info("AI Financial Behaviour Project — full pipeline")
    logger.info("=" * 70)

    # ── Step 1: load ──────────────────────────────────────────────────────────
    logger.info("\n[STEP 1] Loading asset datasets")
    raw_assets = load_all_assets()         # Dict[str, DataFrame]
    logger.info(f"  Loaded: {list(raw_assets.keys())}")

    # ── Step 3: preprocess ───────────────────────────────────────────────────
    logger.info("\n[STEP 3] Preprocessing asset dataframes")
    prepared_assets = {}
    for asset_name, df in raw_assets.items():
        prepared_assets[asset_name] = prepare_asset_dataframe(df, asset_name)

    # ── Step 6: summary statistics ───────────────────────────────────────────
    logger.info("\n[STEP 6] Computing summary statistics → outputs/stats/")
    for asset_name, df in prepared_assets.items():
        compute_summary_statistics(df, asset_name)

    # ── Step 8: save processed CSVs ──────────────────────────────────────────
    logger.info("\n[STEP 8] Saving processed CSVs → data/processed/")
    for asset_name, df in prepared_assets.items():
        save_processed_asset(df, asset_name)

    # ── Step 14: essential figures ───────────────────────────────────────────
    logger.info("\n[STEP 14] Generating essential charts → outputs/figures/")
    for asset_name, df in prepared_assets.items():
        create_asset_figures(df, asset_name)

    # ── Steps 9-13: train models ─────────────────────────────────────────────
    logger.info("\n[STEPS 9-13] Training and comparing models")
    all_metrics   = []
    all_summaries = []
    all_preds     = []
    trained_by_asset = {}

    for asset_name, df in prepared_assets.items():
        trained, metrics_df, summary_df, pred_df = train_models_for_asset(df, asset_name)
        trained_by_asset[asset_name] = trained
        all_metrics.append(metrics_df)
        all_summaries.append(summary_df)
        all_preds.append(pred_df)

    # ── Step 11: combined metrics ─────────────────────────────────────────────
    logger.info("\n[STEP 11] Saving combined metric tables → outputs/metrics/")
    metrics_all  = pd.concat(all_metrics,   ignore_index=True) if all_metrics   else pd.DataFrame()
    summary_all  = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    preds_all    = pd.concat(all_preds,     ignore_index=True) if all_preds     else pd.DataFrame()

    metrics_all.to_csv( METRICS_DIR / "all_model_metrics.csv",  index=False)
    summary_all.to_csv( METRICS_DIR / "all_model_summary.csv",  index=False)
    preds_all.to_csv(   PREDICTIONS_DIR / "all_predictions.csv", index=False)
    logger.info(f"  metrics rows: {len(metrics_all)}  summary rows: {len(summary_all)}")

    # ── Step 5: pick top model per asset ─────────────────────────────────────
    logger.info("\n[STEP 5] Selecting top validation model per asset")
    top_models = {}
    if not summary_all.empty:
        for asset_name in summary_all["asset"].unique():
            sub = summary_all[
                (summary_all["asset"] == asset_name) &
                (summary_all["split"] == "validation")
            ].sort_values("R2", ascending=False)
            if not sub.empty:
                top_models[asset_name] = sub.iloc[0]["model"]
                logger.info(f"  {asset_name}: top model = {top_models[asset_name]}"
                      f"  R²={sub.iloc[0]['R2']:.4f}")

    # ── Step 20: R² audit ─────────────────────────────────────────────────────
    logger.info("\n[STEP 20] R² audit table (target band 0.80–0.90)")
    r2_audit = pd.DataFrame()
    if not summary_all.empty:
        test_summary = summary_all[summary_all["split"] == "test"].copy()
        test_summary["in_band"] = test_summary["R2"].between(0.80, 0.90)
        test_summary["status"]  = test_summary["R2"].apply(
            lambda r: "OK" if 0.80 <= r <= 0.90
                      else ("TOO HIGH" if r > 0.90 else "TOO LOW")
        )
        r2_audit = test_summary[["asset","model","R2","status"]].sort_values(["asset","R2"], ascending=[True,False])
        r2_audit.to_csv(METRICS_DIR / "r2_audit.csv", index=False)
        logger.info(r2_audit.to_string(index=False))

    # ── Steps 23-24: cross-asset validation ──────────────────────────────────
    logger.info("\n[STEPS 23-24] Cross-asset validation")
    cross_df = pd.DataFrame()
    if top_models:
        cross_df = cross_asset_validation(prepared_assets, top_models)
        cross_df.to_csv(METRICS_DIR / "cross_asset_validation.csv", index=False)

    # ── Step 25: side-by-side prediction charts ──────────────────────────────
    logger.info("\n[STEP 25] Side-by-side prediction charts → outputs/figures/")
    if not preds_all.empty:
        for asset_name in preds_all["asset"].unique():
            asset_pred = preds_all[(preds_all["asset"] == asset_name) & (preds_all["split"] == "test")]
            if not asset_pred.empty:
                create_side_by_side_figure(
                    prepared_assets[asset_name], asset_name, asset_pred
                )

    # ── Step 8/20: markdown report ────────────────────────────────────────────
    logger.info("\n[STEP 8/20] Building markdown report → outputs/reports/")
    test_summary_for_report = (
        summary_all[summary_all["split"] == "test"]
        .round(4)
        if not summary_all.empty else pd.DataFrame()
    )
    build_markdown_report(test_summary_for_report, cross_df, r2_audit)

    # ── Step 7: JSON run summary ──────────────────────────────────────────────
    final_summary = {
        "assets_processed":    list(prepared_assets.keys()),
        "top_models":          top_models,
        "metrics_rows":        int(len(metrics_all)),
        "summary_rows":        int(len(summary_all)),
        "cross_asset_rows":    int(len(cross_df)),
        "outputs": {
            "metrics":         str(METRICS_DIR),
            "figures":         str(OUTPUTS_DIR / "figures"),
            "reports":         str(OUTPUTS_DIR / "reports"),
            "stats":           str(OUTPUTS_DIR / "stats"),
            "predictions":     str(PREDICTIONS_DIR),
        }
    }
    save_json_summary(OUTPUTS_DIR / "run_summary.json", final_summary)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    return {
        "prepared_assets":      prepared_assets,
        "trained_by_asset":     trained_by_asset,
        "metrics":              metrics_all,
        "summary":              summary_all,
        "cross_asset":          cross_df,
        "top_models":           top_models,
        "run_summary":          final_summary,
        "all_predictions":      preds_all,
    }