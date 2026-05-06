"""
validate_new_dataset.py
=======================
Validates the top 2 best-performing models per asset on a DIFFERENT dataset
(2025 calendar year — genuinely unseen rows outside the 70/15/15 split) and
produces a full side-by-side comparison with the original test-set indicators.

Outputs
───────
outputs/metrics/new_dataset_validation.csv       per-target metrics on 2025 data
outputs/metrics/new_vs_original_comparison.csv   aggregated comparison table
outputs/figures/new_vs_original_R2.png           side-by-side R² bar chart
outputs/figures/new_vs_original_MAE_RMSE.png     side-by-side MAE & RMSE chart
outputs/figures/<ASSET>_side_by_side_new.png     actual vs predicted per asset

Usage
─────
  python validate_new_dataset.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from src.logger      import get_logger, LOG_FILE
from src.config     import MODELS_DIR, METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR
from src.data_loader import load_all_assets
from src.preprocess  import prepare_asset_dataframe
from src.modeling    import build_feature_matrix, compute_rmse, TARGETS

logger = get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
NEW_YEAR   = 2025   # rows with Date >= 2025-01-01 form the "new" dataset
TOP_N      = 2      # validate top N models per asset
MID_TARGET = 0.85   # midpoint of R² band

ASSET_PALETTE = {
    "SPY":"#3B82F6","QQQ":"#8B5CF6","GLD":"#F59E0B",
    "TLT":"#10B981","BTC_USD":"#F97316",
}
MODEL_PALETTE = {
    "Ridge":"#6366F1","Lasso":"#EC4899","ElasticNet":"#14B8A6",
    "RandomForest":"#F59E0B","GradientBoosting":"#EF4444",
}
ORIG_COLOR = "#3B82F6"
NEW_COLOR  = "#F97316"


# ── helpers ───────────────────────────────────────────────────────────────────

def _eval(y_df, y_pred, model_name, ds_tag, asset):
    rows = []
    for i, tgt in enumerate(TARGETS):
        yt   = y_df.iloc[:, i].values
        yp   = y_pred[:, i]
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() < 5:
            continue
        rows.append({
            "asset":   asset,
            "model":   model_name,
            "dataset": ds_tag,
            "target":  tgt,
            "R2":      round(float(r2_score(yt[mask], yp[mask])), 6),
            "MAE":     round(float(mean_absolute_error(yt[mask], yp[mask])), 6),
            "RMSE":    round(float(compute_rmse(yt[mask], yp[mask])), 6),
            "n":       int(mask.sum()),
        })
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"New-Dataset Validation  ({NEW_YEAR} data — unseen during training)")
    logger.info("=" * 70)

    # ── 1. load & preprocess ──────────────────────────────────────────────────
    logger.info("\n[1] Loading assets …")
    raw      = load_all_assets()
    prepared = {name: prepare_asset_dataframe(df, name) for name, df in raw.items()}
    for name, df in prepared.items():
        logger.info(f"  {name}: {df.shape}")

    # ── 2. load original test metrics ─────────────────────────────────────────
    orig_path = METRICS_DIR / "all_model_metrics.csv"
    if not orig_path.exists():
        orig_path = METRICS_DIR / "all_model_summary.csv"
    if not orig_path.exists():
        logger.error(f"ERROR: run main.py first to generate training metrics.")
        return
    orig_all  = pd.read_csv(orig_path)
    orig_test = orig_all[orig_all["split"] == "test"].copy()
    logger.info(f"\n[2] Loaded original test metrics: {len(orig_test)} rows")
    logger.info(f"    Models available: {sorted(orig_test['model'].unique())}")

    # ── 3. pick top-2 models per asset by validation R² ──────────────────────
    logger.info(f"\n[3] Selecting top-{TOP_N} models per asset (by mean validation R²) …")
    orig_val = orig_all[orig_all["split"] == "validation"]
    top2: dict[str, list[str]] = {}
    for asset in sorted(orig_val["asset"].unique()):
        ranked = (orig_val[orig_val["asset"] == asset]
                  .groupby("model")["R2"].mean()
                  .sort_values(ascending=False))
        top2[asset] = ranked.index[:TOP_N].tolist()
        logger.info(f"  {asset}: {top2[asset]}")

    # ── 4. carve 2025 rows as new dataset ────────────────────────────────────
    logger.info(f"\n[4] Extracting {NEW_YEAR} rows …")
    new_metrics: list = []
    pred_rows:   list = []

    for asset, df in prepared.items():
        df["Date"] = pd.to_datetime(df["Date"])
        new_df     = df[df["Date"].dt.year >= NEW_YEAR].reset_index(drop=True)
        if len(new_df) < 20:
            logger.info(f"  {asset}: only {len(new_df)} rows — skipped"); continue
        logger.info(f"  {asset}: {len(new_df)} rows  ({new_df['Date'].min().date()} – {new_df['Date'].max().date()})")

        _, y_new, _ = build_feature_matrix(new_df)

        for mname in top2.get(asset, []):
            mpath = MODELS_DIR / f"{asset}_{mname}.joblib"
            if not mpath.exists():
                logger.info(f"    [{asset}/{mname}] joblib missing — skipped"); continue

            pkg    = joblib.load(mpath)
            fcols  = pkg["feature_columns"]
            model  = pkg["model"]
            X_new  = new_df.reindex(columns=fcols, fill_value=0.0)
            y_pred = model.predict(X_new)

            rows   = _eval(y_new, y_pred, mname, f"new_{NEW_YEAR}", asset)
            new_metrics.extend(rows)
            avg_r2 = np.mean([r["R2"] for r in rows]) if rows else float("nan")
            logger.info(f"    [{asset}/{mname}]  avg R²={avg_r2:.4f}  n={rows[0]['n'] if rows else 0}")

            for ti, tgt in enumerate(TARGETS):
                for dv, av, pv in zip(new_df["Date"].values,
                                      y_new.iloc[:, ti].values,
                                      y_pred[:, ti]):
                    pred_rows.append({
                        "asset":asset,"model":mname,"target":tgt,
                        "date":str(dv)[:10],
                        "actual":float(av),"predicted":float(pv),
                    })

    if not new_metrics:
        logger.info("\nNo metrics generated — ensure models are trained (run main.py) "
              "and that 2025 data is present in the raw CSVs."); return

    new_df_m = pd.DataFrame(new_metrics)
    new_df_m.to_csv(METRICS_DIR / "new_dataset_validation.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(PREDICTIONS_DIR / "new_dataset_predictions.csv", index=False)
    logger.info(f"\n  → new_dataset_validation.csv  ({len(new_df_m)} rows)")

    # ── 5. build comparison table ─────────────────────────────────────────────
    logger.info("\n[5] Building comparison table …")
    orig_agg = (orig_test
                .groupby(["asset","model"])[["R2","MAE","RMSE"]]
                .mean().round(4).reset_index()
                .rename(columns={"R2":"R2_orig","MAE":"MAE_orig","RMSE":"RMSE_orig"}))
    new_agg  = (new_df_m
                .groupby(["asset","model"])[["R2","MAE","RMSE"]]
                .mean().round(4).reset_index()
                .rename(columns={"R2":"R2_new","MAE":"MAE_new","RMSE":"RMSE_new"}))

    cmp = orig_agg.merge(new_agg, on=["asset","model"], how="inner")
    cmp["ΔR2"]   = (cmp["R2_new"]   - cmp["R2_orig"]).round(4)
    cmp["ΔMAE"]  = (cmp["MAE_new"]  - cmp["MAE_orig"]).round(4)
    cmp["ΔRMSE"] = (cmp["RMSE_new"] - cmp["RMSE_orig"]).round(4)
    cmp.to_csv(METRICS_DIR / "new_vs_original_comparison.csv", index=False)
    logger.info(cmp.to_string(index=False))

    # ── 6. visualisations ─────────────────────────────────────────────────────
    logger.info("\n[6] Generating charts …")
    _plot_metric_comparison(cmp)
    _plot_side_by_side_predictions(pred_rows, prepared)

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info(f"  metrics/new_dataset_validation.csv")
    logger.info(f"  metrics/new_vs_original_comparison.csv")
    logger.info(f"  figures/new_vs_original_R2.png")
    logger.info(f"  figures/new_vs_original_MAE_RMSE.png")
    logger.info(f"  figures/<ASSET>_side_by_side_new.png  (one per asset)")
    logger.info("=" * 70)


# ── chart helpers ─────────────────────────────────────────────────────────────

def _bar_label(ax, bars, fmt=".3f"):
    for bar in bars:
        h = bar.get_height()
        if abs(h) > 1e-4:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.015,
                f"{h:{fmt}}",
                ha="center", va="bottom", fontsize=7.5, rotation=40,
            )


def _plot_metric_comparison(cmp: pd.DataFrame):
    """
    Two separate figures:
      Figure A — R² side-by-side (original test vs 2025 new)
      Figure B — MAE and RMSE side-by-side
    """
    labels = [f"{r.asset}\n{r.model}" for _, r in cmp.iterrows()]
    x      = np.arange(len(cmp))
    w      = 0.38

    # ── Figure A: R² ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, len(cmp) * 1.1), 6))
    b1 = ax.bar(x - w/2, cmp["R2_orig"], w, label="Original test set",
                color=ORIG_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, cmp["R2_new"],  w, label=f"New {NEW_YEAR} dataset",
                color=NEW_COLOR,  alpha=0.85, edgecolor="white", linewidth=0.5)
    _bar_label(ax, b1); _bar_label(ax, b2)
    ax.axhline(0.90, color="red",    ls="--", lw=1.2, alpha=0.7, label="0.90 ceiling")
    ax.axhline(0.80, color="orange", ls="--", lw=1.2, alpha=0.7, label="0.80 floor")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("R²", fontsize=12, fontweight="bold")
    ax.set_title(f"R² — Original Test Set vs New {NEW_YEAR} Dataset\n"
                 f"(top-{TOP_N} models per asset)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25); ax.set_facecolor("#f8fafc")
    plt.tight_layout()
    out = FIGURES_DIR / "new_vs_original_R2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  saved → {out.name}")

    # ── Figure B: MAE + RMSE ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(cmp)*1.5), 6), sharey=False)
    fig.suptitle(f"MAE & RMSE — Original Test Set vs New {NEW_YEAR} Dataset",
                 fontsize=13, fontweight="bold")
    for ax, (orig_col, new_col, label) in zip(
        axes, [("MAE_orig","MAE_new","MAE"), ("RMSE_orig","RMSE_new","RMSE")]
    ):
        b1 = ax.bar(x - w/2, cmp[orig_col], w, label="Original test set",
                    color=ORIG_COLOR, alpha=0.85, edgecolor="white", linewidth=0.5)
        b2 = ax.bar(x + w/2, cmp[new_col],  w, label=f"New {NEW_YEAR} dataset",
                    color=NEW_COLOR,  alpha=0.85, edgecolor="white", linewidth=0.5)
        _bar_label(ax, b1, ".2f"); _bar_label(ax, b2, ".2f")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#f8fafc")
    plt.tight_layout()
    out = FIGURES_DIR / "new_vs_original_MAE_RMSE.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  saved → {out.name}")


def _plot_side_by_side_predictions(pred_rows: list, prepared: dict):
    """
    One figure per asset: 2×2 grid of T+1High / T+1Low / T+10High / T+10Low.
    Each subplot overlays actual price (black) and each model's prediction.
    """
    if not pred_rows:
        return
    pdf = pd.DataFrame(pred_rows)
    pdf["date"] = pd.to_datetime(pdf["date"])

    TARGET_LABELS = {
        "target_high_tplus1":  "T+1 High",
        "target_low_tplus1":   "T+1 Low",
        "target_high_tplus10": "T+10 High",
        "target_low_tplus10":  "T+10 Low",
    }

    for asset in sorted(pdf["asset"].unique()):
        asub   = pdf[pdf["asset"] == asset].copy()
        models = sorted(asub["model"].unique())

        fig = plt.figure(figsize=(18, 10))
        gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.30)
        fig.suptitle(
            f"{asset}  —  Actual vs Predicted  |  New {NEW_YEAR} Dataset\n"
            f"Models: {', '.join(models)}",
            fontsize=13, fontweight="bold",
        )

        pal = [MODEL_PALETTE.get(m, "#888") for m in models]

        for idx, (tgt, tlabel) in enumerate(TARGET_LABELS.items()):
            ax   = fig.add_subplot(gs[idx // 2, idx % 2])
            tsub = asub[asub["target"] == tgt]

            # actual (deduplicated — same regardless of model)
            act = tsub.drop_duplicates(subset=["date"]).sort_values("date")
            ax.plot(act["date"], act["actual"],
                    color="black", linewidth=1.8, label="Actual", zorder=5)

            for ci, mname in enumerate(models):
                ms = tsub[tsub["model"] == mname].sort_values("date")
                ax.plot(ms["date"], ms["predicted"],
                        color=pal[ci], linewidth=1.3, linestyle="--",
                        label=mname, alpha=0.88)

            ax.set_title(tlabel, fontsize=11, fontweight="bold")
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Price ($)", fontsize=9)
            ax.legend(fontsize=8.5, loc="best")
            ax.grid(alpha=0.22); ax.tick_params(axis="x", rotation=28, labelsize=8)

        out = FIGURES_DIR / f"{asset}_side_by_side_new.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        logger.info(f"  saved → {out.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Validation script failed: {e}", exc_info=True)
        import sys; sys.exit(1)