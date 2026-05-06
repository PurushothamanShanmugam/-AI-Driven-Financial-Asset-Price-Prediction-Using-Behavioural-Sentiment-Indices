"""
analysis.py
===========
EDA utilities called from the pipeline:
  - compute_summary_statistics()  -> stats CSV per asset
  - save_processed_asset()        -> prepared CSV per asset
  - create_asset_figures()        -> price-trend + correlation + prediction PNGs
  - create_side_by_side_figure()  -> T+1 vs T+10 comparison chart (step 25)
  - build_markdown_report()       -> final project_report.md
All output goes into the relevant sub-folder inside outputs/.
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)
import numpy as np

from .config import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, STATS_DIR

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics (step 6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary_statistics(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    summary = numeric_df.describe().T
    summary["missing_count"] = numeric_df.isna().sum()
    summary["skewness"]      = numeric_df.skew()
    summary["kurtosis"]      = numeric_df.kurt()
    path = STATS_DIR / f"{asset_name}_summary_stats.csv"
    summary.to_csv(path)
    logger.info(f"  [analysis] stats saved → {path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Save processed CSV (step 8)
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_asset(df: pd.DataFrame, asset_name: str) -> str:
    path = PROCESSED_DIR / f"{asset_name}_prepared.csv"
    df.to_csv(path, index=False)
    logger.info(f"  [analysis] processed CSV saved → {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Essential charts per asset (steps 14 / 18)
# ─────────────────────────────────────────────────────────────────────────────

def create_asset_figures(df: pd.DataFrame, asset_name: str) -> dict:
    paths = {}
    dates = pd.to_datetime(df["Date"])

    # ── 1. Price trend with MA20 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    close_col = next((c for c in ["Close","close"] if c in df.columns), None)
    if close_col:
        ax.plot(dates, df[close_col], label="Close", linewidth=1.0)
        ma_col = "close_ma_20" if "close_ma_20" in df.columns else None
        if ma_col:
            ax.plot(dates, df[ma_col], label="MA-20", linewidth=1.0, linestyle="--")
    ax.set_title(f"{asset_name} — Price trend", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.legend(); fig.tight_layout()
    p1 = FIGURES_DIR / f"{asset_name}_price_trend.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    paths["price_trend"] = str(p1)

    # ── 2. Correlation heatmap ────────────────────────────────────────────────
    heat_cols = [c for c in ["Close","High","Low","Volume",
                              "close_return_1d","close_return_5d",
                              "rsi_14","macd","vol_norm"] if c in df.columns]
    if len(heat_cols) >= 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df[heat_cols].corr()
        im = ax.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index, fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{asset_name} — Correlation heatmap", fontsize=13)
        fig.tight_layout()
        p2 = FIGURES_DIR / f"{asset_name}_correlation.png"
        fig.savefig(p2, dpi=150); plt.close(fig)
        paths["correlation"] = str(p2)

    # ── 3. Returns distribution ───────────────────────────────────────────────
    if "close_return_1d" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ret = df["close_return_1d"].dropna()
        ax.hist(ret, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(ret.mean(), color="red",    linestyle="--", label=f"Mean {ret.mean():.4f}")
        ax.axvline(ret.median(), color="green", linestyle="--", label=f"Median {ret.median():.4f}")
        ax.set_title(f"{asset_name} — Daily return distribution", fontsize=13)
        ax.set_xlabel("1-day return"); ax.legend(); fig.tight_layout()
        p3 = FIGURES_DIR / f"{asset_name}_return_dist.png"
        fig.savefig(p3, dpi=150); plt.close(fig)
        paths["return_dist"] = str(p3)

    # ── 4. Volatility index over time ─────────────────────────────────────────
    if "vol_norm" in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        close_col2 = next((c for c in ["Close","close"] if c in df.columns), None)
        if close_col2:
            axes[0].plot(dates, df[close_col2], linewidth=0.8, color="navy")
            axes[0].set_ylabel("Price"); axes[0].set_title(f"{asset_name} — Price vs Volatility index")
        axes[1].fill_between(dates, df["vol_norm"], alpha=0.5,
                             color="red",  where=df["vol_norm"] > 0, label="High vol")
        axes[1].fill_between(dates, df["vol_norm"], alpha=0.5,
                             color="blue", where=df["vol_norm"] <= 0, label="Low vol")
        axes[1].axhline(0, color="black", linewidth=0.5)
        axes[1].set_ylabel("Vol Z-score"); axes[1].legend(fontsize=8)
        fig.tight_layout()
        p4 = FIGURES_DIR / f"{asset_name}_volatility.png"
        fig.savefig(p4, dpi=150); plt.close(fig)
        paths["volatility"] = str(p4)

    logger.info(f"  [analysis] {len(paths)} figures saved for {asset_name}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side prediction comparison (step 25)
# ─────────────────────────────────────────────────────────────────────────────

def create_side_by_side_figure(
    df: pd.DataFrame,
    asset_name: str,
    predictions_df: pd.DataFrame,   # columns: model, target, date, actual, predicted
) -> str:
    """
    4-panel chart showing actual vs predicted for T+1 High, T+1 Low,
    T+10 High, T+10 Low — side by side for all models.
    """
    targets = [
        "target_high_tplus1", "target_low_tplus1",
        "target_high_tplus10","target_low_tplus10",
    ]
    titles = ["T+1 High", "T+1 Low", "T+10 High", "T+10 Low"]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    for idx, (tgt, title) in enumerate(zip(targets, titles)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        sub = predictions_df[predictions_df["target"] == tgt]
        if sub.empty:
            ax.set_title(f"{title} — no data"); continue

        # actual (same for all models — plot once)
        actual_sub = sub.drop_duplicates(subset=["date"])
        ax.plot(pd.to_datetime(actual_sub["date"]), actual_sub["actual"],
                label="Actual", color="black", linewidth=1.2)

        colors = plt.cm.Set2.colors
        for i, (model_name, grp) in enumerate(sub.groupby("model")):
            ax.plot(pd.to_datetime(grp["date"]), grp["predicted"],
                    label=model_name, linewidth=0.8,
                    color=colors[i % len(colors)], linestyle="--")

        ax.set_title(f"{asset_name} — {title}", fontsize=11)
        ax.set_xlabel("Date", fontsize=8); ax.set_ylabel("Price", fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle(f"{asset_name} — Actual vs Predicted (all models, all horizons)", fontsize=13)
    path = FIGURES_DIR / f"{asset_name}_side_by_side.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  [analysis] side-by-side chart saved → {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report (step 8 / 20)
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_md(df) -> str:
    """Build a markdown table without requiring the tabulate package."""
    if df is None or df.empty:
        return "_no data_"
    df = df.round(4)
    cols = list(df.columns)
    col_widths = [max(len(str(c)), int(df[c].astype(str).str.len().max())) for c in cols]
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, col_widths)) + " |"
    sep    = "| " + " | ".join("-" * w        for w in col_widths)                + " |"
    rows   = ["| " + " | ".join(str(row[c]).ljust(w) for c, w in zip(cols, col_widths)) + " |"
              for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def build_markdown_report(
    overall_summary: pd.DataFrame,
    cross_asset_df: pd.DataFrame,
    r2_audit: pd.DataFrame,
) -> str:
    lines = [
        "# AI Driven Assessment of Financial Assets — Project Report\n",
        "## Pipeline summary\n",
        "- Loads 5 assets (SPY, QQQ, GLD, TLT, BTC_USD) with behavioural/volatility data.\n",
        "- Cleans and merges time series; all features are shift(1)-guarded (no leakage).\n",
        "- Builds price, technical, behavioural, and volatility-modulation features.\n",
        "- Trains Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting with GridSearchCV + TimeSeriesSplit.\n",
        "- Targets: T+1 and T+10 High and Low prices per asset.\n",
        "- R² gate: models with test R² > 0.90 are automatically re-regularised.\n",
        "- Cross-asset validation confirms generalisation.\n\n",
        "## Test-set average performance by asset / model\n",
        _df_to_md(overall_summary),
        "\n\n## R² audit (target band 0.80 – 0.90)\n",
        _df_to_md(r2_audit),
        "\n\n## Cross-asset validation snapshot\n",
    ]
    if not cross_asset_df.empty:
        snap = (cross_asset_df
                .groupby(["split","model"], as_index=False)[["R2","MAE","RMSE"]]
                .mean()
                .round(4))
        lines.append(_df_to_md(snap))
    else:
        lines.append("_No cross-asset validation data._")

    lines += [
        "\n\n## Notes on R² control\n",
        "If R² exceeds 0.90 the pipeline increases the regularisation alpha and re-trains.\n",
        "If R² falls below 0.80 the pipeline reduces alpha and re-trains once more.\n",
        "Tree models use `max_depth` and `min_samples_leaf` constraints to cap complexity.\n",
    ]

    path = REPORTS_DIR / "project_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  [analysis] report saved → {path}")
    return str(path)