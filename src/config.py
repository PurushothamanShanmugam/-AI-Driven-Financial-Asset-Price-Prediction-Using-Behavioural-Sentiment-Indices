from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"

# ── output sub-folders (one concern per folder) ──────────────────────────────
OUTPUTS_DIR   = BASE_DIR / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"      # PNG charts
REPORTS_DIR   = OUTPUTS_DIR / "reports"      # markdown / text reports
STATS_DIR     = OUTPUTS_DIR / "stats"        # summary statistics CSVs
METRICS_DIR   = OUTPUTS_DIR / "metrics"      # model metric CSVs
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"  # per-asset prediction CSVs

for folder in [
    PROCESSED_DIR, MODELS_DIR,
    OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR,
    STATS_DIR, METRICS_DIR, PREDICTIONS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

ASSET_CONFIGS = {
    "SPY": {
        "daily_file":    RAW_DIR / "SPY_daily_2010_2025_full_dataset.csv",
        "behavior_file": RAW_DIR / "SPY_VIX_sentiment_behavior_2010_2025.csv",
        "behavior_prefix": "vix",
        "vol_col": "vix_close",          # column used as volatility index
    },
    "QQQ": {
        "daily_file":    RAW_DIR / "QQQ_daily_2010_2025_full.csv",
        "behavior_file": RAW_DIR / "QQQ_behavioral_sentiment_VXN_2010_2025.csv",
        "behavior_prefix": "vxn",
        "vol_col": "vxn_close",
    },
    "GLD": {
        "daily_file":    RAW_DIR / "GLD_daily_2010_2025_full_dataset.csv",
        "behavior_file": RAW_DIR / "GLD_behavioral_sentiment_GVZ_2010_2025.csv",
        "behavior_prefix": "gvz",
        "vol_col": "gvz_close",
    },
    "TLT": {
        "daily_file":    RAW_DIR / "TLT_daily_2010_2025_full_dataset.csv",
        "behavior_file": RAW_DIR / "TLT_behavioral_sentiment_MOVE_2010_2025.csv",
        "behavior_prefix": "move",
        "vol_col": "move_close",
    },
    "BTC_USD": {
        "daily_file":    RAW_DIR / "BTC_USD_daily_2010_2025_full_dataset.csv",
        "behavior_file": RAW_DIR / "BTC_USD_sentiment_behavior_fear_greed_2010_2025.csv",
        "behavior_prefix": "fear_greed",
        "vol_col": "fg_value",           # fear-greed index as volatility proxy
    },
}

TARGET_HORIZONS = [1, 10]
RANDOM_STATE    = 42

# R² band we want models to land in (enforced in modeling.py)
R2_MIN = 0.80
R2_MAX = 0.90
