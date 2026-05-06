from typing import Dict
import pandas as pd
from .config import ASSET_CONFIGS
from .logger import get_logger

logger = get_logger(__name__)


def _load_price_file(path) -> pd.DataFrame:
    logger.debug(f"Loading price file: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def _prepare_behavior_file(asset_name: str, behavior_path, prefix: str) -> pd.DataFrame:
    logger.debug(f"Loading behaviour file for {asset_name}: {behavior_path}")
    behavior_df = pd.read_csv(behavior_path)
    behavior_df["Date"] = pd.to_datetime(behavior_df["Date"], errors="coerce")
    behavior_df = behavior_df.sort_values("Date").reset_index(drop=True)

    if asset_name == "BTC_USD":
        rename = {
            "FearGreedValue":          "fg_value",
            "FearGreedClassification": "fg_classification",
            "TimestampUnix":           "fg_timestamp_unix",
            "TimeUntilUpdate":         "fg_time_until_update",
        }
        return behavior_df.rename(columns={k: v for k, v in rename.items() if k in behavior_df.columns})

    keep = [c for c in ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
            if c in behavior_df.columns]
    behavior_df = behavior_df[keep].copy()
    rename_map = {c: f"{prefix}_{c.lower().replace(' ', '_')}"
                  for c in behavior_df.columns if c != "Date"}
    return behavior_df.rename(columns=rename_map)


def load_all_assets() -> Dict[str, pd.DataFrame]:
    """Load and merge price + sentiment data for all 5 assets."""
    logger.info("Loading all assets from raw CSV files...")
    out: Dict[str, pd.DataFrame] = {}
    for asset_name, cfg in ASSET_CONFIGS.items():
        try:
            price_df    = _load_price_file(cfg["daily_file"])
            behavior_df = _prepare_behavior_file(asset_name, cfg["behavior_file"],
                                                 cfg["behavior_prefix"])
            merged = price_df.merge(behavior_df, on="Date", how="left")
            merged["asset"] = asset_name
            out[asset_name] = merged
            logger.info(f"  {asset_name}: {len(merged)} rows, {len(merged.columns)} cols")
        except FileNotFoundError as e:
            logger.error(f"  {asset_name}: raw file not found — {e}")
            raise
        except Exception as e:
            logger.error(f"  {asset_name}: unexpected error loading data — {e}")
            raise
    logger.info(f"All assets loaded: {list(out.keys())}")
    return out