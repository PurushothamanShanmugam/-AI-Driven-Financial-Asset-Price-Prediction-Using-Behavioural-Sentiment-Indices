"""
preprocess.py
=============
Single-responsibility pre-processing pipeline:
  1. Standardise date column
  2. Attach asset label
  3. Coerce all numerics
  4. Basic cleaning (dedup, ffill, bfill, median fallback)
  5. Price features   — ALL lagged/rolled values use .shift(1) (leakage fix)
  6. Behavioural / volatility features — same .shift(1) guard
  7. Volatility-modulation column for interval widening in dashboard
  8. Target creation  — .shift(-h) forward-looking, created LAST
  9. Final cleanup    — inf→nan→ffill→bfill→median, drop NaN target rows
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ─── Step 1: date ────────────────────────────────────────────────────────────

def _standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = _find_col(df, ["Date","date","Datetime","datetime","Timestamp","timestamp"])
    if date_col is None:
        raise ValueError("No date column found.")
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


# ─── Step 2: asset label ─────────────────────────────────────────────────────

def _ensure_asset_column(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    df = df.copy()
    df["asset"] = asset_name
    return df


# ─── Step 3: numeric coercion ────────────────────────────────────────────────

def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    skip = {"Date","asset","fg_classification"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─── Step 4: basic cleaning ──────────────────────────────────────────────────

def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)
    num_cols = [c for c in df.columns if c not in {"Date","asset","fg_classification"}]
    if num_cols:
        df[num_cols] = df[num_cols].ffill().bfill()
        medians = df[num_cols].median(numeric_only=True)
        df[num_cols] = df[num_cols].fillna(medians)
    return df


# ─── Step 5: price features — ALL .shift(1) ──────────────────────────────────

def _create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    open_col   = _find_col(df, ["Open","open"])
    high_col   = _find_col(df, ["High","high"])
    low_col    = _find_col(df, ["Low","low"])
    close_col  = _find_col(df, ["Close","close","Adj Close","adj_close"])
    volume_col = _find_col(df, ["Volume","volume"])

    if close_col:
        c = df[close_col]
        df["close_return_1d"]  = c.pct_change(1).shift(1)
        df["close_return_5d"]  = c.pct_change(5).shift(1)
        df["close_return_10d"] = c.pct_change(10).shift(1)

        for lag in [1,2,3,5,10]:
            df[f"close_lag_{lag}"] = c.shift(lag)

        for w in [3,5,10,20,50]:
            df[f"close_ma_{w}"]  = c.rolling(w).mean().shift(1)
            df[f"close_std_{w}"] = c.rolling(w).std().shift(1)
            df[f"close_ema_{w}"] = c.ewm(span=w, adjust=False).mean().shift(1)

        df["close_zscore_10"] = (
            (c - c.rolling(10).mean()) / (c.rolling(10).std() + 1e-9)
        ).shift(1)
        df["log_close"] = np.log1p(c.clip(lower=0)).shift(1)

        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"] = (100 - (100 / (1 + gain / (loss + 1e-9)))).shift(1)

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        df["macd"]        = macd.shift(1)
        df["macd_signal"] = sig.shift(1)
        df["macd_hist"]   = (macd - sig).shift(1)

    if high_col:
        for lag in [1,2,5]: df[f"high_lag_{lag}"] = df[high_col].shift(lag)
    if low_col:
        for lag in [1,2,5]: df[f"low_lag_{lag}"]  = df[low_col].shift(lag)
    if open_col:
        df["open_lag_1"] = df[open_col].shift(1)

    if high_col and low_col:
        df["intraday_range"]     = (df[high_col] - df[low_col]).shift(1)
        df["intraday_range_pct"] = ((df[high_col] - df[low_col]) / (df[low_col] + 1e-9)).shift(1)

    if open_col and close_col:
        df["open_close_diff"]   = (df[close_col] - df[open_col]).shift(1)
        df["open_close_return"] = ((df[close_col] - df[open_col]) / (df[open_col] + 1e-9)).shift(1)

    if volume_col:
        v = df[volume_col]
        df["volume_lag_1"] = v.shift(1)
        df["volume_ma_5"]  = v.rolling(5).mean().shift(1)
        df["volume_ma_20"] = v.rolling(20).mean().shift(1)
        df["log_volume"]   = np.log1p(v.clip(lower=0)).shift(1)

    return df


# ─── Step 6: behavioural / volatility features — .shift(1) everywhere ────────

_PRICE_RAW_COLS = {
    "Open","High","Low","Close","Adj Close","Volume",
    "open","high","low","close","volume",
}

def _create_behavioral_and_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    exclude = _PRICE_RAW_COLS | {"Date","asset","fg_classification"}
    candidate_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    new_cols = {}
    for col in candidate_cols:
        s = df[col]
        roll10_mean = s.rolling(10).mean()
        roll10_std  = s.rolling(10).std()
        new_cols[f"{col}_lag_1"]     = s.shift(1)
        new_cols[f"{col}_lag_3"]     = s.shift(3)
        new_cols[f"{col}_ma_5"]      = s.rolling(5).mean().shift(1)
        new_cols[f"{col}_ma_10"]     = roll10_mean.shift(1)
        new_cols[f"{col}_std_10"]    = roll10_std.shift(1)
        new_cols[f"{col}_zscore_10"] = (
            (s - roll10_mean) / (roll10_std + 1e-9)
        ).shift(1)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    if "fg_classification" in df.columns:
        dummies = pd.get_dummies(
            df["fg_classification"].shift(1).astype(str),
            prefix="fg_class", dummy_na=False
        )
        df = pd.concat([df, dummies], axis=1)
    return df


# ─── Step 7: volatility-modulation column ────────────────────────────────────

VOL_COL_MAP = {
    "SPY":     "vix_close",
    "QQQ":     "vxn_close",
    "GLD":     "gvz_close",
    "TLT":     "move_close",
    "BTC_USD": "fg_value",
}

def _add_volatility_modulation(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    """
    Adds vol_norm (60-day z-score) and vol_pctrank (252-day percentile rank)
    of the primary volatility index.  Both are shifted(1) — leakage-free.
    The dashboard uses vol_norm to scale the width of prediction intervals:
    wider when vol_norm is high, narrower when it is negative.
    """
    df = df.copy()
    vol_col = VOL_COL_MAP.get(asset_name)
    if vol_col and vol_col in df.columns:
        v = df[vol_col]
        df["vol_norm"] = (
            (v - v.rolling(60, min_periods=10).mean())
            / (v.rolling(60, min_periods=10).std() + 1e-9)
        ).shift(1)
        df["vol_pctrank"] = v.rolling(252, min_periods=20).rank(pct=True).shift(1)
    else:
        df["vol_norm"]    = 0.0
        df["vol_pctrank"] = 0.5
    return df


# ─── Step 8: targets ─────────────────────────────────────────────────────────

def _create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_col = _find_col(df, ["High","high"])
    low_col  = _find_col(df, ["Low","low"])
    if high_col is None or low_col is None:
        raise ValueError("High and/or Low columns not found.")
    df["target_high_tplus1"]  = df[high_col].shift(-1)
    df["target_low_tplus1"]   = df[low_col].shift(-1)
    df["target_high_tplus10"] = df[high_col].shift(-10)
    df["target_low_tplus10"]  = df[low_col].shift(-10)
    return df


# ─── Step 9: final cleanup ───────────────────────────────────────────────────

_TARGETS = [
    "target_high_tplus1","target_low_tplus1",
    "target_high_tplus10","target_low_tplus10",
]

def _final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop any remaining non-numeric columns except Date and asset.
    # This includes fg_classification and any other string columns that
    # survived (they have already been one-hot encoded above).
    non_numeric = [
        c for c in df.columns
        if c not in {"Date", "asset"}
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        df = df.drop(columns=non_numeric)

    num_cols = [c for c in df.columns if c not in {"Date", "asset"}]
    if num_cols:
        df[num_cols] = df[num_cols].ffill().bfill()
        medians = df[num_cols].median(numeric_only=True)
        df[num_cols] = df[num_cols].fillna(medians)

    df = df.dropna(subset=_TARGETS).reset_index(drop=True)
    return df


# ─── Public entry point ───────────────────────────────────────────────────────

def prepare_asset_dataframe(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
    logger.info("-" * 70)
    logger.info(f"Preprocessing: {asset_name}  |  initial shape {df.shape}")
    df = _standardize_date_column(df)
    df = _ensure_asset_column(df, asset_name)
    df = _clean_numeric_columns(df)
    df = _basic_cleaning(df)
    logger.info(f"  [{asset_name}] after cleaning: {df.shape}")
    df = _create_price_features(df)
    logger.info(f"  [{asset_name}] after price features: {df.shape}")
    df = _create_behavioral_and_volatility_features(df)
    logger.info(f"  [{asset_name}] after behavioural features: {df.shape}")
    df = _add_volatility_modulation(df, asset_name)
    df = _create_targets(df)          # forward-looking — MUST come last
    df = _final_cleanup(df)
    logger.info(f"  [{asset_name}] final shape: {df.shape}")
    logger.info("-" * 70)
    return df