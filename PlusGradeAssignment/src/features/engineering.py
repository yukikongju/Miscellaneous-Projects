"""
Feature engineering for the offer allocation pipeline.

Usage:
    from features.engineering import build_features, fit_imputer, apply_imputer
    df = build_features(raw_df)
    medians = fit_imputer(train_df, COLS_TO_IMPUTE)
    df = apply_imputer(df, medians)

Caveats:
    - fit_imputer / fit_scaler must be called on training data only; apply_*
      variants are used on val/test to prevent data leakage.
    - Sentinel values (9999, -1) are replaced with NaN before imputation;
      build_features must run before fit_imputer.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import SENTINEL_9999, SENTINEL_NEG1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Sentinel flags ---
    df["miss_DAYS_SINCE_LAST_PURCHASE_L12M"] = (
        df["DAYS_SINCE_LAST_PURCHASE_L12M"] == SENTINEL_9999
    ).astype(int)
    df["miss_DAYS_SINCE_LAST_VISIT_NO_PURCHASE"] = (
        df["DAYS_SINCE_LAST_VISIT_NO_PURCHASE"] == SENTINEL_9999
    ).astype(int)
    df["BALANCE_IS_ZERO"] = (df["CURRENT_BALANCE"] == 0).astype(int)
    df["TRANX_L12M_ZERO"] = (df["COUNT_TRANX_L12M"] == 0).astype(int)

    # --- Log transforms ---
    df["BALANCE_LOG"] = np.log1p(df["CURRENT_BALANCE"])
    df["TRANX_L12M_LOG"] = np.log1p(df["COUNT_TRANX_L12M"])
    df["POINTS_PURCHASED_LAST_TRANX_L12M_LOG"] = np.log1p(
        df["POINTS_PURCHASED_LAST_TRANX_L12M"].clip(lower=0)
    )

    # --- Replace sentinels with NaN for imputation ---
    df["DAYS_SINCE_LAST_PURCHASE_L12M_imputed"] = df["DAYS_SINCE_LAST_PURCHASE_L12M"].replace(
        SENTINEL_9999, np.nan
    )
    df["DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed"] = df[
        "DAYS_SINCE_LAST_VISIT_NO_PURCHASE"
    ].replace(SENTINEL_9999, np.nan)
    df["LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed"] = df[
        "LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M"
    ].replace(SENTINEL_NEG1, np.nan)

    # --- Temporal ---
    df["HOUR_OF_DAY"] = df["SESSION_DATE"].dt.hour
    df["DAY_OF_WEEK"] = df["SESSION_DATE"].dt.dayofweek

    # --- Offer vs last purchase (set to 0 when no L12M purchase history) ---
    df["OFFER_VS_LAST_PURCHASE"] = np.where(
        df["miss_DAYS_SINCE_LAST_PURCHASE_L12M"] == 0,
        df["OFFER_RICHNESS_SERVED"] - df["LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed"],
        0,
    )

    return df


def fit_imputer(train_df: pd.DataFrame, cols: list) -> dict:
    """Compute medians on training set only."""
    return {col: train_df[col].median() for col in cols}


def apply_imputer(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    return df.fillna(medians)


COLS_TO_IMPUTE = [
    "DAYS_SINCE_LAST_PURCHASE_L12M_imputed",
    "DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed",
    "LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed",
]

COLS_TO_SCALE = [
    "BALANCE_LOG",
    "TRANX_L12M_LOG",
    "POINTS_PURCHASED_LAST_TRANX_L12M_LOG",
    "DAYS_SINCE_LAST_PURCHASE_L12M_imputed",
    "DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed",
    "HOUR_OF_DAY",
]


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[COLS_TO_SCALE])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[COLS_TO_SCALE] = scaler.transform(df[COLS_TO_SCALE])
    return df
