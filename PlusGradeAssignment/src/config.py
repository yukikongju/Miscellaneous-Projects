"""
Pipeline-wide constants and hyperparameters for offer allocation.

Usage:
    from config import FEATURES_MODEL1, OFFERS, PRICE_FLOOR

Caveats:
    - DATA_PATH is resolved relative to this file's location; move carefully.
    - SENTINEL_9999 and SENTINEL_NEG1 represent "missing" in the raw data —
      they must be handled in feature engineering before modeling.
"""

from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "assignments" / "DATA_OFFER_ALLOCATION.csv"
RANDOM_SEED = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10

OFFERS = [0.40, 0.45, 0.50]
MIN_POINTS = 3000
PRICE_FLOOR = 0.016
OFFER_COVERAGE_FLOOR = {"0.40": 0.05, "0.45": 0.10, "0.50": 0.15}

SENTINEL_9999 = 9999
SENTINEL_NEG1 = -1

FEATURES_MODEL1 = [
    "FLAG_FIRST_TIME_VISITOR",
    "FLAG_FIRST_TIME_BUYER",
    "BALANCE_IS_ZERO",
    "BALANCE_LOG",
    "TRANX_L12M_ZERO",
    "TRANX_L12M_LOG",
    "miss_DAYS_SINCE_LAST_PURCHASE_L12M",
    "DAYS_SINCE_LAST_PURCHASE_L12M_imputed",
    "miss_DAYS_SINCE_LAST_VISIT_NO_PURCHASE",
    "DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed",
    "POINTS_PURCHASED_LAST_TRANX_L12M_LOG",
    "LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed",
    "HOUR_OF_DAY",
    # "DAY_OF_WEEK",
    "OFFER_RICHNESS_SERVED",
    "OFFER_VS_LAST_PURCHASE",
]
FEATURES_MODEL2 = FEATURES_MODEL1  # same for now, adjust after feature selection
TARGET_M1 = "FLAG_TRANSACTION"
TARGET_M2 = "POINTS_PURCHASED"
