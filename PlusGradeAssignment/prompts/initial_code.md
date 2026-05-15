---- PROMPT ----

Context: EDA has been completed in `playground/auto_EDA.ipynb` and `playground/EDA.ipynb`. The next steps implement the full modeling pipeline across 4 stages. The code should live in `.py` files.


### Step 1 — Train / Validation / Test Split

Split the dataset with the following constraints:

- **No member leakage**: a single `MEMBER_KEY` must appear in exactly one of train, validation, or test. Do not split by session row — split by unique member first, then assign all their sessions to the corresponding fold.
- **Stratification**: stratify the member-level split on two groups — members with more than one session, and members with exactly one session — to preserve their proportions across folds.
- **Split ratio**: 80% train / 10% validation / 10% test at the member level.
- After splitting, verify: `assert len(set(train_members) & set(test_members)) == 0`
- Drop `SESSION_KEY`, `SESSION_DATE`, and `MEMBER_KEY` from all feature matrices after the split is complete. These are identifiers, not features.

### Step 2 — Feature Engineering

Create the following features in order. Document each one with a one-line comment explaining its purpose.

**Binary / flag features:**
- `FLAG_FIRST_TIME_VISITOR` — already in data, pass through
- `FLAG_FIRST_TIME_BUYER` — already in data, pass through
- `BALANCE_IS_ZERO` = 1 if `CURRENT_BALANCE == 0`, else 0
- `TRANX_L12M_ZERO` = 1 if `COUNT_TRANX_L12M == 0`, else 0
- `miss_DAYS_SINCE_LAST_PURCHASE_L12M` = 1 if `DAYS_SINCE_LAST_PURCHASE_L12M == 9999`, else 0 (sentinel = no L12M purchase)
- `miss_DAYS_SINCE_LAST_VISIT_NO_PURCHASE` = 1 if `DAYS_SINCE_LAST_VISIT_NO_PURCHASE == 9999`, else 0

**Continuous features (log-transformed):**
- `BALANCE_LOG` = `log1p(CURRENT_BALANCE)`
- `TRANX_L12M_LOG` = `log1p(COUNT_TRANX_L12M)`
- `POINTS_PURCHASED_LAST_TRANX_L12M_LOG` = `log1p(POINTS_PURCHASED_LAST_TRANX_L12M)`

**Imputed continuous features** (impute using the median computed on the training set only — fit on train, transform on val/test to prevent leakage):
- `DAYS_SINCE_LAST_PURCHASE_L12M_imputed`: replace 9999 sentinel with `NaN`, then impute with training median
- `DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed`: replace 9999 sentinel with `NaN`, then impute with training median
- `LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed`: replace -1 sentinel with `NaN`, then impute with training median

**Temporal features:**
- `HOUR_OF_DAY` — extracted from `SESSION_DATE`
- `DAY_OF_WEEK` — extracted from `SESSION_DATE` (0=Monday … 6=Sunday)

**Offer and interaction features:**
- `OFFER_RICHNESS_SERVED` — keep as numeric (0.40, 0.45, 0.50). This is the arm feature; it must be present in every model.
- `OFFER_VS_LAST_PURCHASE` = `OFFER_RICHNESS_SERVED - LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed`. Captures whether the current offer is richer or poorer than what the member last responded to. Set to 0 where `miss_DAYS_SINCE_LAST_PURCHASE_L12M == 1`.

**Scaling:**
- Apply `StandardScaler` to continuous features **only for linear models** (Logistic Regression, Linear Regression, Ridge, Lasso). Fit the scaler on the training set only; transform val and test.
- Tree-based models (Random Forest, XGBoost, LightGBM) do **not** need scaling — pass raw engineered features directly.

**Final feature list for both models:**
```
FLAG_FIRST_TIME_VISITOR, FLAG_FIRST_TIME_BUYER,
BALANCE_IS_ZERO, BALANCE_LOG,
TRANX_L12M_ZERO, TRANX_L12M_LOG,
miss_DAYS_SINCE_LAST_PURCHASE_L12M, DAYS_SINCE_LAST_PURCHASE_L12M_imputed,
miss_DAYS_SINCE_LAST_VISIT_NO_PURCHASE, DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed,
POINTS_PURCHASED_LAST_TRANX_L12M_LOG,
LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed,
HOUR_OF_DAY, DAY_OF_WEEK,
OFFER_RICHNESS_SERVED, OFFER_VS_LAST_PURCHASE
```

### Step 3 — Modeling

#### Model 1: Conversion Probability

- **Target**: `FLAG_TRANSACTION` (binary: 0 or 1)
- **Use** `class_weight='balanced'` for all models that support it (Logistic Regression, Random Forest). For XGBoost/LightGBM use `scale_pos_weight = n_negative / n_positive` computed on the training set.
- **Evaluation metric**: AUC-ROC (primary), calibration curve (required — conversion probabilities feed directly into revenue estimation and must be well-calibrated). Also report AUC separately for `FLAG_FIRST_TIME_BUYER=0` and `FLAG_FIRST_TIME_BUYER=1` subgroups.

**Models to train:**
- Logistic Regression (scaled features)
- Naive Bayes (GaussianNB)
- Random Forest
- XGBoost
- LightGBM

#### Model 2: Expected Points Purchased

- **Train only on transacting sessions** (`FLAG_TRANSACTION == 1`) — do not include zero-purchase rows
- **Target**: `log(POINTS_PURCHASED)` — right-skewed, log transformation required. Back-transform predictions with `np.exp()` and clip at 3000 (business minimum)
- **Evaluation metrics**: RMSE on log scale (primary), RMSE on original scale, predicted vs. actual scatter plot

**Models to train:**
- Linear Regression (scaled features)
- Ridge (scaled features)
- Lasso (scaled features)
- Random Forest
- XGBoost

#### Hyperparameter Tuning (both models)

- Use **Bayesian Optimization** (`scikit-optimize` `BayesSearchCV` or `bayesian-optimization` library) — not grid search
- Use **early stopping on the validation fold** for XGBoost and LightGBM (pass `eval_set=[(X_val, y_val)]`)
- Track all experiments with **MLflow**: log model name, hyperparameters, AUC/RMSE on validation set, feature importance, and the fitted model artifact. Use a single MLflow experiment named `offer_allocation_model1` and `offer_allocation_model2` respectively.

#### Feature Selection

Run after hyperparameter tuning, on the best-performing model per model type:
- **Recursive Feature Elimination (RFE)** — uses `coef_` or `feature_importances_`, recalculates importance after each removal. Preferred for correlated features.
- **Backward Elimination** — uses p-values (for linear models only). Compare retained feature sets.
- Report which features were dropped and retrain the final model on the reduced feature set. Verify AUC/RMSE does not degrade by more than 0.01.

### Step 4 — Offer Allocation Strategy

#### A. Greedy Allocation

For each session in the test set, score all three offers by substituting `OFFER_RICHNESS_SERVED` and recomputing `OFFER_VS_LAST_PURCHASE` accordingly. All other features remain fixed.

```python
expected_revenue = {}

for offer in [0.40, 0.45, 0.50]:
    test_with_offer = test.copy()
    test_with_offer['OFFER_RICHNESS_SERVED'] = offer
    test_with_offer['OFFER_VS_LAST_PURCHASE'] = (
        offer - test_with_offer['LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed']
    ).where(test_with_offer['miss_DAYS_SINCE_LAST_PURCHASE_L12M'] == 0, other=0)

    p_convert = best_model1.predict_proba(test_with_offer[features])[:, 1]
    expected_points = np.exp(
        best_model2.predict(test_with_offer[features_model2])
    ).clip(min=3000)
    price = 0.03 * (1 - offer)

    expected_revenue[offer] = p_convert * expected_points * price

assigned_offer = pd.DataFrame(expected_revenue).idxmax(axis=1)
```

#### B. Constraint Verification and Repair

**Constraint 1 — Price floor (avg price/point ≥ 0.016 across transacting sessions):**

```python
# Compute implied weighted average price per point
p_dict = {o: best_model1.predict_proba(score_with_offer(test, o))[:, 1] for o in [0.40, 0.45, 0.50]}
pts_dict = {o: np.exp(best_model2.predict(score_with_offer(test, o))).clip(min=3000) for o in [0.40, 0.45, 0.50]}

assigned_p = np.array([p_dict[o][i] for i, o in enumerate(assigned_offer)])
assigned_pts = np.array([pts_dict[o][i] for i, o in enumerate(assigned_offer)])
assigned_price = np.array([0.03 * (1 - o) for o in assigned_offer])

weighted_avg_price = (
    (assigned_p * assigned_pts * assigned_price).sum() /
    (assigned_p * assigned_pts).sum()
)

# Repair pass if constraint is violated
if weighted_avg_price < 0.016:
    sessions_at_50 = np.where(assigned_offer == 0.50)[0]
    revenue_delta = expected_revenue[0.50] - expected_revenue[0.45]
    # Flip the sessions where downgrading from 50% to 45% costs the least
    flip_order = sessions_at_50[np.argsort(revenue_delta[sessions_at_50])]
    for idx in flip_order:
        assigned_offer[idx] = 0.45
        # Recompute weighted_avg_price and stop when constraint is met
        ...
```

**Constraint 2 — Offer coverage (minimum allocation per tier):**

Define floors as: at least **5% of test sessions** must receive the 40% offer, at least **10%** must receive the 45% offer. If either floor is not met after the greedy + repair pass, randomly sample the deficit from sessions where that tier has the smallest revenue gap vs. the assigned offer, and reassign.

Verify and report final offer distribution.

#### C. Comparison Table

Report the following for: (1) Always-50% baseline, (2) Uniform random baseline, (3) Historical observed distribution, (4) Model-driven unconstrained, (5) Model-driven constrained:

| Strategy | E[Rev/session] | Avg Price/Point | 40% share | 45% share | 50% share | Price floor ✓ | Coverage ✓ |
|----------|----------------|-----------------|-----------|-----------|-----------|---------------|------------|


## Recommended File Structure

```
offer_allocation/
├── config.py
├── data/
│   └── loader.py
├── features/
│   └── engineering.py
├── models/
|   ├── registry.py    — register() decorator + MODEL_REGISTRY
|   ├── estimators.py  — make_*() functions per algorithm
|   ├── models.py      — BaseModel, BaseClassifier, BaseRegressor, all @register classes, build_model()
|   ├── tuning.py      — SEARCH_SPACES + tune()
|   ├── selection.py   — rfe_selection(), importance_selection()
|   └── evaluate.py    — evaluate_classifier(), evaluate_regressor()
├── allocation/
│   └── strategy.py
├── utils/
│   └── mlflow_utils.py
└── main.py
```


## `config.py`
```python
from pathlib import Path

DATA_PATH = Path("data/DATA_OFFER_ALLOCATION.csv")
RANDOM_SEED = 42
TEST_SIZE = 0.10
VAL_SIZE  = 0.10

OFFERS = [0.40, 0.45, 0.50]
MIN_POINTS = 3000
PRICE_FLOOR = 0.016
OFFER_COVERAGE_FLOOR = {"0.40": 0.05, "0.45": 0.10}

SENTINEL_9999 = 9999
SENTINEL_NEG1 = -1

FEATURES_MODEL1 = [
    "FLAG_FIRST_TIME_VISITOR", "FLAG_FIRST_TIME_BUYER",
    "BALANCE_IS_ZERO", "BALANCE_LOG",
    "TRANX_L12M_ZERO", "TRANX_L12M_LOG",
    "miss_DAYS_SINCE_LAST_PURCHASE_L12M",
    "DAYS_SINCE_LAST_PURCHASE_L12M_imputed",
    "miss_DAYS_SINCE_LAST_VISIT_NO_PURCHASE",
    "DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed",
    "POINTS_PURCHASED_LAST_TRANX_L12M_LOG",
    "LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed",
    "HOUR_OF_DAY", "DAY_OF_WEEK",
    "OFFER_RICHNESS_SERVED", "OFFER_VS_LAST_PURCHASE",
]
FEATURES_MODEL2 = FEATURES_MODEL1  # same for now, adjust after feature selection
TARGET_M1 = "FLAG_TRANSACTION"
TARGET_M2 = "POINTS_PURCHASED"
```


## `data/loader.py`
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import DATA_PATH, RANDOM_SEED, TEST_SIZE, VAL_SIZE


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["SESSION_DATE"])
    return df


def split_by_member(df: pd.DataFrame):
    """
    Member-level split: no MEMBER_KEY appears in more than one fold.
    Stratified by whether a member has single vs. multiple sessions.
    Returns train_df, val_df, test_df.
    """
    member_sessions = df.groupby("MEMBER_KEY")["SESSION_KEY"].count().reset_index()
    member_sessions.columns = ["MEMBER_KEY", "n_sessions"]
    member_sessions["stratum"] = (member_sessions["n_sessions"] > 1).astype(int)

    members = member_sessions["MEMBER_KEY"].values
    strata  = member_sessions["stratum"].values

    # First cut: train vs temp (val + test)
    train_members, temp_members = train_test_split(
        members,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=strata,
        random_state=RANDOM_SEED,
    )

    # Second cut: val vs test from temp
    temp_strata = member_sessions.set_index("MEMBER_KEY").loc[temp_members, "stratum"]
    val_members, test_members = train_test_split(
        temp_members,
        test_size=TEST_SIZE / (TEST_SIZE + VAL_SIZE),
        stratify=temp_strata,
        random_state=RANDOM_SEED,
    )

    train_df = df[df["MEMBER_KEY"].isin(train_members)].copy()
    val_df   = df[df["MEMBER_KEY"].isin(val_members)].copy()
    test_df  = df[df["MEMBER_KEY"].isin(test_members)].copy()

    # Sanity check
    assert len(set(train_members) & set(test_members)) == 0
    assert len(set(train_members) & set(val_members))  == 0

    return train_df, val_df, test_df
```

## `features/engineering.py`
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import SENTINEL_9999, SENTINEL_NEG1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Sentinel flags ---
    df["miss_DAYS_SINCE_LAST_PURCHASE_L12M"]      = (df["DAYS_SINCE_LAST_PURCHASE_L12M"] == SENTINEL_9999).astype(int)
    df["miss_DAYS_SINCE_LAST_VISIT_NO_PURCHASE"]  = (df["DAYS_SINCE_LAST_VISIT_NO_PURCHASE"] == SENTINEL_9999).astype(int)
    df["BALANCE_IS_ZERO"]                         = (df["CURRENT_BALANCE"] == 0).astype(int)
    df["TRANX_L12M_ZERO"]                         = (df["COUNT_TRANX_L12M"] == 0).astype(int)

    # --- Log transforms ---
    df["BALANCE_LOG"]                            = np.log1p(df["CURRENT_BALANCE"])
    df["TRANX_L12M_LOG"]                         = np.log1p(df["COUNT_TRANX_L12M"])
    df["POINTS_PURCHASED_LAST_TRANX_L12M_LOG"]   = np.log1p(df["POINTS_PURCHASED_LAST_TRANX_L12M"])

    # --- Replace sentinels with NaN for imputation ---
    df["DAYS_SINCE_LAST_PURCHASE_L12M_imputed"]             = df["DAYS_SINCE_LAST_PURCHASE_L12M"].replace(SENTINEL_9999, np.nan)
    df["DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed"]         = df["DAYS_SINCE_LAST_VISIT_NO_PURCHASE"].replace(SENTINEL_9999, np.nan)
    df["LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed"] = df["LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M"].replace(SENTINEL_NEG1, np.nan)

    # --- Temporal ---
    df["HOUR_OF_DAY"] = df["SESSION_DATE"].dt.hour
    df["DAY_OF_WEEK"] = df["SESSION_DATE"].dt.dayofweek

    # --- Offer interaction ---
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
    "BALANCE_LOG", "TRANX_L12M_LOG",
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
```



## models/registry.py
```python

from typing import Any

MODEL_REGISTRY: dict[str, type] = {}

def register(registry: dict, name: str):
    """Decorator that registers a class under a given name."""
    def decorator(cls):
        registry[name] = cls
        return cls
    return decorator
```


## models/estimators.py
```python

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from config import RANDOM_SEED


def make_logistic_regression(cfg: dict):
    return LogisticRegression(
        C=            cfg.get("C",            1.0),
        max_iter=     cfg.get("max_iter",     1000),
        class_weight= cfg.get("class_weight", "balanced"),
        random_state= RANDOM_SEED,
    )

def make_naive_bayes(cfg: dict):
    return GaussianNB(var_smoothing=cfg.get("var_smoothing", 1e-9))

def make_random_forest_classifier(cfg: dict):
    return RandomForestClassifier(
        n_estimators= cfg.get("n_estimators", 100),
        max_depth=    cfg.get("max_depth",    None),
        class_weight= cfg.get("class_weight", "balanced"),
        n_jobs=-1,
        random_state= RANDOM_SEED,
    )

def make_random_forest_regressor(cfg: dict):
    return RandomForestRegressor(
        n_estimators= cfg.get("n_estimators", 100),
        max_depth=    cfg.get("max_depth",    None),
        n_jobs=-1,
        random_state= RANDOM_SEED,
    )

def make_lgbm_classifier(cfg: dict):
    return LGBMClassifier(
        scale_pos_weight= cfg.get("scale_pos_weight", 1.0),
        num_leaves=       cfg.get("num_leaves",       31),
        learning_rate=    cfg.get("learning_rate",    0.05),
        n_estimators=     cfg.get("n_estimators",     300),
        n_jobs=-1,
        random_state=     RANDOM_SEED,
        verbose=-1,
    )

def make_lgbm_regressor(cfg: dict):
    return LGBMRegressor(
        num_leaves=    cfg.get("num_leaves",    31),
        learning_rate= cfg.get("learning_rate", 0.05),
        n_estimators=  cfg.get("n_estimators",  300),
        n_jobs=-1,
        random_state=  RANDOM_SEED,
        verbose=-1,
    )

def make_xgb_classifier(cfg: dict):
    return XGBClassifier(
        scale_pos_weight= cfg.get("scale_pos_weight", 1.0),
        max_depth=        cfg.get("max_depth",        6),
        learning_rate=    cfg.get("learning_rate",    0.05),
        n_estimators=     cfg.get("n_estimators",     300),
        eval_metric=      "auc",
        random_state=     RANDOM_SEED,
        verbosity=0,
    )

def make_xgb_regressor(cfg: dict):
    return XGBRegressor(
        max_depth=     cfg.get("max_depth",     6),
        learning_rate= cfg.get("learning_rate", 0.05),
        n_estimators=  cfg.get("n_estimators",  300),
        eval_metric=   "rmse",
        random_state=  RANDOM_SEED,
        verbosity=0,
    )

def make_linear_regression(cfg: dict): return LinearRegression()
def make_ridge(cfg: dict):  return Ridge(alpha=cfg.get("alpha", 1.0))
def make_lasso(cfg: dict):  return Lasso(alpha=cfg.get("alpha", 1.0))
```


## models/models.py
```python
# All model classes in one file — one registry, classifier + regressor tasks.

from __future__ import annotations
import numpy as np
from typing import Any

from models.registry import register, MODEL_REGISTRY
from models.estimators import (
    make_logistic_regression, make_naive_bayes,
    make_random_forest_classifier, make_random_forest_regressor,
    make_lgbm_classifier, make_lgbm_regressor,
    make_xgb_classifier, make_xgb_regressor,
    make_linear_regression, make_ridge, make_lasso,
)
from config import RANDOM_SEED, MIN_POINTS


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseModel:
    def __init__(self, cfg: dict[str, Any]):
        self.name      = cfg["name"]
        self.cfg       = cfg
        self.estimator = self._build_estimator(cfg)

    def _build_estimator(self, cfg): raise NotImplementedError
    def fit(self, X, y, eval_set=None): raise NotImplementedError

    def feature_importances(self):
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        if hasattr(self.estimator, "coef_"):
            return self.estimator.coef_[0]
        return None


class BaseClassifier(BaseModel):
    def fit(self, X, y, eval_set=None):
        kwargs = {"eval_set": eval_set} if eval_set and hasattr(self.estimator, "n_estimators") else {}
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


class BaseRegressor(BaseModel):
    def fit(self, X, y, eval_set=None):
        kwargs = {"eval_set": eval_set} if eval_set and hasattr(self.estimator, "n_estimators") else {}
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Back-transform from log scale and clip at MIN_POINTS."""
        return np.exp(self.estimator.predict(X)).clip(min=MIN_POINTS)

    def predict_log(self, X):
        return self.estimator.predict(X)


# ── Classifiers ───────────────────────────────────────────────────────────────

@register("logistic_regression")
class LogisticRegressionModel(BaseClassifier):
    def _build_estimator(self, cfg): return make_logistic_regression(cfg)

@register("naive_bayes")
class NaiveBayesModel(BaseClassifier):
    def _build_estimator(self, cfg): return make_naive_bayes(cfg)

@register("random_forest_classifier")
class RandomForestClassifierModel(BaseClassifier):
    def _build_estimator(self, cfg): return make_random_forest_classifier(cfg)

@register("lgbm_classifier")
class LGBMClassifierModel(BaseClassifier):
    def _build_estimator(self, cfg): return make_lgbm_classifier(cfg)

@register("xgb_classifier")
class XGBClassifierModel(BaseClassifier):
    def _build_estimator(self, cfg): return make_xgb_classifier(cfg)


# ── Regressors ────────────────────────────────────────────────────────────────

@register("linear_regression")
class LinearRegressionModel(BaseRegressor):
    def _build_estimator(self, cfg): return make_linear_regression(cfg)

@register("ridge")
class RidgeModel(BaseRegressor):
    def _build_estimator(self, cfg): return make_ridge(cfg)

@register("lasso")
class LassoModel(BaseRegressor):
    def _build_estimator(self, cfg): return make_lasso(cfg)

@register("random_forest_regressor")
class RandomForestRegressorModel(BaseRegressor):
    def _build_estimator(self, cfg): return make_random_forest_regressor(cfg)

@register("lgbm_regressor")
class LGBMRegressorModel(BaseRegressor):
    def _build_estimator(self, cfg): return make_lgbm_regressor(cfg)

@register("xgb_regressor")
class XGBRegressorModel(BaseRegressor):
    def _build_estimator(self, cfg): return make_xgb_regressor(cfg)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(cfg: dict[str, Any]) -> BaseModel:
    name = cfg.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"'{name}' not found. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](cfg)
```

## models/tuning.py
```python

from __future__ import annotations
from typing import Any, Callable

import numpy as np
import mlflow
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import roc_auc_score, mean_squared_error

from models.models import build_model, BaseClassifier, BaseRegressor


# ── Search spaces — one dict per model name ───────────────────────────────────

SEARCH_SPACES: dict[str, dict] = {
    "logistic_regression":      {"C":             Real(1e-3, 10.0, prior="log-uniform")},
    "random_forest_classifier": {"n_estimators":  Integer(50, 300),
                                 "max_depth":      Integer(3, 15)},
    "lgbm_classifier":          {"num_leaves":     Integer(15, 127),
                                 "learning_rate":  Real(0.01, 0.3, prior="log-uniform"),
                                 "n_estimators":   Integer(100, 500)},
    "xgb_classifier":           {"max_depth":      Integer(3, 10),
                                 "learning_rate":  Real(0.01, 0.3, prior="log-uniform"),
                                 "n_estimators":   Integer(100, 500)},
    "ridge":                    {"alpha":          Real(1e-3, 100.0, prior="log-uniform")},
    "lasso":                    {"alpha":          Real(1e-4, 10.0,  prior="log-uniform")},
    "random_forest_regressor":  {"n_estimators":   Integer(50, 300),
                                 "max_depth":       Integer(3, 15)},
    "lgbm_regressor":           {"num_leaves":      Integer(15, 127),
                                 "learning_rate":   Real(0.01, 0.3, prior="log-uniform"),
                                 "n_estimators":    Integer(100, 500)},
    "xgb_regressor":            {"max_depth":       Integer(3, 10),
                                 "learning_rate":   Real(0.01, 0.3, prior="log-uniform"),
                                 "n_estimators":    Integer(100, 500)},
}


def _make_predefined_split(X_train, X_val):
    """
    PredefinedSplit tells BayesSearchCV to always validate on X_val,
    never re-split — correct since our split is member-level.
    """
    test_fold = np.concatenate([
        np.full(len(X_train), -1),   # -1 = always in train
        np.zeros(len(X_val)),         #  0 = always in validation
    ])
    return PredefinedSplit(test_fold)


def tune(
    cfg:        dict[str, Any],
    X_train,    y_train,
    X_val,      y_val,
    n_iter:     int = 30,
    task:       str = "classification",   # "classification" | "regression"
) -> dict[str, Any]:
    """
    Run Bayesian search for cfg["name"], return best params dict.
    Logs every trial to MLflow.
    """
    name  = cfg["name"]
    space = SEARCH_SPACES.get(name, {})

    if not space:
        print(f"No search space for '{name}' — skipping tuning.")
        return {}

    # Combine train + val for BayesSearchCV with PredefinedSplit
    import pandas as pd
    X_combined = pd.concat([X_train, X_val]) if hasattr(X_train, "iloc") else np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    cv         = _make_predefined_split(X_train, X_val)

    # Wrap estimator so BayesSearchCV can call fit/score normally
    base_model  = build_model(cfg)
    scoring     = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"

    searcher = BayesSearchCV(
        estimator=  base_model.estimator,
        search_spaces= space,
        n_iter=     n_iter,
        cv=         cv,
        scoring=    scoring,
        refit=      False,       # we refit manually with best params below
        random_state= 42,
        n_jobs=     -1,
    )
    searcher.fit(X_combined, y_combined)
    best_params = searcher.best_params_

    # Log to MLflow
    with mlflow.start_run(run_name=f"tuning_{name}", nested=True):
        mlflow.log_params({"model": name, **best_params})
        mlflow.log_metric("best_val_score", searcher.best_score_)

    return dict(best_params)
```

## models/selection.py
```python

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression

from models.models import BaseModel


def rfe_selection(
    model:      BaseModel,
    X_train:    pd.DataFrame,
    y_train:    np.ndarray,
    n_features: int,
) -> list[str]:
    """
    Recursive Feature Elimination using the model's underlying estimator.
    Returns the list of selected feature names.
    """
    selector = RFE(
        estimator=  model.estimator,
        n_features_to_select= n_features,
        step=1,
    )
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    print(f"RFE selected {len(selected)} features: {selected}")
    return selected


def importance_selection(
    model:      BaseModel,
    X_train:    pd.DataFrame,
    threshold:  float = 0.01,
) -> list[str]:
    """
    Drop features whose normalised importance is below threshold.
    Works for any model exposing feature_importances_ or coef_.
    """
    importances = model.feature_importances()
    if importances is None:
        raise ValueError(f"Model '{model.name}' does not expose feature importances.")

    norm   = np.abs(importances) / np.abs(importances).sum()
    mask   = norm >= threshold
    selected = X_train.columns[mask].tolist()
    dropped  = X_train.columns[~mask].tolist()
    print(f"Importance selection — kept {len(selected)}, dropped {dropped}")
    return selected
```


## models/evaluate.py  (unchanged — works for both tasks)
```python

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_classifier(model, X, y, label: str = "") -> dict:
    preds = model.predict_proba(X)[:, 1]
    return {"label": label, "auc": round(roc_auc_score(y, preds), 4)}


def evaluate_classifier_by_segment(model, X, y, segment: pd.Series, label: str = "") -> dict:
    results = {}
    for val in [0, 1]:
        mask = segment == val
        auc  = roc_auc_score(y[mask], model.predict_proba(X[mask])[:, 1])
        results[f"{label}_seg{val}_auc"] = round(auc, 4)
    return results


def evaluate_regressor(model, X, y_log: np.ndarray, label: str = "") -> dict:
    preds_log  = model.predict_log(X)
    preds_orig = model.predict(X)
    y_orig     = np.exp(y_log)
    return {
        "label":     label,
        "rmse_log":  round(np.sqrt(((preds_log  - y_log)  ** 2).mean()), 4),
        "rmse_orig": round(np.sqrt(((preds_orig - y_orig) ** 2).mean()), 2),
    }
```

---

## Usage in `models/train.py`

```python
from models.models import build_model
from models.tuning import tune
from models.selection import rfe_selection, importance_selection
from models.evaluate import evaluate_classifier, evaluate_regressor
import mlflow

mlflow.set_experiment("offer_allocation")

spw = compute_scale_pos_weight(train_df[TARGET_M1])

CONVERSION_CONFIGS = [
    {"name": "logistic_regression", "C": 1.0},
    {"name": "naive_bayes"},
    {"name": "random_forest_classifier"},
    {"name": "lgbm_classifier",  "scale_pos_weight": spw},
    {"name": "xgb_classifier",   "scale_pos_weight": spw},
]

POINTS_CONFIGS = [
    {"name": "linear_regression"},
    {"name": "ridge"},
    {"name": "lasso"},
    {"name": "random_forest_regressor"},
    {"name": "lgbm_regressor"},
    {"name": "xgb_regressor"},
]

with mlflow.start_run(run_name="conversion_models"):
    for cfg in CONVERSION_CONFIGS:
        # 1. Tune
        best_params = tune(cfg, X_train, y_train, X_val, y_val, task="classification")
        cfg.update(best_params)

        # 2. Train on full train set with best params
        model = build_model(cfg)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # 3. Feature selection
        selected = importance_selection(model, X_train, threshold=0.01)
        model_fs = build_model(cfg)
        model_fs.fit(X_train[selected], y_train, eval_set=[(X_val[selected], y_val)])

        # 4. Evaluate
        results = evaluate_classifier(model_fs, X_val[selected], y_val, label=cfg["name"])
        mlflow.log_metrics({cfg["name"]: results["auc"]})
        print(results)
```


## `allocation/strategy.py`
```python
import numpy as np
import pandas as pd
from config import OFFERS, MIN_POINTS, PRICE_FLOOR, OFFER_COVERAGE_FLOOR, FEATURES_MODEL1, FEATURES_MODEL2


def score_offers(test_df, model1, model2, feature_cols_m1, feature_cols_m2):
    """Score each session under each offer. Returns dict of arrays."""
    p_convert, exp_points, exp_revenue = {}, {}, {}

    for offer in OFFERS:
        tmp = test_df.copy()
        tmp["OFFER_RICHNESS_SERVED"] = offer
        tmp["OFFER_VS_LAST_PURCHASE"] = np.where(
            tmp["miss_DAYS_SINCE_LAST_PURCHASE_L12M"] == 0,
            offer - tmp["LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed"],
            0,
        )
        p_convert[offer]  = model1.predict_proba(tmp[feature_cols_m1])[:, 1]
        exp_points[offer] = np.exp(model2.predict(tmp[feature_cols_m2])).clip(min=MIN_POINTS)
        price             = 0.03 * (1 - offer)
        exp_revenue[offer] = p_convert[offer] * exp_points[offer] * price

    return p_convert, exp_points, exp_revenue


def greedy_allocate(exp_revenue: dict) -> np.ndarray:
    return pd.DataFrame(exp_revenue).idxmax(axis=1).values


def enforce_price_floor(assigned, p_convert, exp_points, exp_revenue):
    assigned = assigned.copy()

    def weighted_avg_price(assigned):
        p   = np.array([p_convert[o][i]  for i, o in enumerate(assigned)])
        pts = np.array([exp_points[o][i] for i, o in enumerate(assigned)])
        prc = np.array([0.03 * (1 - o)   for o in assigned])
        return (p * pts * prc).sum() / (p * pts).sum()

    if weighted_avg_price(assigned) >= PRICE_FLOOR:
        return assigned

    sessions_at_50 = np.where(assigned == 0.50)[0]
    revenue_delta  = exp_revenue[0.50] - exp_revenue[0.45]
    flip_order     = sessions_at_50[np.argsort(revenue_delta[sessions_at_50])]

    for idx in flip_order:
        assigned[idx] = 0.45
        if weighted_avg_price(assigned) >= PRICE_FLOOR:
            break

    return assigned


def enforce_coverage(assigned, exp_revenue):
    assigned = assigned.copy()
    n = len(assigned)

    for offer, floor_pct in OFFER_COVERAGE_FLOOR.items():
        offer = float(offer)
        current_pct = (assigned == offer).mean()
        if current_pct >= floor_pct:
            continue

        needed = int(np.ceil(floor_pct * n)) - (assigned == offer).sum()
        candidates = np.where(assigned != offer)[0]
        # Pick candidates where switching costs the least revenue
        next_best  = np.array([exp_revenue[offer][i] for i in candidates])
        current_rev = np.array([exp_revenue[assigned[i]][i] for i in candidates])
        delta = current_rev - next_best
        flip_idx = candidates[np.argsort(delta)[:needed]]
        assigned[flip_idx] = offer

    return assigned
```


## Key Principles Applied

**One responsibility per file** — `engineering.py` only transforms features, never touches models. `strategy.py` only does allocation, never retrains.

**Fit on train, transform everywhere** — imputer and scaler are fitted in `main.py` on `train_df` only, then passed as objects to val and test. This pattern makes leakage impossible by construction.

**`config.py` as single source of truth** — changing `PRICE_FLOOR` or `FEATURES_MODEL1` in one place propagates everywhere. No magic numbers scattered across files.


----- BROUILLON -----


Given the Exploratory Data Analysis (EDA) in `playground/auto_EDA.ipynb` and `playground/EDA.ipynb`, the next steps to code are the following:
1. Split Data into Training and Testing
2. Feature Engineering
3. Coding the Model (x2)
    a. Conversion Probability: Probability that a user convert (between 0 and 1)
    b. Expected Points: Number of points that a user will buy (regression)
4. Offer Allocation Srategy => Greedy, then verify constraints

**Step 1. Split the data**

The dataset should be split into training and testing set such that there is
no data leakage. Data should be split such that a single MEMBER_KEY is found in either
training or testing set, not both. Users should be stratified: MEMBER_KEY
who have had more than one session, and those with only one session.

Split should be 80% training, 10% validation, 10% testing.


**Step 2. Feature Engineering and Feature Selection**

- HOUR_OF_DAY
- FLAG_FIRST_TIME_BUYER
- FLAG_FIRST_TIME_VISITOR
- BALANCE_IS_ZERO
- BALANCE_LOG
- TRANX_L12M_LOG
- TRANX_L12M_ZERO
- miss_DAYS_SINCE_LAST_PURCHASE_L12M
- miss_DAYS_SINCE_LAST_VISIT_NO_PURCHASE
- DAYS_SINCE_LAST_PURCHASE_L12M_imputed
- DAYS_SINCE_LAST_VISIT_NO_PURCHASE_imputed
- POINTS_PURCHASED_LAST_TRANX_L12M_LOG
- LAST_OFFER_RICHNESS_SERVED_ON_PURCHASE_L12M_imputed
- OFFER_RICHNESS_SERVED

Is there anything missing or double

Notes:
- Scale all continuous features with StandardScaler
- class_weight='balanced'


**Step 3. Coding the Model**


Model 1: Conversion Probability: Probability that a user convert (between 0 and 1)
- Target: FLAG_TRANSACTION
- Models to test:
    * Logistic Regression
    * Naive Bayes
    * Random Forest
    * Gradient Boosting Machines: XGBoost, LightGBM


Model 2: Expected Points: Number of points that a user will buy (regression)
- Target: log(POINTS_PURCHASED) (becuase right skewed)
- Models to test:
    * Linear Models: Linear Regression, Ridge, Lasso
    * XGBoost
    * Random Forest

Hyperparameter Tuning:
- Bayes Search using BayesianOptimization Library
- MLFlow
- Early Stopping on validation fold

Feature Selection:
- Recursive Feature Elimination vs Backward Elimination => handle correlated feature better by recalculating importance after each removal (uses coef_ or feature_importances) vs relies on p-values or AIC


**Step 4. Offer Allocation Strategy**

A. Greedy Allocation

```
for offer in [0.40, 0.45, 0.50]:
    p_convert[offer] = model1.predict_proba(test_with_offer)[:, 1]
    expected_points[offer] = np.exp(model2.predict(test_with_offer)).clip(min=3000)
    price[offer] = 0.03 * (1 - offer)
    expected_revenue[offer] = p_convert[offer] * expected_points[offer] * price[offer]

assigned_offer = pd.DataFrame(expected_revenue).idxmax(axis=1)

```

B. Verify Constraints

- constraint 1: price floor enforcement

After the greedy assignment, compute the implied average price per point across predicted-transacting sessions. Since price_per_point is fixed per offer and you can predict who will convert, the formula is:
python# Weighted average price per point
predicted_conversions = assigned_offer.map(lambda o: p_convert[o][i])
weighted_avg_price = sum(p_convert_i * points_i * price_i) / sum(p_convert_i * points_i)
If this is below 0.016, implement the constraint repair pass: identify all sessions assigned 50%, rank them by revenue_delta = E[Rev|50%] - E[Rev|45%] ascending (i.e., the sessions where downgrading to 45% hurts the least), and flip the bottom-ranked ones to 45% until the price floor is satisfied


- constraint 2: offer coverage
    * ensure minimum of X% users get 40%, Y% users get 45% and Z% users get 50% discount
