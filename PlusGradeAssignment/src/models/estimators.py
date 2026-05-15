"""
Factory functions that return configured sklearn-compatible estimators.

Usage:
    from models.estimators import make_lgbm_classifier
    est = make_lgbm_classifier({"n_estimators": 300, "learning_rate": 0.05})

Caveats:
    - All make_* functions read hyperparameters from a config dict with
      sensible defaults; unknown keys are silently ignored.
    - RANDOM_SEED is injected automatically — do not pass random_state manually.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from config import RANDOM_SEED


def make_logistic_regression(cfg: dict):
    return LogisticRegression(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 1000),
        class_weight=cfg.get("class_weight", "balanced"),
        random_state=RANDOM_SEED,
    )


def make_naive_bayes(cfg: dict):
    return GaussianNB(var_smoothing=cfg.get("var_smoothing", 1e-9))


def make_random_forest_classifier(cfg: dict):
    return RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 100),
        max_depth=cfg.get("max_depth", None),
        class_weight=cfg.get("class_weight", "balanced"),
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def make_random_forest_regressor(cfg: dict):
    return RandomForestRegressor(
        n_estimators=cfg.get("n_estimators", 100),
        max_depth=cfg.get("max_depth", None),
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def make_lgbm_classifier(cfg: dict):
    return LGBMClassifier(
        scale_pos_weight=cfg.get("scale_pos_weight", 1.0),
        num_leaves=cfg.get("num_leaves", 31),
        learning_rate=cfg.get("learning_rate", 0.05),
        n_estimators=cfg.get("n_estimators", 300),
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )


def make_lgbm_regressor(cfg: dict):
    return LGBMRegressor(
        objective=cfg.get("objective", "regression"),
        num_leaves=cfg.get("num_leaves", 31),
        learning_rate=cfg.get("learning_rate", 0.05),
        n_estimators=cfg.get("n_estimators", 300),
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=-1,
    )


def make_xgb_classifier(cfg: dict):
    return XGBClassifier(
        scale_pos_weight=cfg.get("scale_pos_weight", 1.0),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        n_estimators=cfg.get("n_estimators", 300),
        eval_metric="auc",
        random_state=RANDOM_SEED,
        verbosity=0,
    )


def make_xgb_regressor(cfg: dict):
    return XGBRegressor(
        objective=cfg.get("objective", "reg:squarederror"),
        max_depth=cfg.get("max_depth", 6),
        learning_rate=cfg.get("learning_rate", 0.05),
        n_estimators=cfg.get("n_estimators", 300),
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        verbosity=0,
    )


def make_linear_regression(cfg: dict):
    return LinearRegression()


def make_ridge(cfg: dict):
    return Ridge(alpha=cfg.get("alpha", 1.0))


def make_lasso(cfg: dict):
    return Lasso(alpha=cfg.get("alpha", 1.0))
