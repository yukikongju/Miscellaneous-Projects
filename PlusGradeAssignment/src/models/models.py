"""
Model abstraction layer: base classes, concrete implementations, and factory.

Usage:
    from models.models import build_model
    model = build_model({"name": "lgbm_classifier", "scale_pos_weight": 3.2})
    model.fit(X_train, y_train)

Caveats:
    - BaseRegressor.predict clips predictions to MIN_POINTS; predictions below
      that threshold are replaced silently.
    - All concrete classes self-register via the @register decorator in
      registry.py — do not instantiate directly if using the factory.
"""

from __future__ import annotations
import numpy as np
from typing import Any

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from models.registry import register, MODEL_REGISTRY
from models.estimators import (
    make_logistic_regression,
    make_naive_bayes,
    make_random_forest_classifier,
    make_random_forest_regressor,
    make_lgbm_classifier,
    make_lgbm_regressor,
    make_xgb_classifier,
    make_xgb_regressor,
    make_linear_regression,
    make_ridge,
    make_lasso,
)
from config import RANDOM_SEED, MIN_POINTS


# ── Base ──────────────────────────────────────────────────────────────────────


class BaseModel:
    def __init__(self, cfg: dict[str, Any]):
        self.name = cfg["name"]
        self.cfg = cfg
        self.estimator = self._build_estimator(cfg)

    def _build_estimator(self, cfg):
        raise NotImplementedError

    def fit(self, X, y, eval_set=None):
        raise NotImplementedError

    def feature_importances(self):
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        if hasattr(self.estimator, "coef_"):
            coef = self.estimator.coef_
            return coef[0] if coef.ndim > 1 else coef
        return None


class BaseClassifier(BaseModel):
    def fit(self, X, y, eval_set=None):
        kwargs = {}
        if eval_set and isinstance(
            self.estimator, (XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor)
        ):
            kwargs["eval_set"] = eval_set
            if isinstance(self.estimator, (XGBClassifier, XGBRegressor)):
                kwargs["verbose"] = False
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


class BaseRegressor(BaseModel):
    def fit(self, X, y, eval_set=None):
        kwargs = {}
        if eval_set and isinstance(
            self.estimator, (XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor)
        ):
            kwargs["eval_set"] = eval_set
            if isinstance(self.estimator, (XGBClassifier, XGBRegressor)):
                kwargs["verbose"] = False
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.estimator.predict(X).clip(min=MIN_POINTS)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register(MODEL_REGISTRY, "logistic_regression")
class LogisticRegressionModel(BaseClassifier):
    def _build_estimator(self, cfg):
        return make_logistic_regression(cfg)


@register(MODEL_REGISTRY, "naive_bayes")
class NaiveBayesModel(BaseClassifier):
    def _build_estimator(self, cfg):
        return make_naive_bayes(cfg)


@register(MODEL_REGISTRY, "random_forest_classifier")
class RandomForestClassifierModel(BaseClassifier):
    def _build_estimator(self, cfg):
        return make_random_forest_classifier(cfg)


@register(MODEL_REGISTRY, "lgbm_classifier")
class LGBMClassifierModel(BaseClassifier):
    def _build_estimator(self, cfg):
        return make_lgbm_classifier(cfg)


@register(MODEL_REGISTRY, "xgb_classifier")
class XGBClassifierModel(BaseClassifier):
    def _build_estimator(self, cfg):
        return make_xgb_classifier(cfg)


# ── Regressors ────────────────────────────────────────────────────────────────


@register(MODEL_REGISTRY, "linear_regression")
class LinearRegressionModel(BaseRegressor):
    def _build_estimator(self, cfg):
        return make_linear_regression(cfg)


@register(MODEL_REGISTRY, "ridge")
class RidgeModel(BaseRegressor):
    def _build_estimator(self, cfg):
        return make_ridge(cfg)


@register(MODEL_REGISTRY, "lasso")
class LassoModel(BaseRegressor):
    def _build_estimator(self, cfg):
        return make_lasso(cfg)


@register(MODEL_REGISTRY, "random_forest_regressor")
class RandomForestRegressorModel(BaseRegressor):
    def _build_estimator(self, cfg):
        return make_random_forest_regressor(cfg)


@register(MODEL_REGISTRY, "lgbm_regressor")
class LGBMRegressorModel(BaseRegressor):
    def _build_estimator(self, cfg):
        return make_lgbm_regressor(cfg)


@register(MODEL_REGISTRY, "xgb_regressor")
class XGBRegressorModel(BaseRegressor):
    def _build_estimator(self, cfg):
        return make_xgb_regressor(cfg)


# ── Factory ───────────────────────────────────────────────────────────────────


def build_model(cfg: dict[str, Any]) -> BaseModel:
    name = cfg.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"'{name}' not found. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](cfg)
