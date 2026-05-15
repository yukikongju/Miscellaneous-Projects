"""
Bayesian hyperparameter search for classifiers and regressors.

Usage:
    from models.tuning import tune
    best_cfg = tune(cfg, X_train, y_train, X_val, y_val, n_iter=30, task="clf")

Caveats:
    - Uses a PredefinedSplit so val rows are never used for training during CV
      — do not shuffle the combined X passed to tune.
    - Each trial is logged to the active MLflow run; call inside
      mlflow.start_run() to avoid orphaned runs.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import mlflow
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import PredefinedSplit

from models.models import build_model


# ── Search spaces — one dict per model name ───────────────────────────────────

SEARCH_SPACES: dict[str, dict] = {
    "logistic_regression": {
        "C": Real(1e-3, 10.0, prior="log-uniform"),
    },
    "random_forest_classifier": {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 15),
    },
    "lgbm_classifier": {
        "num_leaves": Integer(15, 127),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "n_estimators": Integer(100, 500),
    },
    "xgb_classifier": {
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "n_estimators": Integer(100, 500),
    },
    "ridge": {
        "alpha": Real(1e-3, 100.0, prior="log-uniform"),
    },
    "lasso": {
        "alpha": Real(1e-4, 10.0, prior="log-uniform"),
    },
    "random_forest_regressor": {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 15),
    },
    "lgbm_regressor": {
        "num_leaves": Integer(15, 127),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "n_estimators": Integer(100, 500),
    },
    "xgb_regressor": {
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "n_estimators": Integer(100, 500),
    },
}


def _make_predefined_split(X_train, X_val):
    """
    PredefinedSplit tells BayesSearchCV to always validate on X_val,
    never re-split — correct since our split is member-level.
    """
    test_fold = np.concatenate(
        [
            np.full(len(X_train), -1),  # -1 = always in train
            np.zeros(len(X_val)),  #  0 = always in validation
        ]
    )
    return PredefinedSplit(test_fold)


def tune(
    cfg: dict[str, Any],
    X_train,
    y_train,
    X_val,
    y_val,
    n_iter: int = 30,
    task: str = "classification",  # "classification" | "regression"
) -> dict[str, Any]:
    """
    Run Bayesian search for cfg["name"], return best params dict.
    Logs every trial to MLflow.
    """
    name = cfg["name"]
    space = SEARCH_SPACES.get(name, {})

    if not space:
        print(f"No search space for '{name}' — skipping tuning.")
        return {}

    X_combined = (
        pd.concat([X_train, X_val]) if hasattr(X_train, "iloc") else np.vstack([X_train, X_val])
    )
    y_combined = np.concatenate([y_train, y_val])
    cv = _make_predefined_split(X_train, X_val)

    base_model = build_model(cfg)
    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"

    searcher = BayesSearchCV(
        estimator=base_model.estimator,
        search_spaces=space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit=False,
        random_state=42,
        n_jobs=-1,
    )
    searcher.fit(X_combined, y_combined)
    best_params = searcher.best_params_

    with mlflow.start_run(run_name=f"tuning_{name}", nested=True):
        mlflow.log_params({"model": name, **best_params})
        mlflow.log_metric("best_val_score", searcher.best_score_)

    return dict(best_params)
