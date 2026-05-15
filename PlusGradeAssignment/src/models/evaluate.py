"""
Evaluation metrics for classifiers and regressors.

Usage:
    from models.evaluate import evaluate_classifier, evaluate_regressor
    metrics = evaluate_classifier(model, X_val, y_val, label="val")

Caveats:
    - evaluate_classifier_by_segment assumes binary segment values (0 and 1).
    - RMSE/MAE from evaluate_regressor are on the raw point scale; only
      compare across models trained on the same target.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score


def evaluate_classifier(model, X, y, label: str = "") -> dict:
    preds = model.predict_proba(X)[:, 1]
    return {"label": label, "auc": round(roc_auc_score(y, preds), 4)}


def evaluate_classifier_by_segment(model, X, y, segment: pd.Series, label: str = "") -> dict:
    results = {}
    for val in [0, 1]:
        mask = segment == val
        auc = roc_auc_score(y[mask], model.predict_proba(X[mask])[:, 1])
        results[f"{label}_seg{val}_auc"] = round(auc, 4)
    return results


def evaluate_regressor(model, X, y: np.ndarray, label: str = "") -> dict:
    preds = model.predict(X)
    return {
        "label": label,
        "rmse": round(np.sqrt(mean_squared_error(y, preds)), 2),
        "mae": round(mean_absolute_error(y, preds), 2),
        "r2": round(r2_score(y, preds), 4),
    }
