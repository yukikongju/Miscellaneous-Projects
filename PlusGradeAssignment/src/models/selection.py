"""
Feature selection utilities: RFE and importance-based thresholding.

Usage:
    from models.selection import importance_selection, rfe_selection
    selected = importance_selection(model, X_train, threshold=0.01)

Caveats:
    - importance_selection normalizes importances; threshold is a fraction of
      total importance, not an absolute value.
    - rfe_selection is slow on large feature sets — prefer importance_selection
      for tree-based models that expose feature_importances_.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE

from models.models import BaseModel


def rfe_selection(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_features: int,
) -> list[str]:
    """
    Recursive Feature Elimination using the model's underlying estimator.
    Returns the list of selected feature names.
    """
    selector = RFE(
        estimator=model.estimator,
        n_features_to_select=n_features,
        step=1,
    )
    selector.fit(X_train, y_train)
    selected = X_train.columns[selector.support_].tolist()
    print(f"RFE selected {len(selected)} features: {selected}")
    return selected


def importance_selection(
    model: BaseModel,
    X_train: pd.DataFrame,
    threshold: float = 0.01,
) -> list[str]:
    """
    Drop features whose normalised importance is below threshold.
    Works for any model exposing feature_importances_ or coef_.
    """
    importances = model.feature_importances()
    if importances is None:
        raise ValueError(f"Model '{model.name}' does not expose feature importances.")

    norm = np.abs(importances) / np.abs(importances).sum()
    mask = norm >= threshold
    selected = X_train.columns[mask].tolist()
    dropped = X_train.columns[~mask].tolist()
    print(f"Importance selection — kept {len(selected)}, dropped {dropped}")
    return selected
