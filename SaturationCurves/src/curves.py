"""Utilities for fitting and evaluating logistic saturation curves."""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def logistic_curve(x, L, k, x0):
    """Compute a logistic curve value.

    Args:
        x: Spend values.
        L: Curve ceiling (maximum attainable outcome).
        k: Growth steepness.
        x0: Midpoint spend where growth starts to slow.

    Returns:
        Logistic outcome values.
    """
    return L / (1 + np.exp(-k * (x - x0)))


def logistic_marginal_return(x, L, k, x0):
    """Compute the derivative of the logistic curve.

    Args:
        x: Spend values.
        L: Curve ceiling (maximum attainable outcome).
        k: Growth steepness.
        x0: Midpoint spend.

    Returns:
        Marginal outcome values.
    """
    exp_term = np.exp(-k * (x - x0))
    return (L * k * exp_term) / (1 + exp_term) ** 2


def get_logistic_params(func, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit a logistic-like function and return estimated parameters.

    Args:
        func: Curve function passed to `scipy.optimize.curve_fit`.
        x: Input spend values.
        y: Observed outcome values.

    Returns:
        Fitted parameters as `(L, k, x0)`.
    """
    p0 = [
        y.max(),
        0.0035,
        np.median(x),
    ]  # note: [L, k, x0] are the initial guess (ceiling, slope, midpoint)
    params, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
    L, k, x0 = params
    return (L, k, x0)


def get_logistic_asymptote_bound(x0, k, perc: float) -> float:
    """Convert a target logistic percentile into the corresponding spend value.

    Args:
        x0: Logistic midpoint parameter.
        k: Logistic steepness parameter.
        perc: Target percentile on the asymptote, between 0 and 1.

    Returns:
        Spend value at the requested percentile.
    """
    return round(x0 + (1.0 / k) * np.log(perc / (1.0 - perc)), 2)


def get_spend_metric_cutoff(
    df: pd.DataFrame,
    spend_col: str,
    metric_col: str,
    p_asymptote: float = 0.95,
    min_points: int = 12,
):
    """Estimate spend at which a metric reaches a saturation percentile.

    Args:
        df: Segment-level data containing spend and metric columns.
        spend_col: Name of the spend column.
        metric_col: Name of the outcome metric column.
        p_asymptote: Target saturation percentile on the fitted logistic curve.
        min_points: Minimum required row count before fitting.

    Returns:
        Estimated spend cutoff, or `np.nan` when fitting is not possible.
    """
    if len(df) < min_points or df[spend_col].nunique() < 5:
        return np.nan

    x = df[spend_col].to_numpy(dtype=float)
    y = df[metric_col].to_numpy(dtype=float)

    # Reasonable init + bounds for stability
    p0 = [max(y.max(), 1.0), 0.001, np.median(x)]
    bounds = ([0.0, 1e-9, x.min() * 0.5], [np.inf, np.inf, x.max() * 1.5])

    try:
        params, _ = curve_fit(logistic_curve, x, y, p0=p0, bounds=bounds, maxfev=40000)
        _, k, x0 = params
        if k <= 0:
            return np.nan
    except:
        return np.nan

    # Spend cutoff at chosen asymptote level (e.g., 95%)
    cutoff = get_logistic_asymptote_bound(x0, k, p_asymptote)

    return float(cutoff)
